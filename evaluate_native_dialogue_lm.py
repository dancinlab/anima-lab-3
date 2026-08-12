#!/usr/bin/env python3
"""Evaluate the self-trained dialogue model on the frozen bilingual panel."""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
from pathlib import Path

import torch

from conscious_lm import generate
from measurement.native_dialogue_registry import NATIVE_DIALOGUE_SPEC
from native_dialogue_lm import checkpoint_sha256, load_native_model


PANEL_PATH = Path(__file__).parent / "measurement" / "native_dialogue_panel.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def semantic_pass(text: str, turn: dict) -> bool:
    folded = text.casefold()
    groups = turn.get("required_groups", [])
    forbidden = turn.get("forbidden_terms", [])
    return all(any(term.casefold() in folded for term in group) for group in groups) and not any(
        term.casefold() in folded for term in forbidden
    )


def repeated_trigram_ratio(text: str) -> float:
    words = text.casefold().split()
    units = words if len(words) >= 3 else list("".join(words))
    triples = list(zip(units, units[1:], units[2:]))
    return 0.0 if not triples else 1.0 - len(set(triples)) / len(triples)


def prompt_echo_ratio(prompt: str, response: str) -> float:
    return difflib.SequenceMatcher(None, prompt.casefold(), response.casefold()).ratio()


def jaccard(left: str, right: str) -> float:
    a, b = set(left.casefold().split()), set(right.casefold().split())
    return 0.0 if not a or not b else len(a & b) / len(a | b)


def evaluate(model_dir: Path, panel_path: Path, device: str | None = None) -> dict:
    model, tokenizer, payload = load_native_model(model_dir, device=device)
    panel = json.loads(panel_path.read_text(encoding="utf-8"))
    rows = []
    passes = {"en": 0, "ko": 0}
    structural = True
    multiturn = True
    responses = []
    for item_index, item in enumerate(panel["items"]):
        history = []
        turn_rows = []
        item_pass = True
        for turn_index, turn in enumerate(item["turns"]):
            torch.manual_seed(20260813 + item_index * 10 + turn_index)
            prompt = tokenizer.format_prompt(
                turn["user"], "", history,
                max_tokens=model.block_size - NATIVE_DIALOGUE_SPEC["generation"]["max_new_tokens"],
            )
            output, _ = generate(
                model, prompt,
                max_new=NATIVE_DIALOGUE_SPEC["generation"]["max_new_tokens"],
                temperature=NATIVE_DIALOGUE_SPEC["generation"]["temperature"],
                device=str(next(model.parameters()).device),
                eos_token_id=tokenizer.ids["<eos>"],
                top_p=NATIVE_DIALOGUE_SPEC["generation"]["top_p"],
                repetition_penalty=NATIVE_DIALOGUE_SPEC["generation"]["repetition_penalty"],
            )
            answer = tokenizer.trim_response(output[len(prompt):])
            meaning = semantic_pass(answer, turn)
            generated_ids = output[len(prompt):]
            leaked = any(
                token_id in generated_ids
                for token, token_id in tokenizer.ids.items()
                if token not in {"<eos>", "<unk>", "<pad>"}
            )
            damage = "\ufffd" in answer
            shape = (
                bool(answer) and not leaked and not damage
                and repeated_trigram_ratio(answer) <= 0.35
                and prompt_echo_ratio(turn["user"], answer) <= 0.9
            )
            passes[item["lang"]] += int(meaning and shape)
            item_pass &= meaning and shape
            structural &= shape
            if turn.get("multiturn_final"):
                multiturn &= meaning and shape
            turn_rows.append({
                "user": turn["user"], "response": answer,
                "semantic_pass": meaning, "structural_pass": shape,
                "role_leak": leaked, "utf8_damage": damage,
                "repeated_trigram_ratio": repeated_trigram_ratio(answer),
                "prompt_echo_ratio": prompt_echo_ratio(turn["user"], answer),
            })
            responses.append(answer)
            history.extend((
                {"role": "user", "content": turn["user"]},
                {"role": "assistant", "content": answer},
            ))
        rows.append({"id": item["id"], "lang": item["lang"], "pass": item_pass, "turns": turn_rows})
    max_cross_jaccard = max(
        (jaccard(left, right) for index, left in enumerate(responses) for right in responses[index + 1:]),
        default=0.0,
    )
    structural &= max_cross_jaccard <= 0.85
    minimum = NATIVE_DIALOGUE_SPEC["thresholds"]["minimum_semantic_pass_per_language"]
    verdict = "PASS" if passes["en"] >= minimum and passes["ko"] >= minimum and structural and multiturn else "FAIL"
    return {
        "format": "anima_native_dialogue_v1_evaluation",
        "panel_sha256": sha256(panel_path),
        "checkpoint_sha256": checkpoint_sha256(model_dir / "final.pt"),
        "tokenizer_sha256": checkpoint_sha256(model_dir / "tokenizer.json"),
        "checkpoint_step": payload["step"],
        "verdict": verdict,
        "semantic_passes": passes,
        "structural_all": structural,
        "multiturn_final_all": multiturn,
        "max_cross_response_jaccard": max_cross_jaccard,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--panel", type=Path, default=PANEL_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device")
    args = parser.parse_args()
    result = evaluate(args.model_dir, args.panel, args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(args.output.name + ".tmp")
    temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({key: result[key] for key in ("verdict", "semantic_passes", "structural_all", "multiturn_final_all")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
