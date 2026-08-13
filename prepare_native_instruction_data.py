#!/usr/bin/env python3
"""Prepare balanced grounded instruction and dynamic-memory dialogue data."""
from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
from pathlib import Path

from measurement.native_dialogue_registry import NATIVE_DIALOGUE_SPEC, spec_sha256
from native_dialogue_lm import NativeDialogueTokenizer, checkpoint_sha256
from prepare_native_dialogue_data import (
    PANEL_PATH,
    is_panel_near_duplicate,
    panel_fingerprints,
    parquet_batches,
    sha256_file,
    validation_row,
)


def download_sources(source_dir: Path) -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    source = NATIVE_DIALOGUE_SPEC["native_dialogue3"]["source"]
    paths = {}
    for lang, filename in source["files"].items():
        paths[lang] = Path(hf_hub_download(
            repo_id=source["repo_id"],
            repo_type="dataset",
            revision=source["revision"],
            filename=filename,
            local_dir=source_dir / source["repo_id"].replace("/", "--"),
        ))
    return paths


def dialogue_token_count(tokenizer: NativeDialogueTokenizer, messages: list[dict]) -> int:
    count = 2  # <bos>, <eos>
    for message in messages:
        count += 1 + len(tokenizer.encode(message["content"] + "\n"))
    return count


def select_instructions(path: Path, lang: str, tokenizer: NativeDialogueTokenizer,
                        panel_rows, count: int, multiplier: int, maximum_tokens: int):
    candidate_count = count * multiplier
    candidates = []
    scanned = 0
    for batch in parquet_batches(path, ["id", "inputs", "targets", "dataset_name"]):
        for row in batch.to_pylist():
            scanned += 1
            user, answer = row.get("inputs"), row.get("targets")
            if not isinstance(user, str) or not isinstance(answer, str):
                continue
            user, answer = user.strip(), answer.strip()
            if not user or not answer or len(user) + len(answer) > 4000:
                continue
            identity = f"{lang}:{row.get('dataset_name')}:{row.get('id')}"
            score = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], "big")
            item = (-score, identity, user, answer)
            if len(candidates) < candidate_count:
                heapq.heappush(candidates, item)
            elif score < -candidates[0][0]:
                heapq.heapreplace(candidates, item)
    selected = []
    removed_panel = removed_length = 0
    for negative_score, identity, user, answer in sorted(candidates, key=lambda item: -item[0]):
        if is_panel_near_duplicate(user, panel_rows) or is_panel_near_duplicate(answer, panel_rows):
            removed_panel += 1
            continue
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ]
        if dialogue_token_count(tokenizer, messages) > maximum_tokens:
            removed_length += 1
            continue
        selected.append((identity, messages))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(f"only {len(selected)} valid {lang} instructions after scanning {scanned}")
    return selected, {
        "scanned": scanned,
        "selected": len(selected),
        "panel_removed": removed_panel,
        "length_removed": removed_length,
    }


PEOPLE = (
    "Mina", "Joon", "Sora", "Noah", "Ari", "Yuna", "Leo", "Hana", "Theo", "Nari",
    "Mira", "Jun", "Sara", "Ian", "Rina", "Evan", "Dara", "Minho", "Lina", "Owen",
)
PROJECTS = (
    "Aurora", "Nimbus", "Harbor", "Maple", "Comet", "Willow", "Atlas", "Cedar",
    "Meadow", "Lighthouse", "Pebble", "Falcon", "Lotus", "Summit", "Coral", "Spruce",
)
DAYS_EN = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday")
DAYS_KO = ("월요일", "화요일", "수요일", "목요일", "금요일", "토요일")
CITIES_EN = ("Seoul", "Busan", "Incheon", "Daejeon", "Daegu", "Jeju", "Suwon", "Gwangju")
CITIES_KO = ("서울", "부산", "인천", "대전", "대구", "제주", "수원", "광주")
RISKS_EN = ("schedule delay", "supplier change", "test coverage", "weather", "budget review")
RISKS_KO = ("일정 지연", "공급처 변경", "시험 범위", "날씨", "예산 검토")


def memory_messages(lang: str, index: int) -> list[dict]:
    person = PEOPLE[index % len(PEOPLE)]
    project = PROJECTS[(index // len(PEOPLE)) % len(PROJECTS)]
    day_index = (index // 7) % len(DAYS_EN)
    next_day_index = (day_index + 2 + index % 3) % len(DAYS_EN)
    city_index = (index // 11) % len(CITIES_EN)
    risk_index = (index // 13) % len(RISKS_EN)
    corrected = index % 2 == 1
    kind = (index // 2) % 3
    if lang == "en":
        day, next_day = DAYS_EN[day_index], DAYS_EN[next_day_index]
        city, risk = CITIES_EN[city_index], RISKS_EN[risk_index]
        if kind == 0:
            fact = f"{person} leads Project {project}, and its review is on {day}."
            question = f"Who leads Project {project}, and when is its review?"
            answer = f"{person} leads Project {project}; its review is on {day}."
            replacement = f"Correction: the review moved from {day} to {next_day}. When is the review now?"
            corrected_answer = f"The Project {project} review is now on {next_day}."
        elif kind == 1:
            fact = f"Project {project}'s meeting place is {city}, and {person} owns the notes."
            question = f"Where is Project {project}'s meeting, and who owns the notes?"
            answer = f"The meeting is in {city}, and {person} owns the notes."
            new_person = PEOPLE[(index + 5) % len(PEOPLE)]
            replacement = f"Update: {new_person} owns the notes now. Who owns them now?"
            corrected_answer = f"{new_person} owns the notes now."
        else:
            fact = f"The main risk for Project {project} is {risk}; {person} will monitor it."
            question = f"What is Project {project}'s main risk, and who monitors it?"
            answer = f"Its main risk is {risk}, monitored by {person}."
            new_risk = RISKS_EN[(risk_index + 2) % len(RISKS_EN)]
            replacement = f"Correction: the main risk is now {new_risk}. What is the current main risk?"
            corrected_answer = f"The current main risk is {new_risk}."
        first = f"Keep the following detail available for later: {fact}"
        acknowledgement = f"Noted. I will remember the Project {project} details."
    else:
        day, next_day = DAYS_KO[day_index], DAYS_KO[next_day_index]
        city, risk = CITIES_KO[city_index], RISKS_KO[risk_index]
        if kind == 0:
            fact = f"{person}가 {project} 프로젝트를 맡고 검토일은 {day}입니다."
            question = f"{project} 프로젝트 담당자와 검토일은 무엇인가요?"
            answer = f"담당자는 {person}이고 검토일은 {day}입니다."
            replacement = f"정정합니다. 검토일이 {day}에서 {next_day}로 바뀌었습니다. 지금 검토일은 언제인가요?"
            corrected_answer = f"현재 검토일은 {next_day}입니다."
        elif kind == 1:
            fact = f"{project} 프로젝트 회의 장소는 {city}이고 {person}가 회의록을 맡습니다."
            question = f"{project} 프로젝트 회의 장소와 회의록 담당자는 누구인가요?"
            answer = f"회의 장소는 {city}이고 회의록 담당자는 {person}입니다."
            new_person = PEOPLE[(index + 5) % len(PEOPLE)]
            replacement = f"변경 사항입니다. 이제 {new_person}가 회의록을 맡습니다. 지금 담당자는 누구인가요?"
            corrected_answer = f"현재 회의록 담당자는 {new_person}입니다."
        else:
            fact = f"{project} 프로젝트의 주요 위험은 {risk}이고 {person}가 확인합니다."
            question = f"{project} 프로젝트의 주요 위험과 확인 담당자는 무엇인가요?"
            answer = f"주요 위험은 {risk}이고 {person}가 확인합니다."
            new_risk = RISKS_KO[(risk_index + 2) % len(RISKS_KO)]
            replacement = f"정정합니다. 주요 위험은 이제 {new_risk}입니다. 현재 주요 위험은 무엇인가요?"
            corrected_answer = f"현재 주요 위험은 {new_risk}입니다."
        first = f"뒤에서 물어볼 정보를 저장해 두세요. {fact}"
        acknowledgement = f"알겠습니다. {project} 프로젝트 정보를 기억하겠습니다."
    messages = [
        {"role": "user", "content": first},
        {"role": "assistant", "content": acknowledgement},
    ]
    if corrected:
        messages.extend((
            {"role": "user", "content": replacement},
            {"role": "assistant", "content": corrected_answer},
        ))
    else:
        messages.extend((
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ))
    return messages


def atomic_jsonl_writers(output_dir: Path, names: list[str]):
    paths = {name: output_dir / name for name in names}
    temporary = {name: path.with_suffix(path.suffix + ".tmp") for name, path in paths.items()}
    handles = {name: temporary[name].open("w", encoding="utf-8") for name in names}
    return paths, temporary, handles


def finish_writers(paths, temporary, handles) -> None:
    for name, handle in handles.items():
        handle.close()
        os.replace(temporary[name], paths[name])


def write_dataset(output_dir: Path, tokenizer: NativeDialogueTokenizer, selected, panel_rows):
    spec = NATIVE_DIALOGUE_SPEC["native_dialogue3"]
    shards = spec["instruction_shards_per_language"]
    names = []
    for lang in ("en", "ko"):
        for index in range(shards):
            names.extend((f"instruction.{lang}.{index:03d}.train.jsonl",
                          f"instruction.{lang}.{index:03d}.validation.jsonl"))
        names.extend((f"memory.{lang}.train.jsonl", f"memory.{lang}.validation.jsonl"))
    paths, temporary, handles = atomic_jsonl_writers(output_dir, names)
    splits = {"train_general": [], "validation_general": [],
              "train_dialogue": [], "validation_dialogue": []}
    statistics = {"instruction": {}, "memory": {}}
    for lang in ("en", "ko"):
        counts = {"train": 0, "validation": 0}
        for item_index, (identity, messages) in enumerate(selected[lang]):
            shard = item_index % shards
            split = "validation" if validation_row(identity, spec["validation_percent"]) else "train"
            name = f"instruction.{lang}.{shard:03d}.{split}.jsonl"
            handles[name].write(json.dumps({"messages": messages}, ensure_ascii=False, separators=(",", ":")) + "\n")
            counts[split] += 1
        statistics["instruction"][lang] = counts
        memory_counts = {"train": 0, "validation": 0, "panel_removed": 0}
        for index in range(spec["memory_examples_per_language"]):
            messages = memory_messages(lang, index)
            if any(is_panel_near_duplicate(message["content"], panel_rows) for message in messages):
                memory_counts["panel_removed"] += 1
                continue
            if dialogue_token_count(tokenizer, messages) > spec["maximum_screen_tokens"]:
                raise RuntimeError("registered dynamic memory example exceeds the screen context")
            identity = f"memory:{lang}:{index}"
            split = "validation" if validation_row(identity, spec["validation_percent"]) else "train"
            name = f"memory.{lang}.{split}.jsonl"
            handles[name].write(json.dumps({"messages": messages}, ensure_ascii=False, separators=(",", ":")) + "\n")
            memory_counts[split] += 1
        statistics["memory"][lang] = memory_counts
    finish_writers(paths, temporary, handles)
    for lang in ("en", "ko"):
        for index in range(shards):
            splits["train_dialogue"].append(f"instruction.{lang}.{index:03d}.train.jsonl")
            splits["validation_dialogue"].append(f"instruction.{lang}.{index:03d}.validation.jsonl")
        splits["train_dialogue"].append(f"memory.{lang}.train.jsonl")
        splits["validation_dialogue"].append(f"memory.{lang}.validation.jsonl")
    return paths, splits, statistics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--panel", type=Path, default=PANEL_PATH)
    args = parser.parse_args()
    if not args.tokenizer.is_file():
        parser.error("screen tokenizer must exist")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = args.source_dir or args.output_dir / ".source"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_paths = download_sources(source_dir)
    tokenizer = NativeDialogueTokenizer.load(args.tokenizer)
    panel_rows = panel_fingerprints(args.panel)
    config = NATIVE_DIALOGUE_SPEC["native_dialogue3"]
    selected, selection_stats = {}, {}
    for lang in ("en", "ko"):
        selected[lang], selection_stats[lang] = select_instructions(
            source_paths[lang], lang, tokenizer, panel_rows,
            config["instruction_examples_per_language"],
            config["instruction_candidate_multiplier"],
            config["maximum_screen_tokens"],
        )
    paths, splits, statistics = write_dataset(args.output_dir, tokenizer, selected, panel_rows)
    source = config["source"]
    manifest = {
        "format": "anima_native_dialogue_data_v2",
        "profile": "instruction-screen",
        "spec_sha256": spec_sha256(),
        "panel_sha256": sha256_file(args.panel),
        "base_tokenizer_sha256": checkpoint_sha256(args.tokenizer),
        "splits": splits,
        "tokenizer_files": [],
        "source_files": [
            {"repo_id": source["repo_id"], "revision": source["revision"],
             "file": source["files"][lang], "license": source["license"],
             "sha256": sha256_file(source_paths[lang])}
            for lang in ("en", "ko")
        ],
        "statistics": {**statistics, "selection": selection_stats},
        "outputs": {
            name: {"size": path.stat().st_size, "sha256": sha256_file(path)}
            for name, path in paths.items()
        },
    }
    temporary = args.output_dir / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output_dir / "manifest.json")
    print(json.dumps({
        "selected": {lang: selection_stats[lang]["selected"] for lang in ("en", "ko")},
        "memory": {lang: sum(statistics["memory"][lang][key] for key in ("train", "validation"))
                   for lang in ("en", "ko")},
        "outputs": len(paths),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
