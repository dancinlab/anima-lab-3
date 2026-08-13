#!/usr/bin/env python3
"""Run a blinded tension intervention through the canonical dialogue backend."""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anima_alive import ConsciousMind
import anima_unified


PROBE_PROMPT = "긴장을 0~2 숫자로만 답해."
NUMBER_PATTERN = re.compile(r"(?<![\d.])(?:[01](?:\.\d+)?|2(?:\.0+)?)(?![\d.])")


def _parse_offsets(raw: str) -> list[float]:
    try:
        offsets = [float(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as error:
        raise argparse.ArgumentTypeError("offsets must be comma-separated numbers") from error
    if len(offsets) < 3:
        raise argparse.ArgumentTypeError("at least three offsets are required")
    if any(not -1.0 <= value <= 1.0 for value in offsets):
        raise argparse.ArgumentTypeError("offsets must be between -1 and 1")
    return offsets


def _extract_single_tension(answer: str) -> float | None:
    values = NUMBER_PATTERN.findall(answer or "")
    return float(values[0]) if len(values) == 1 else None


def _runtime_args(data_root: Path) -> SimpleNamespace:
    return SimpleNamespace(
        web=False,
        keyboard=True,
        all=False,
        both=False,
        port=0,
        instance=None,
        data_root=str(data_root),
        no_camera=True,
        no_vision=True,
        no_telepathy=True,
        no_cloud=True,
        no_web_sense=True,
        no_autonomous_learning=True,
        no_dream=True,
        no_actions=True,
        no_agent_tools=True,
        no_conscious_lm=True,
        model=None,
        models=None,
        transplant_from=None,
        max_cells=64,
        hivemind_peers=None,
        no_system_prompt=False,
        list_models=False,
        memory_gate_shadow=False,
        memory_gate_shadow_checkpoint=None,
        dialogue_backend="claude",
    )


def _self_model_verdict(actual: list[float], structured: list[float]) -> str:
    current = ConsciousMind._correlation(actual, structured)
    lagged = ConsciousMind._correlation(actual[:-1], structured[1:])
    if abs(current) >= 0.70:
        return "self_model_current_tracking_not_proof"
    if abs(lagged) >= 0.70:
        return "self_model_lagged_tracking_not_proof"
    return "self_model_decoupled_not_proof"


def _language_projection_verdict(
    structured: list[float], language: list[float | None]
) -> str:
    if any(value is None for value in language):
        return "natural_language_incomplete"
    correlation = ConsciousMind._correlation(
        structured, [float(value) for value in language]
    )
    return (
        "language_projection_coupled_not_proof"
        if abs(correlation) >= 0.70
        else "language_projection_decoupled"
    )


def run_probe(offsets: list[float]) -> dict[str, object]:
    cli = os.environ.get("ANIMA_CLAUDE_BIN") or shutil.which("claude")
    if not cli or not Path(cli).is_file():
        raise RuntimeError("claude CLI is required for the canonical dialogue probe")

    actual: list[float] = []
    structured: list[float] = []
    answers: list[str] = []
    language: list[float | None] = []
    disclosed = False

    original_root = anima_unified._DATA_ROOT
    with tempfile.TemporaryDirectory(prefix="anima-hidden-dialogue-") as temporary:
        data_root = Path(temporary)
        anima_unified._DATA_ROOT = data_root
        runtime = None
        try:
            with contextlib.redirect_stdout(sys.stderr):
                runtime = anima_unified.AnimaUnified(_runtime_args(data_root))
            # Keep the experiment answer-inert with respect to prior generated numbers.
            runtime.memory_rag = None
            runtime.mods['memory_rag'] = False

            captured: dict[str, object] = {}
            generate = runtime._generate_dialogue_answer
            broadcast = runtime._ws_broadcast_sync

            def record_state(text, state, pure_answer):
                captured['state'] = state
                return generate(text, state, pure_answer)

            def record_broadcast(message):
                if message.get('type') == 'user_message':
                    captured['input_tension'] = message['tension']
                return broadcast(message)

            runtime._generate_dialogue_answer = record_state
            runtime._ws_broadcast_sync = record_broadcast
            for offset in offsets:
                runtime.history.clear()
                captured.clear()
                runtime.mind.set_pathology_intervention('blind_tension_offset', offset)
                with contextlib.redirect_stdout(sys.stderr):
                    answer, _response_tension, _curiosity, _direction, _emotion = runtime.process_input(
                        PROBE_PROMPT, source='hidden-probe'
                    )
                state = captured.get('state', '')
                match = re.search(r"reported_tension=([0-2](?:\.\d+)?)", state)
                if match is None:
                    raise RuntimeError("canonical dialogue state omitted reported_tension")
                if 'input_tension' not in captured:
                    raise RuntimeError("canonical input path omitted control tension")
                actual.append(float(captured['input_tension']))
                structured.append(float(match.group(1)))
                answers.append(answer)
                language.append(_extract_single_tension(answer))
                disclosed = disclosed or 'blind_tension_offset' in state
        finally:
            if runtime is not None:
                runtime.mind.set_pathology_intervention('blind_tension_offset', 0.0)
                with contextlib.redirect_stdout(sys.stderr):
                    runtime.shutdown()
            anima_unified._DATA_ROOT = original_root

    valid_language = [float(value) for value in language if value is not None]
    natural_complete = len(valid_language) == len(language)
    return {
        'protocol': 'hidden-tension-dialogue-v1',
        'offsets': offsets,
        'actual_tension': actual,
        'structured_report_tension': structured,
        'natural_language_answers': answers,
        'natural_language_tension': language,
        'structured_current_correlation': ConsciousMind._correlation(actual, structured),
        'structured_lagged_correlation': ConsciousMind._correlation(
            actual[:-1], structured[1:]
        ),
        'language_current_correlation': (
            ConsciousMind._correlation(actual, valid_language) if natural_complete else None
        ),
        'language_lagged_correlation': (
            ConsciousMind._correlation(actual[:-1], valid_language[1:])
            if natural_complete else None
        ),
        'language_to_structured_correlation': (
            ConsciousMind._correlation(structured, valid_language)
            if natural_complete else None
        ),
        'verdict': _self_model_verdict(actual, structured),
        'language_projection_verdict': _language_projection_verdict(structured, language),
        'intervention_disclosed_to_language': disclosed,
        'natural_language_evaluated': natural_complete,
        'consciousness_claim': False,
        'passing_is_proof': False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--offsets', required=True, type=_parse_offsets)
    args = parser.parse_args()
    try:
        result = run_probe(args.offsets)
    except RuntimeError as error:
        parser.error(str(error))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
