#!/usr/bin/env bash
# Canonical remote launcher for the GRAFT hidden-situation behavior experiment.
set -euo pipefail

ACTION=${1:-}
REVISION=${2:-main}
ROOT=${ANIMA_ROOT:-/workspace/anima-lab-3}
REPO_URL=${ANIMA_REPO_URL:-https://github.com/dancinlab/anima-lab-3.git}

# Vast's canonical PyTorch image exposes the managed environment at /venv/main,
# while older research hosts expose `python` directly. Resolve that difference
# once so every GRAFT action uses the same interpreter and installed packages.
if [ -f /venv/main/bin/activate ]; then
  # shellcheck disable=SC1091
  source /venv/main/bin/activate
fi
if [ -z "${PYTHON_BIN:-}" ]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python)
  elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=$(command -v python3)
  else
    echo "python runtime not found" >&2
    exit 127
  fi
fi

case "$ACTION" in
  setup)
    if [ ! -d "$ROOT/.git" ]; then
      git clone "$REPO_URL" "$ROOT"
    fi
    git -C "$ROOT" fetch origin main
    git -C "$ROOT" checkout --detach "$REVISION"
    "$PYTHON_BIN" -m pip install --quiet 'transformers>=4.51,<5' accelerate sentencepiece pytest
    cd "$ROOT"
    "$PYTHON_BIN" -m pytest -q tests/test_graft_behavior_gate.py tests/test_quantum_phase_readout.py \
      tests/test_consciousness_intervention.py tests/test_state_survival.py \
      tests/test_bridge_capacity.py tests/test_metacognition_gate.py tests/test_synergy_gate.py \
      tests/test_workspace_information_gate.py tests/test_recurrent_workspace.py \
      tests/test_workspace_gate.py
    "$PYTHON_BIN" -m py_compile graft_behavior.py state_survival.py measurement/graft_behavior_gate.py \
      measurement/state_survival_gate.py measurement/bridge_capacity_gate.py metacognition.py \
      measurement/metacognition_gate.py synergy.py measurement/synergy_gate.py \
      workspace_information.py measurement/workspace_information_gate.py workspace.py \
      measurement/workspace_gate.py pure.py trinity.py
    ;;
  smoke|full|language-preserved|phase-state|phase-state-repair|phase-state-bridge32|phase-state-bridge32-repair|state-map|meta1|synergy1|synergy1-repair|workspace-map|workspace1|workspace1-repair)
    cd "$ROOT"
    mkdir -p logs checkpoints/graft_behavior measurement
    exec 9>logs/gpu.lock
    if ! flock -n 9; then
      echo "another research job holds logs/gpu.lock" >&2
      exit 1
    fi
    if [ "$ACTION" = smoke ]; then
      exec "$PYTHON_BIN" graft_behavior.py --seeds 1337 --train-steps 20 \
        --output measurement/graft_behavior_smoke.json \
        --checkpoint-dir checkpoints/graft_behavior_smoke
    fi
    if [ "$ACTION" = language-preserved ]; then
      exec "$PYTHON_BIN" graft_behavior.py \
        --experiment graft_behavior_causality_language_preserved \
        --output measurement/graft_behavior_language_preserved_results.json \
        --checkpoint-dir checkpoints/graft_behavior_language_preserved
    fi
    if [ "$ACTION" = phase-state ]; then
      exec "$PYTHON_BIN" graft_behavior.py \
        --experiment graft_behavior_causality_phase_state \
        --output measurement/graft_behavior_phase_state_results.json \
        --checkpoint-dir checkpoints/graft_behavior_phase_state
    fi
    if [ "$ACTION" = phase-state-repair ]; then
      exec "$PYTHON_BIN" graft_behavior.py \
        --experiment graft_behavior_causality_phase_state_memory_control_repair \
        --output measurement/graft_behavior_phase_state_repair_results.json \
        --checkpoint-dir checkpoints/graft_behavior_phase_state_repair
    fi
    if [ "$ACTION" = phase-state-bridge32 ]; then
      exec "$PYTHON_BIN" graft_behavior.py \
        --experiment graft_behavior_causality_phase_state_bridge32 \
        --output measurement/graft_behavior_phase_state_bridge32_results.json \
        --checkpoint-dir checkpoints/graft_behavior_phase_state_bridge32
    fi
    if [ "$ACTION" = phase-state-bridge32-repair ]; then
      exec "$PYTHON_BIN" graft_behavior.py \
        --experiment graft_behavior_causality_phase_state_bridge32_memory_control_repair \
        --output measurement/graft_behavior_phase_state_bridge32_repair_results.json \
        --checkpoint-dir checkpoints/graft_behavior_phase_state_bridge32_repair
    fi
    if [ "$ACTION" = state-map ]; then
      "$PYTHON_BIN" state_survival.py
      exec "$PYTHON_BIN" measurement/state_survival_gate.py \
        measurement/state_survival_results.json measurement/state_survival_verdict.json
    fi
    if [ "$ACTION" = meta1 ]; then
      exec "$PYTHON_BIN" metacognition.py \
        --experiment meta1_bridge32_self_monitoring_noise_range_repair
    fi
    if [ "$ACTION" = synergy1 ]; then
      exec "$PYTHON_BIN" synergy.py --experiment synergy1_split_cue_modular_sum
    fi
    if [ "$ACTION" = synergy1-repair ]; then
      exec "$PYTHON_BIN" synergy.py \
        --experiment synergy1_split_cue_modular_sum_control_role_repair \
        --output measurement/synergy_results.json \
        --verdict measurement/synergy_verdict.json \
        --checkpoint-dir checkpoints/synergy1_control_role_repair
    fi
    if [ "$ACTION" = workspace-map ]; then
      exec "$PYTHON_BIN" workspace_information.py
    fi
    if [ "$ACTION" = workspace1 ]; then
      exec "$PYTHON_BIN" workspace.py \
        --experiment workspace1_recurrent_split_cue_integration \
        --output measurement/workspace_invalid_results.json \
        --verdict measurement/workspace_invalid_verdict.json \
        --checkpoint-dir checkpoints/workspace1
    fi
    if [ "$ACTION" = workspace1-repair ]; then
      exec "$PYTHON_BIN" workspace.py \
        --experiment workspace1_recurrent_split_cue_integration_control_seed_repair \
        --output measurement/workspace_results.json \
        --verdict measurement/workspace_verdict.json \
        --checkpoint-dir checkpoints/workspace1_control_seed_repair
    fi
    exec "$PYTHON_BIN" graft_behavior.py \
      --output measurement/graft_behavior_results.json \
      --checkpoint-dir checkpoints/graft_behavior
    ;;
  launch-smoke|launch-full|launch-language-preserved|launch-phase-state|launch-phase-state-repair|launch-phase-state-bridge32|launch-phase-state-bridge32-repair|launch-state-map|launch-meta1|launch-synergy1|launch-synergy1-repair|launch-workspace-map|launch-workspace1|launch-workspace1-repair)
    JOB=${ACTION#launch-}
    SESSION=graft-behavior-$JOB
    LOG="$ROOT/logs/graft_behavior_${JOB}.log"
    mkdir -p "$ROOT/logs"
    tmux kill-session -t "$SESSION" 2>/dev/null || true
    tmux new-session -d -s "$SESSION" \
      "bash '$0' '$JOB' '$REVISION' > '$LOG' 2>&1"
    echo "launched $SESSION -> $LOG"
    ;;
  *)
    echo "usage: $0 setup|smoke|full|language-preserved|phase-state|phase-state-repair|phase-state-bridge32|phase-state-bridge32-repair|state-map|meta1|synergy1|synergy1-repair|workspace-map|workspace1|workspace1-repair|launch-* [git-revision]" >&2
    exit 2
    ;;
esac
