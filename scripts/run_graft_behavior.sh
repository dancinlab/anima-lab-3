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
      tests/test_consciousness_intervention.py
    "$PYTHON_BIN" -m py_compile graft_behavior.py measurement/graft_behavior_gate.py pure.py trinity.py
    ;;
  smoke|full|language-preserved|phase-state|phase-state-repair)
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
    exec "$PYTHON_BIN" graft_behavior.py \
      --output measurement/graft_behavior_results.json \
      --checkpoint-dir checkpoints/graft_behavior
    ;;
  launch-smoke|launch-full|launch-language-preserved|launch-phase-state|launch-phase-state-repair)
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
    echo "usage: $0 setup|smoke|full|language-preserved|phase-state|phase-state-repair|launch-* [git-revision]" >&2
    exit 2
    ;;
esac
