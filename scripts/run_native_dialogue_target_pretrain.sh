#!/usr/bin/env bash
set -euo pipefail

if [[ -f /venv/main/bin/activate ]]; then
  source /venv/main/bin/activate
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ -f "$SCRIPT_DIR/train_native_dialogue_lm.py" ]]; then
  DEFAULT_ROOT="$SCRIPT_DIR"
else
  DEFAULT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
ROOT="${ANIMA_TRAIN_ROOT:-$DEFAULT_ROOT}"
cd "$ROOT"
read -r STEPS GLOBAL_BATCH MICRO_BATCH < <("$PYTHON_BIN" -c '
from measurement.native_dialogue_registry import NATIVE_DIALOGUE_SPEC
s = NATIVE_DIALOGUE_SPEC["native_dialogue5"]
print(s["pretrain_steps"], s["global_batch"], s["micro_batch"])
')
if (( GLOBAL_BATCH % MICRO_BATCH != 0 )); then
  echo "registered global batch must be divisible by target micro batch" >&2
  exit 2
fi
GRAD_ACCUM=$((GLOBAL_BATCH / MICRO_BATCH))

"$PYTHON_BIN" train_native_dialogue_lm.py \
  --data-manifest data-target/manifest.json \
  --output-dir target \
  --preset target \
  --steps "$STEPS" \
  --batch-size "$MICRO_BATCH" \
  --grad-accum "$GRAD_ACCUM" \
  --lr 3e-4 \
  --save-every 1000 \
  --log-every 100 \
  --validation-batches 8 \
  --device cuda \
  --response-only-fraction 0
