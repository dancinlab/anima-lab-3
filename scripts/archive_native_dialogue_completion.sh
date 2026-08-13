#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  archive_native_dialogue_completion.sh \
    --ssh-target USER@HOST --ssh-port PORT --remote-root PATH \
    --repo-id ORG/REPO [--step STEP] [--parameters COUNT] \
    [--poll-seconds SECONDS] [--staging-root PATH] [--once]

Wait for the registered native-dialogue training to finish, preserve the
completed checkpoint in Hugging Face, download that exact commit into a new
directory, and verify every archived SHA-256 digest.
EOF
}

SSH_TARGET=""
SSH_PORT=""
REMOTE_ROOT=""
REPO_ID=""
EXPECTED_STEP=45000
EXPECTED_PARAMETERS=303628504
POLL_SECONDS=60
STAGING_ROOT="${TMPDIR:-/tmp}/anima-native-dialogue-archive"
ONCE=0

while (($#)); do
  case "$1" in
    --ssh-target) SSH_TARGET="$2"; shift 2 ;;
    --ssh-port) SSH_PORT="$2"; shift 2 ;;
    --remote-root) REMOTE_ROOT="$2"; shift 2 ;;
    --repo-id) REPO_ID="$2"; shift 2 ;;
    --step) EXPECTED_STEP="$2"; shift 2 ;;
    --parameters) EXPECTED_PARAMETERS="$2"; shift 2 ;;
    --poll-seconds) POLL_SECONDS="$2"; shift 2 ;;
    --staging-root) STAGING_ROOT="$2"; shift 2 ;;
    --once) ONCE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$SSH_TARGET" || -z "$SSH_PORT" || -z "$REMOTE_ROOT" || -z "$REPO_ID" ]]; then
  usage >&2
  exit 2
fi
if [[ ! "$SSH_PORT" =~ ^[0-9]+$ || ! "$EXPECTED_STEP" =~ ^[0-9]+$ ||
      ! "$EXPECTED_PARAMETERS" =~ ^[0-9]+$ || ! "$POLL_SECONDS" =~ ^[0-9]+$ ||
      "$POLL_SECONDS" -lt 1 ]]; then
  echo "port, step, parameters, and poll-seconds must be positive integers" >&2
  exit 2
fi
if [[ ! "$SSH_TARGET" =~ ^[A-Za-z0-9._-]+@[A-Za-z0-9._-]+$ ||
      ! "$REMOTE_ROOT" =~ ^/[A-Za-z0-9._/-]+$ ||
      ! "$REPO_ID" =~ ^[A-Za-z0-9._-]+/[A-Za-z0-9._-]+$ ]]; then
  echo "ssh target, remote root, or repository id contains unsupported characters" >&2
  exit 2
fi

for command_name in ssh rsync python3 shasum hf secret; do
  command -v "$command_name" >/dev/null || {
    echo "required command is unavailable: $command_name" >&2
    exit 2
  }
done
secret check huggingface.token >/dev/null || {
  echo "secret key huggingface.token is unavailable" >&2
  exit 2
}

SSH_ARGS=(-o BatchMode=yes -o ConnectTimeout=20 -p "$SSH_PORT")
STEP_PADDED=$(printf '%06d' "$EXPECTED_STEP")
ARCHIVE_SUBPATH="checkpoints/step-${STEP_PADDED}"
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
RUN_ROOT="$STAGING_ROOT/runs/step-${STEP_PADDED}-${RUN_ID}"
SOURCE_DIR="$RUN_ROOT/source"
VERIFY_DIR="$RUN_ROOT/verified-download"
STATUS_FILE="$STAGING_ROOT/step-${STEP_PADDED}-status.json"
LOCK_DIR="$STAGING_ROOT/.step-${STEP_PADDED}.lock"

mkdir -p "$STAGING_ROOT"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  echo "another completion archive process already owns $LOCK_DIR" >&2
  exit 3
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

remote_status() {
  # The validated values below are intentionally expanded by the local shell.
  # shellcheck disable=SC2029
  ssh "${SSH_ARGS[@]}" "$SSH_TARGET" \
    "cd '$REMOTE_ROOT' && if pgrep -f '[t]rain_native_dialogue_lm.py' >/dev/null; then echo running; elif test -f 'target/train_summary_step_${EXPECTED_STEP}.json'; then echo complete; else echo stopped_without_summary; fi"
}

while true; do
  state=$(remote_status | tail -n 1)
  if [[ "$state" == "complete" ]]; then
    break
  fi
  if [[ "$state" == "stopped_without_summary" ]]; then
    echo "training stopped without the registered step-${EXPECTED_STEP} summary" >&2
    exit 4
  fi
  if [[ "$state" != "running" ]]; then
    echo "unexpected remote training state: $state" >&2
    exit 4
  fi
  echo "$(date -u +%FT%TZ) training is still running"
  if ((ONCE)); then
    exit 10
  fi
  sleep "$POLL_SECONDS"
done

mkdir -p "$SOURCE_DIR" "$VERIFY_DIR"

# The validated paths and numeric expectations are intentionally expanded locally.
# shellcheck disable=SC2029
REMOTE_VERIFY=$(ssh "${SSH_ARGS[@]}" "$SSH_TARGET" \
  "cd '$REMOTE_ROOT' && if test -x /venv/main/bin/python3; then REMOTE_PYTHON=/venv/main/bin/python3; else REMOTE_PYTHON=python3; fi && EXPECTED_STEP='$EXPECTED_STEP' EXPECTED_PARAMETERS='$EXPECTED_PARAMETERS' \"\$REMOTE_PYTHON\" - <<'PY'
import gc
import json
import os
from pathlib import Path

import torch
from conscious_lm import build_model_from_config

root = Path('target')
step = int(os.environ['EXPECTED_STEP'])
parameters = int(os.environ['EXPECTED_PARAMETERS'])
summary = json.loads((root / f'train_summary_step_{step}.json').read_text())
assert summary['steps'] == step
assert summary['parameters'] == parameters
assert summary['response_only'] is True
assert summary['source_mode'] == 'dialogue'

resume = torch.load(root / 'resume.pt', map_location='cpu', weights_only=False, mmap=True)
assert resume['step'] == step
model = build_model_from_config(resume['config'])
assert model.count_params() == parameters
model.load_state_dict(resume['model_state'], strict=True)
resume_tensors = len(resume['model_state'])
optimizer_states = len(resume['optimizer_state']['state'])
tokenizer_sha256 = resume['tokenizer_sha256']
del model, resume
gc.collect()

final = torch.load(root / 'final.pt', map_location='cpu', weights_only=False, mmap=True)
assert final['step'] == step
model = build_model_from_config(final['config'])
assert model.count_params() == parameters
model.load_state_dict(final['model_state'], strict=True)
assert len(final['model_state']) == resume_tensors
del model, final

error_log = Path('target-answer-supervisor.err.log')
assert error_log.is_file() and error_log.stat().st_size == 0
print(json.dumps({
    'step': step,
    'parameters': parameters,
    'model_tensors': resume_tensors,
    'optimizer_states': optimizer_states,
    'tokenizer_sha256': tokenizer_sha256,
    'initial_validation_ce': summary['initial_validation_ce'],
    'final_validation_ce': summary['final_validation_ce'],
    'validation_descended': summary['validation_descended'],
}, sort_keys=True))
PY")

REMOTE_FILES=(
  target/resume.pt
  target/final.pt
  target/tokenizer.json
  target/manifest.json
  target/train_summary.json
  "target/train_summary_step_${EXPECTED_STEP}.json"
  target-answer-supervisor.log
  target-answer-supervisor.err.log
)
for remote_file in "${REMOTE_FILES[@]}"; do
  rsync -a --partial -e "ssh -o BatchMode=yes -o ConnectTimeout=20 -p $SSH_PORT" \
    "$SSH_TARGET:$REMOTE_ROOT/$remote_file" "$SOURCE_DIR/"
done

printf '%s\n' "$REMOTE_VERIFY" >"$SOURCE_DIR/source_verification.json"
(
  cd "$SOURCE_DIR"
  for archive_file in *; do
    [[ "$archive_file" == "SHA256SUMS" ]] && continue
    shasum -a 256 "$archive_file"
  done >SHA256SUMS
)

HF_TOKEN=$(secret get huggingface.token)
export HF_TOKEN
UPLOAD_JSON=$(hf upload "$REPO_ID" "$SOURCE_DIR" "$ARCHIVE_SUBPATH" \
  --private --commit-message "Preserve verified native dialogue step ${EXPECTED_STEP}" \
  --format json)
COMMIT_URL=$(printf '%s' "$UPLOAD_JSON" | python3 -c 'import json,sys; print(json.load(sys.stdin)["url"])')
COMMIT_SHA=${COMMIT_URL##*/}
if [[ ! "$COMMIT_SHA" =~ ^[0-9a-f]{40}$ ]]; then
  echo "Hugging Face returned an unexpected commit URL" >&2
  exit 5
fi

hf download "$REPO_ID" --revision "$COMMIT_SHA" \
  --include "$ARCHIVE_SUBPATH/*" --local-dir "$VERIFY_DIR" --force-download \
  --format quiet >/dev/null
unset HF_TOKEN
HF_TOKEN=""

DOWNLOADED_DIR="$VERIFY_DIR/$ARCHIVE_SUBPATH"
(
  cd "$DOWNLOADED_DIR"
  shasum -a 256 -c SHA256SUMS
)

SOURCE_VERIFY="$SOURCE_DIR/source_verification.json" \
COMMIT_SHA="$COMMIT_SHA" COMMIT_URL="$COMMIT_URL" REPO_ID="$REPO_ID" \
ARCHIVE_SUBPATH="$ARCHIVE_SUBPATH" STATUS_FILE="$STATUS_FILE" python3 - <<'PY'
import json
import os
from datetime import datetime, timezone
from pathlib import Path

source = json.loads(Path(os.environ['SOURCE_VERIFY']).read_text())
status = {
    'format': 'anima_native_dialogue_hf_archive_receipt_v1',
    'completed_at': datetime.now(timezone.utc).isoformat(),
    'repo_id': os.environ['REPO_ID'],
    'path_in_repo': os.environ['ARCHIVE_SUBPATH'],
    'commit_sha': os.environ['COMMIT_SHA'],
    'commit_url': os.environ['COMMIT_URL'],
    'downloaded_exact_commit': True,
    'all_sha256_verified': True,
    **source,
}
path = Path(os.environ['STATUS_FILE'])
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(status, ensure_ascii=False, indent=2, sort_keys=True) + '\n')
print(json.dumps(status, sort_keys=True))
PY
