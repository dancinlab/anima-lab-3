#!/usr/bin/env bash
# Score the λ-ladder, in order, in ONE process.
#
# Running the three scorers as separate detached jobs makes them fight over the
# GPU and overwrite each other's log — this session lost results twice that way,
# and a stray pkill from one waiter killed another waiter's run. Sequential is
# slower and finishes.
#
# With no arguments this preserves the canonical full-natural-family run.  A
# list of registered arm names creates an isolated, mergeable result shard:
#
#   bash run_ladder.sh n50drop37 n50drop37v
#
# The arm roster is forwarded unchanged to every scorer, so all λ grades cover
# exactly the same set.  The roster-derived tag keeps a partial run from
# overwriting the canonical full-family receipts.
set -euo pipefail
cd "$(dirname "$0")" || exit 1
mkdir -p logs

ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then
  TAG=nat
else
  TAG=$(IFS=_; printf '%s' "${ARMS[*]}")
  case "$TAG" in
    *[!A-Za-z0-9_-]*) echo "invalid arm name in result tag: $TAG" >&2; exit 2 ;;
  esac
fi

PANEL_OUT="logs/panel_${TAG}.json"
G_GATES_OUT="logs/g_gates_${TAG}.json"
LAMBDA4_OUT="logs/lambda4_${TAG}.json"
LADDER_LOG="logs/ladder_${TAG}.log"

# One host-wide ladder at a time.  Waiting jobs must fail visibly instead of
# killing an in-flight scorer or competing for GPU memory.
exec 9>logs/ladder.lock
if ! flock -n 9; then
  echo "another ladder run holds logs/ladder.lock" >&2
  exit 3
fi

rm -f "$PANEL_OUT" "$G_GATES_OUT" "$LAMBDA4_OUT" "$LADDER_LOG"
{
  echo "===== roster: ${ARMS[*]:-all-natural-arms} ====="
  echo "===== λ0/λ1 ====="
  LAMBDA_FAMILY=natural python3 -u panel.py "$PANEL_OUT" "${ARMS[@]}"
  echo "===== λ2/λ3 ====="
  LAMBDA_FAMILY=natural python3 -u g_gates.py "$G_GATES_OUT" "${ARMS[@]}"
  echo "===== λ4 ====="
  python3 -u lambda4.py "$LAMBDA4_OUT" "${ARMS[@]}"
  echo "LADDER_DONE"
} > "$LADDER_LOG" 2>&1 &
echo "launched pid=$! log=$LADDER_LOG"
