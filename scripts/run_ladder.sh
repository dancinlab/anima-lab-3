#!/usr/bin/env bash
# Score the whole λ-ladder, in order, in ONE process.
#
# Running the three scorers as separate detached jobs makes them fight over the
# GPU and overwrite each other's log — this session lost results twice that way,
# and a stray pkill from one waiter killed another waiter's run. Sequential is
# slower and finishes.
#
# Copy to the host and run there (scripts/run_remote.sh does both).
set -u
cd "$(dirname "$0")" || exit 1
pkill -f 'python3 -u lambda4.py' 2>/dev/null
pkill -f 'python3 -u panel.py' 2>/dev/null
pkill -f 'python3 -u g_gates.py' 2>/dev/null
sleep 3
rm -f logs/ladder.log logs/panel_nat.json logs/g_gates_nat.json logs/lambda4.json
{
  echo "===== λ0/λ1 ====="
  LAMBDA_FAMILY=natural python3 -u panel.py logs/panel_nat.json
  echo "===== λ2/λ3 ====="
  LAMBDA_FAMILY=natural python3 -u g_gates.py logs/g_gates_nat.json
  echo "===== λ4 ====="
  python3 -u lambda4.py logs/lambda4.json
  echo "LADDER_DONE"
} > logs/ladder.log 2>&1 &
echo "launched pid=$!"
