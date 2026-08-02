#!/usr/bin/env bash
# Pre-registered LAMBDA-2 replication on public-domain literary prose.
set -euo pipefail
SCRIPT_PATH=$(readlink -f "$0")
cd /home/summer/anima-clm-pure

if [ "${1:-}" != "--worker" ]; then
  nohup bash "$SCRIPT_PATH" --worker > logs/literary_train_launch.log 2>&1 &
  echo "launched pid=$! log=logs/literary_train_launch.log"
  exit 0
fi

exec 9>logs/gpu.lock
if ! flock -n 9; then
  echo "another research job holds logs/gpu.lock" >&2
  exit 3
fi

CORPUS=data/corpus_natural_literary_ko_dedup.txt
printf '%s  %s\n' 336e101a5b9737c2e12073b5562a06320c150b5a19655a8046b7c16e13ddff5e "$CORPUS" | sha256sum -c -
printf '%s  %s\n' 8e196165d525e15bc4b200e395953b19d6007acd0cb2c65746649dc4acb5cecd data/corpus_natural_literary_fresh.txt | sha256sum -c -

for path in checkpoints/arm_lit_drop37 checkpoints/arm_lit_drop37v; do
  if [ -e "$path" ]; then
    echo "refusing to overwrite pre-existing checkpoint directory: $path" >&2
    exit 4
  fi
done

COMMON=(--dim 384 --layers 6 --heads 6 --batch-size 32 --block-size 256
  --lr 3e-4 --max-cells 16 --val-bytes 262144 --eval-every 250
  --phase language --dropout 0.37 --steps 12000 --save-every 6000 --data "$CORPUS")

python3 -u train_conscious_lm.py "${COMMON[@]}" --seed 1337 \
  --checkpoint-dir checkpoints/arm_lit_drop37 > logs/arm_lit_drop37.log 2>&1
python3 -u train_conscious_lm.py "${COMMON[@]}" --seed 7331 \
  --checkpoint-dir checkpoints/arm_lit_drop37v > logs/arm_lit_drop37v.log 2>&1
echo "LITERARY_TRAIN_DONE $(date --iso-8601=seconds)" > logs/literary_train.done
