#!/usr/bin/env bash
# Build the second natural-register family from a pinned public-domain source.
set -euo pipefail
cd /home/summer/anima-clm-pure

REVISION=2d16d39c774ef788069d63223d07e31e038c05df
SOURCE_NAME=gongu.jsonl
SOURCE_URL="https://huggingface.co/datasets/seyoungsong/Open-Korean-Historical-Corpus/resolve/${REVISION}/${SOURCE_NAME}"
SOURCE_SHA256=b689360119c1fc68c2dbcfdabc27589997e2a54a988051739c1644634db1785c
TRAIN_TARGET_BYTES=72885113
FRESH_TARGET_BYTES=29900000

mkdir -p data/source logs
SOURCE_PATH="data/source/${SOURCE_NAME}"
if ! test -f "$SOURCE_PATH" || ! printf '%s  %s\n' "$SOURCE_SHA256" "$SOURCE_PATH" | sha256sum -c - >/dev/null 2>&1; then
  curl --fail --location --retry 5 --output "${SOURCE_PATH}.part" "$SOURCE_URL"
  printf '%s  %s\n' "$SOURCE_SHA256" "${SOURCE_PATH}.part" | sha256sum -c -
  mv "${SOURCE_PATH}.part" "$SOURCE_PATH"
fi

python3 -u build_register_corpus.py \
  "$SOURCE_PATH" \
  data/corpus_natural_literary_ko_dedup.txt \
  data/corpus_natural_literary_fresh.txt \
  logs/literary_corpus_manifest.json \
  --train-target-bytes "$TRAIN_TARGET_BYTES" \
  --fresh-target-bytes "$FRESH_TARGET_BYTES" \
  --train-fraction 0.68 \
  --source-url "$SOURCE_URL" \
  --source-revision "$REVISION" \
  --source-license "source records Public Domain; aggregate CC BY-NC 4.0"

python3 -u corpus_regime.py data/corpus_natural_literary_ko_dedup.txt \
  > logs/literary_corpus_regime.txt
python3 -u natural_floors.py data/corpus_natural_literary_ko_dedup.txt \
  logs/literary_floors.json
echo LITERARY_CORPUS_DONE
