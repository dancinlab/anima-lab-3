#!/usr/bin/env bash
set -euo pipefail
cd /home/summer/anima-clm-pure
LAMBDA_FAMILY=literary bash run_ladder.sh litdrop37 litdrop37v
