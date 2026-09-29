#!/usr/bin/env bash
set -euo pipefail
python -m adar_bigru.cli \
  --input examples/example_targets.csv \
  --output outputs/example_candidates.csv \
  --num-sequences 4 \
  --efficiency 80 \
  --percentile 0.9 \
  --min-complementarity 0.90
