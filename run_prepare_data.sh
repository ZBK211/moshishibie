#!/usr/bin/env bash
set -euo pipefail

python scripts/prepare_data.py --package-root . --work-dir work --val-ratio 0.05

ln -sfn "$(pwd)/work/train_data" PaddleOCR/train_data

cat work/data_stats.json
