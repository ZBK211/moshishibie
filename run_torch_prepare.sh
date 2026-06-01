#!/usr/bin/env bash
set -euo pipefail

python scripts/torch_ocr.py prepare --data-dir data --work-dir work_torch --val-ratio 0.05
cat work_torch/stats.json
