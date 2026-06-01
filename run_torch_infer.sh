#!/usr/bin/env bash
set -euo pipefail

mkdir -p logs results

python scripts/torch_ocr.py predict \
  --checkpoint output_torch_crnn/best.pt \
  --test-image-dir "$(python - <<'PY'
import json
print(json.load(open('work_torch/stats.json', encoding='utf-8'))['test_img_dir'])
PY
)" \
  --output results/submission_torch_crnn.csv \
  --batch-size "${PRED_BATCH_SIZE:-512}" \
  --image-h "${IMAGE_H:-48}" \
  --image-w "${IMAGE_W:-640}" \
  2>&1 | tee "logs/torch_infer_$(date +%Y%m%d_%H%M%S).log"

python scripts/check_submission.py results/submission_torch_crnn.csv
