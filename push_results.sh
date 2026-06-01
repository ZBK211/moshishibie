#!/usr/bin/env bash
set -euo pipefail

if [ ! -f results/submission_finetuned.csv ]; then
  echo "Missing results/submission_finetuned.csv"
  exit 1
fi

python scripts/check_submission.py results/submission_finetuned.csv

git add results/submission_finetuned.csv results/recognition_scores_finetuned.csv logs/*.log README.md || true
git commit -m "add h200 ocr result" || echo "Nothing to commit"
git push
