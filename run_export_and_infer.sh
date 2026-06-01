#!/usr/bin/env bash
set -euo pipefail

cd PaddleOCR

python tools/export_model.py \
  -c configs/rec/aistudio_ppocrv5_server_rec.yml \
  -o Global.pretrained_model=./output/aistudio_ppocrv5_server/best_accuracy \
     Global.save_inference_dir=./inference/aistudio_ppocrv5_server_rec

cd ..

mkdir -p results
python scripts/generate_submission_from_textrec.py \
  --model-dir PaddleOCR/inference/aistudio_ppocrv5_server_rec \
  --test-image-dir work/test_data/test_images \
  --output results/submission_finetuned.csv \
  --score-output results/recognition_scores_finetuned.csv \
  --batch-size "${PRED_BATCH_SIZE:-512}" \
  --resume

python scripts/check_submission.py results/submission_finetuned.csv

tar -czf results/h200_result_ppocrv5_server.tar.gz \
  results/submission_finetuned.csv \
  results/recognition_scores_finetuned.csv \
  PaddleOCR/output/aistudio_ppocrv5_server/best_accuracy.pdparams \
  PaddleOCR/output/aistudio_ppocrv5_server/best_accuracy.states \
  PaddleOCR/inference/aistudio_ppocrv5_server_rec

echo "result files:"
ls -lh results
