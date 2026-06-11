#!/usr/bin/env bash
set -euo pipefail

cd PaddleOCR

CONFIG="${STRONG_CONFIG:-configs/rec/aistudio_ppocrv5_server_rec_s2.yml}"
CKPT="${STRONG_CKPT:-./output/aistudio_ppocrv5_server_s2/best_accuracy}"
MODEL_NAME="${STRONG_MODEL_NAME:-aistudio_ppocrv5_server_s2}"

if [ ! -f "${CKPT}.pdparams" ]; then
  echo "Stage 2 checkpoint not found, falling back to stage 1."
  CONFIG="configs/rec/aistudio_ppocrv5_server_rec_s1.yml"
  CKPT="./output/aistudio_ppocrv5_server_s1/best_accuracy"
  MODEL_NAME="aistudio_ppocrv5_server_s1"
fi

python tools/export_model.py \
  -c "$CONFIG" \
  -o Global.pretrained_model="$CKPT" \
     Global.save_inference_dir="./inference/${MODEL_NAME}_rec"

cd ..

mkdir -p results
python scripts/generate_submission_from_textrec.py \
  --model-dir "PaddleOCR/inference/${MODEL_NAME}_rec" \
  --test-image-dir work/test_data/test_images \
  --output "results/submission_${MODEL_NAME}.csv" \
  --score-output "results/recognition_scores_${MODEL_NAME}.csv" \
  --batch-size "${PRED_BATCH_SIZE:-256}" \
  --resume

python scripts/check_submission.py "results/submission_${MODEL_NAME}.csv"

if [ -n "${TEACHER_SUBMISSION:-}" ] && [ -f "${TEACHER_SUBMISSION:-}" ]; then
  python scripts/fuse_submissions.py \
    --base "results/submission_${MODEL_NAME}.csv" \
    --patch "$TEACHER_SUBMISSION" \
    --output "results/submission_${MODEL_NAME}_teacher_fused.txt" \
    --output-delimiter tab
fi

tar -czf "results/h200_result_${MODEL_NAME}.tar.gz" \
  "results/submission_${MODEL_NAME}.csv" \
  "results/recognition_scores_${MODEL_NAME}.csv" \
  "PaddleOCR/${CKPT}.pdparams" \
  "PaddleOCR/${CKPT}.states" \
  "PaddleOCR/inference/${MODEL_NAME}_rec" \
  2>/dev/null || true

echo "result files:"
ls -lh results
