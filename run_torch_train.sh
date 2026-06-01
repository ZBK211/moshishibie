#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
mkdir -p logs output_torch_crnn results

torchrun --nproc_per_node=4 scripts/torch_ocr.py train \
  --work-dir work_torch \
  --output-dir output_torch_crnn \
  --epochs "${EPOCHS:-120}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --workers "${WORKERS:-8}" \
  --lr "${LR:-0.001}" \
  --image-h "${IMAGE_H:-48}" \
  --image-w "${IMAGE_W:-640}" \
  2>&1 | tee "logs/torch_train_$(date +%Y%m%d_%H%M%S).log"
