#!/usr/bin/env bash
set -euo pipefail

cd PaddleOCR
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
mkdir -p ../logs

echo "== stage 1: PP-OCRv5 server rec, width=${S1_IMAGE_WIDTH:-640} =="
python -m paddle.distributed.launch --gpus "$CUDA_VISIBLE_DEVICES" \
  tools/train.py -c configs/rec/aistudio_ppocrv5_server_rec_s1.yml \
  2>&1 | tee ../logs/train_strong_s1_$(date +%Y%m%d_%H%M%S).log

echo "== stage 2: continue from stage 1 best, width=${S2_IMAGE_WIDTH:-960} =="
python -m paddle.distributed.launch --gpus "$CUDA_VISIBLE_DEVICES" \
  tools/train.py -c configs/rec/aistudio_ppocrv5_server_rec_s2.yml \
  2>&1 | tee ../logs/train_strong_s2_$(date +%Y%m%d_%H%M%S).log
