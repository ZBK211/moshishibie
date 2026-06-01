#!/usr/bin/env bash
set -euo pipefail

cd PaddleOCR
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

mkdir -p ../logs

python -m paddle.distributed.launch --gpus "$CUDA_VISIBLE_DEVICES" \
  tools/train.py -c configs/rec/aistudio_ppocrv5_server_rec.yml \
  2>&1 | tee ../logs/train_$(date +%Y%m%d_%H%M%S).log
