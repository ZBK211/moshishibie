#!/usr/bin/env bash
set -euo pipefail

cd PaddleOCR
mkdir -p pretrain

if [ ! -f pretrain/PP-OCRv5_server_rec_pretrained.pdparams ]; then
  wget -O pretrain/PP-OCRv5_server_rec_pretrained.pdparams \
    https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/PP-OCRv5_server_rec_pretrained.pdparams
fi

BASE_CONFIG="configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml"
if [ ! -f "$BASE_CONFIG" ]; then
  echo "Base config not found: $BASE_CONFIG"
  echo "Available OCRv5/SVTR rec configs:"
  find configs/rec -iname "*OCRv5*rec*.yml" -o -iname "*SVTR*rec*.yml"
  exit 1
fi

python ../scripts/make_ppocr_config.py \
  --base-config "$BASE_CONFIG" \
  --output-config configs/rec/aistudio_ppocrv5_server_rec.yml \
  --train-data-dir ./train_data \
  --save-model-dir ./output/aistudio_ppocrv5_server \
  --pretrained-model ./pretrain/PP-OCRv5_server_rec_pretrained.pdparams \
  --epochs "${EPOCHS:-80}" \
  --train-batch-size "${TRAIN_BATCH_SIZE:-256}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-256}" \
  --max-text-length 100 \
  --image-width "${IMAGE_WIDTH:-640}" \
  --eval-step "${EVAL_STEP:-1000}"

cd ..
echo "config ok"
