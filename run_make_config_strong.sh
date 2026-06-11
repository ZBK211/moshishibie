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
  find configs/rec -iname "*OCRv5*rec*.yml" -o -iname "*SVTR*rec*.yml"
  exit 1
fi

python ../scripts/make_ppocr_config.py \
  --base-config "$BASE_CONFIG" \
  --output-config configs/rec/aistudio_ppocrv5_server_rec_s1.yml \
  --train-data-dir ./train_data \
  --save-model-dir ./output/aistudio_ppocrv5_server_s1 \
  --pretrained-model ./pretrain/PP-OCRv5_server_rec_pretrained.pdparams \
  --epochs "${S1_EPOCHS:-100}" \
  --train-batch-size "${S1_TRAIN_BATCH_SIZE:-192}" \
  --eval-batch-size "${S1_EVAL_BATCH_SIZE:-192}" \
  --max-text-length "${MAX_TEXT_LENGTH:-100}" \
  --image-width "${S1_IMAGE_WIDTH:-640}" \
  --learning-rate "${S1_LR:-0.0005}" \
  --eval-step "${EVAL_STEP:-500}"

python ../scripts/make_ppocr_config.py \
  --base-config "$BASE_CONFIG" \
  --output-config configs/rec/aistudio_ppocrv5_server_rec_s2.yml \
  --train-data-dir ./train_data \
  --save-model-dir ./output/aistudio_ppocrv5_server_s2 \
  --pretrained-model "${S2_PRETRAINED_MODEL:-./output/aistudio_ppocrv5_server_s1/best_accuracy}" \
  --epochs "${S2_EPOCHS:-40}" \
  --train-batch-size "${S2_TRAIN_BATCH_SIZE:-128}" \
  --eval-batch-size "${S2_EVAL_BATCH_SIZE:-128}" \
  --max-text-length "${MAX_TEXT_LENGTH:-100}" \
  --image-width "${S2_IMAGE_WIDTH:-960}" \
  --learning-rate "${S2_LR:-0.00015}" \
  --eval-step "${EVAL_STEP:-500}"

cd ..
echo "strong configs ok"
