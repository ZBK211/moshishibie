#!/usr/bin/env bash
set -euo pipefail

python -m pip install -U pip wheel setuptools
python -m pip install -r requirements.txt

if ! python - <<'PY'
import paddle
assert paddle.device.is_compiled_with_cuda()
print("paddle", paddle.__version__)
print("gpu count", paddle.device.cuda.device_count())
PY
then
  echo "Paddle GPU is not available. Install paddlepaddle-gpu first, then rerun this script."
  echo "Example for CUDA 12.6:"
  echo "python -m pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/"
  exit 1
fi

if [ ! -d PaddleOCR ]; then
  git clone https://github.com/PaddlePaddle/PaddleOCR.git
fi

cd PaddleOCR
python -m pip install -r requirements.txt
python -m pip install -e .
cd ..

echo "setup ok"
