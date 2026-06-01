#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="${CONDA_ROOT:-/inspire/hdd/project/generate-content-identifier/czxs253130383/miniconda3}"
ENV_NAME="${ENV_NAME:-ocr_paddle}"
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

source "$CONDA_ROOT/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -n "$ENV_NAME" python=3.10 -y
fi

conda activate "$ENV_NAME"
python -m pip install -U pip wheel setuptools -i https://pypi.tuna.tsinghua.edu.cn/simple

if [ -f "$PROJECT_DIR/offline_wheels/paddlepaddle_gpu-3.3.1-cp310-cp310-linux_x86_64.whl" ]; then
  python -m pip install "$PROJECT_DIR/offline_wheels/paddlepaddle_gpu-3.3.1-cp310-cp310-linux_x86_64.whl"
elif [ -f "$PROJECT_DIR/wheels/paddlepaddle_gpu-3.3.1-cp310-cp310-linux_x86_64.whl" ]; then
  python -m pip install "$PROJECT_DIR/wheels/paddlepaddle_gpu-3.3.1-cp310-cp310-linux_x86_64.whl"
else
  python -m pip install paddlepaddle-gpu==3.3.1 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ \
    --trusted-host www.paddlepaddle.org.cn
fi

python -m pip install paddleocr paddlex -i https://pypi.tuna.tsinghua.edu.cn/simple

python - <<'PY'
import paddle
print(paddle.__version__)
print("cuda:", paddle.device.is_compiled_with_cuda())
print("gpu count:", paddle.device.cuda.device_count())
PY
