# Restore markllm PyTorch environment

Do not install Paddle, PaddleOCR, or PaddleX in `markllm` again.
Use the isolated `ocr_paddle` environment for the OCR competition.

## 1. Check backup PyTorch environments first

Run this on the server:

```bash
source /inspire/hdd/project/generate-content-identifier/czxs253130383/miniconda3/etc/profile.d/conda.sh

for env in pmark310 reasonmark; do
  echo "===== $env ====="
  conda run -n "$env" python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
PY
done
```

If one of them prints `cuda: True` and `gpu count: 4`, use that environment for other PyTorch experiments while `markllm` is being repaired.

## 2. Try to repair markllm

```bash
source /inspire/hdd/project/generate-content-identifier/czxs253130383/miniconda3/etc/profile.d/conda.sh
conda activate markllm

python -m pip uninstall -y paddlepaddle-gpu paddlepaddle paddleocr paddlex
python -m pip install --force-reinstall torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

Verify:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
print("gpu count:", torch.cuda.device_count())
PY
```

Expected output: `cuda: True` and `gpu count: 4`.

Note: the original `markllm` torch build may have been an NVIDIA platform build such as `2.8.0a0+...nv25.06`. If the platform provides an environment snapshot or reset button, that is the cleanest exact restore. The commands above restore a standard public PyTorch CUDA wheel.
