# H200 OCR Training Project

This repository keeps code and small config files only. Do not commit datasets, checkpoints, or large result archives.

Important: do not install Paddle, PaddleOCR, or PaddleX into existing PyTorch experiment environments such as `markllm`. Paddle can change `nvidia-*` packages and break PyTorch. Use the isolated `ocr_paddle` environment for Assignment 5.

## Server path

Current server project path:

```bash
/inspire/hdd/project/generate-content-identifier/czxs253130383/WaterMoE-main/moshishibie
```

Conda root:

```bash
/inspire/hdd/project/generate-content-identifier/czxs253130383/miniconda3
```

## Pull latest code

```bash
cd /inspire/hdd/project/generate-content-identifier/czxs253130383/WaterMoE-main/moshishibie
git pull
```

## Restore markllm if it was polluted

See:

```bash
cat restore_markllm_torch.md
```

Short version:

```bash
source /inspire/hdd/project/generate-content-identifier/czxs253130383/miniconda3/etc/profile.d/conda.sh
conda activate markllm
python -m pip uninstall -y paddlepaddle-gpu paddlepaddle paddleocr paddlex
python -m pip install --force-reinstall torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/cu128
```

## Create isolated PaddleOCR environment

Run on the CPU side if the GPU side cannot download or pip install. CPU and GPU share the disk, so the environment will be available to the GPU side afterward.

```bash
cd /inspire/hdd/project/generate-content-identifier/czxs253130383/WaterMoE-main/moshishibie
bash create_ocr_paddle_env.sh
source /inspire/hdd/project/generate-content-identifier/czxs253130383/miniconda3/etc/profile.d/conda.sh
conda activate ocr_paddle
```

Verify:

```bash
python - <<'PY'
import paddle
print("paddle:", paddle.__version__)
print("cuda:", paddle.device.is_compiled_with_cuda())
print("gpu count:", paddle.device.cuda.device_count())
PY
```

Expected on the GPU side: `cuda: True`, `gpu count: 4`.

## PaddleOCR strong route

This is the recommended high-score route. It fine-tunes PP-OCRv5 server recognition on the 50,000 training images.

Install PaddleOCR source once:

```bash
cd /inspire/hdd/project/generate-content-identifier/czxs253130383/WaterMoE-main/moshishibie
git clone https://github.com/PaddlePaddle/PaddleOCR.git || git clone https://gitee.com/paddlepaddle/PaddleOCR.git
cd PaddleOCR
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -m pip install -e .
cd ..
```

Prepare data and config:

```bash
bash run_download_data.sh
bash run_prepare_data.sh
EPOCHS=120 TRAIN_BATCH_SIZE=256 EVAL_BATCH_SIZE=256 bash run_make_config.sh
```

Start four-GPU training with nohup:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash nohup_train.sh
```

Watch progress:

```bash
bash status.sh
bash tail_latest_log.sh
```

After training finishes, export and infer:

```bash
CUDA_VISIBLE_DEVICES=0 bash nohup_infer.sh
bash tail_latest_log.sh
```

Final files to download or push back:

```text
results/submission_finetuned.csv
results/recognition_scores_finetuned.csv
results/h200_result_ppocrv5_server.tar.gz
```

If small result files can be pushed through git:

```bash
bash push_results.sh
```

## PyTorch fallback route

This is only a fallback baseline. It is not expected to beat the PaddleOCR fine-tuned model.

```bash
source /inspire/hdd/project/generate-content-identifier/czxs253130383/miniconda3/etc/profile.d/conda.sh
conda activate pmark310
cd /inspire/hdd/project/generate-content-identifier/czxs253130383/WaterMoE-main/moshishibie
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
bash run_download_data.sh
bash run_torch_prepare.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 EPOCHS=120 BATCH_SIZE=128 IMAGE_W=640 MODEL=transformer bash nohup_torch_train.sh
```

## Dataset commands

```bash
aistudio download --dataset 65911/SkLXRq6Q --local_dir data/raw/train_repo --max-workers 1
aistudio download --dataset 65911/51vG7A8E --local_dir data/raw/test_repo --max-workers 1
```

Training label file encoding is GBK. CSV fields are `name,value`.
