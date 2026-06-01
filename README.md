# 中文场景文字识别 H200 训练项目

这个仓库只放代码，不放数据大文件。H200 服务器联网后会通过 AI Studio CLI 下载训练集和测试集。

## 服务器运行顺序

```bash
git clone <你的仓库地址>
cd <仓库目录>

conda create -n ocr_h200 python=3.10 -y
conda activate ocr_h200
```

先装 GPU 版 Paddle。根据服务器 CUDA 版本选择对应源，CUDA 12.6 示例：

```bash
python -m pip install paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

检查：

```bash
python - <<'PY'
import paddle
print(paddle.__version__)
print("cuda:", paddle.device.is_compiled_with_cuda())
print("gpu count:", paddle.device.cuda.device_count())
PY
```

如果服务器已有 PyTorch GPU 环境，优先走 PyTorch 四卡方案，不再下载 2GB 的 Paddle wheel。检查：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.device_count())"
```

若输出 `True` 且 GPU 数量为 4，直接执行。默认是四卡 `CNN + Transformer Encoder + CTC`，比 CRNN 更吃算力也更适合 H200：

```bash
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
bash run_download_data.sh
bash run_torch_prepare.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 EPOCHS=120 BATCH_SIZE=128 IMAGE_W=640 MODEL=transformer bash nohup_torch_train.sh
```

查看训练：

```bash
bash status.sh
bash tail_latest_log.sh
```

训练完成后生成提交：

```bash
bash nohup_torch_infer.sh
bash tail_latest_log.sh
```

最终提交文件：

```text
results/submission_torch_transformer.csv
```

下面是 PaddleOCR 路线。只有当 Paddle GPU 环境已可用时再走。

确认 `cuda: True` 后继续：

```bash
bash run_setup.sh
bash run_download_data.sh
bash run_prepare_data.sh
bash run_make_config.sh
bash nohup_train.sh
```

训练会在后台运行。查看状态：

```bash
bash status.sh
bash tail_latest_log.sh
```

训练完成后导出模型并生成提交文件，也建议后台跑：

```bash
bash nohup_infer.sh
bash tail_latest_log.sh
```

## 训练参数

默认：

- 模型：`PP-OCRv5_server_rec`
- 预训练：官方 `PP-OCRv5_server_rec_pretrained.pdparams`
- 训练轮数：80
- 最大文本长度：100
- 验证集比例：5%
- 多卡：默认 `CUDA_VISIBLE_DEVICES=0,1,2,3`

可通过环境变量调整：

```bash
EPOCHS=120 TRAIN_BATCH_SIZE=256 EVAL_BATCH_SIZE=256 bash run_make_config.sh
CUDA_VISIBLE_DEVICES=0,1,2,3 bash run_train.sh
```

## 结果文件

训练完成后需要拿回本地：

```text
results/submission_finetuned.csv
results/recognition_scores_finetuned.csv
results/h200_result_ppocrv5_server.tar.gz
```

如果服务器允许 push 小文件结果：

```bash
bash push_results.sh
```

模型大文件默认被 `.gitignore` 忽略，不建议直接 push 到普通 git 仓库。

## 数据集来源

```bash
aistudio download --dataset 65911/SkLXRq6Q --local_dir data/raw/train_repo --max-workers 1
aistudio download --dataset 65911/51vG7A8E --local_dir data/raw/test_repo --max-workers 1
```

训练标签为 GBK 编码，字段是 `name,value`。
