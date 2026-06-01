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
