import argparse
from pathlib import Path

import yaml


def set_if_present(obj, key, value):
    if isinstance(obj, dict) and key in obj:
        obj[key] = value


def recursive_set(obj, key, value):
    if isinstance(obj, dict):
        if key in obj:
            obj[key] = value
        for v in obj.values():
            recursive_set(v, key, value)
    elif isinstance(obj, list):
        for v in obj:
            recursive_set(v, key, value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--output-config", required=True)
    parser.add_argument("--train-data-dir", default="./train_data")
    parser.add_argument("--save-model-dir", default="./output/aistudio_ppocrv5_server")
    parser.add_argument("--pretrained-model", default="./pretrain/PP-OCRv5_server_rec_pretrained.pdparams")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--max-text-length", type=int, default=100)
    parser.add_argument("--eval-step", type=int, default=1000)
    args = parser.parse_args()

    with open(args.base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    cfg.setdefault("Global", {})
    cfg["Global"]["epoch_num"] = args.epochs
    cfg["Global"]["save_model_dir"] = args.save_model_dir
    cfg["Global"]["pretrained_model"] = args.pretrained_model
    cfg["Global"]["eval_batch_step"] = [0, args.eval_step]
    cfg["Global"]["save_epoch_step"] = 1
    cfg["Global"]["max_text_length"] = args.max_text_length
    cfg["Global"]["cal_metric_during_train"] = True

    if "Train" in cfg and "dataset" in cfg["Train"]:
        cfg["Train"]["dataset"]["data_dir"] = args.train_data_dir
        cfg["Train"]["dataset"]["label_file_list"] = ["./train_data/train_list.txt"]
    if "Eval" in cfg and "dataset" in cfg["Eval"]:
        cfg["Eval"]["dataset"]["data_dir"] = args.train_data_dir
        cfg["Eval"]["dataset"]["label_file_list"] = ["./train_data/val_list.txt"]

    if "Train" in cfg and "loader" in cfg["Train"]:
        cfg["Train"]["loader"]["batch_size_per_card"] = args.train_batch_size
        cfg["Train"]["loader"]["drop_last"] = True
        cfg["Train"]["loader"]["num_workers"] = 8
    if "Eval" in cfg and "loader" in cfg["Eval"]:
        cfg["Eval"]["loader"]["batch_size_per_card"] = args.eval_batch_size
        cfg["Eval"]["loader"]["drop_last"] = False
        cfg["Eval"]["loader"]["num_workers"] = 8

    recursive_set(cfg, "max_text_length", args.max_text_length)

    out = Path(args.output_config)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
