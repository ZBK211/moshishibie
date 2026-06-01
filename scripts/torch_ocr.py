import argparse
import json
import math
import os
import random
import time
import zipfile
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm


def unzip_if_needed(zip_path, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".extracted"
    if marker.exists():
        return
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(out_dir)
    marker.write_text("ok", encoding="utf-8")


def find_image_dir(root, expected_count):
    candidates = []
    for p in root.rglob("*"):
        if p.is_dir() and "__MACOSX" not in str(p):
            n = len([x for x in p.iterdir() if x.suffix.lower() in {".jpg", ".jpeg", ".png"}])
            if n:
                candidates.append((n, p))
    candidates.sort(reverse=True)
    if not candidates:
        raise RuntimeError(f"No image directory found under {root}")
    n, p = candidates[0]
    if n < expected_count * 0.95:
        raise RuntimeError(f"Best image dir {p} has only {n} images")
    return p


def prepare_data(data_dir, work_dir, val_ratio=0.05, seed=2026):
    data_dir = Path(data_dir)
    work_dir = Path(work_dir)
    extract_dir = work_dir / "extracted"
    unzip_if_needed(data_dir / "train_images.zip", extract_dir / "train")
    unzip_if_needed(data_dir / "test_images.zip", extract_dir / "test")
    train_img_dir = find_image_dir(extract_dir / "train", 50000)
    test_img_dir = find_image_dir(extract_dir / "test", 10000)

    labels = pd.read_csv(data_dir / "train_label.csv", encoding="gbk")
    labels["name"] = labels["name"].astype(str)
    labels["value"] = labels["value"].astype(str).str.strip()
    labels = labels[labels["value"].str.len() > 0].copy()
    labels = labels[labels["name"].map(lambda x: (train_img_dir / x).exists())].copy()

    chars = sorted(set("".join(labels["value"].tolist())))
    vocab = {"<blank>": 0}
    for ch in chars:
        vocab[ch] = len(vocab)
    work_dir.mkdir(parents=True, exist_ok=True)
    (work_dir / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=2), encoding="utf-8")

    idx = list(range(len(labels)))
    random.Random(seed).shuffle(idx)
    val_n = max(1, int(len(idx) * val_ratio))
    val_set = set(idx[:val_n])
    rows = labels.to_dict("records")
    train_rows = [rows[i] for i in range(len(rows)) if i not in val_set]
    val_rows = [rows[i] for i in range(len(rows)) if i in val_set]
    pd.DataFrame(train_rows).to_csv(work_dir / "train.csv", index=False, encoding="utf-8")
    pd.DataFrame(val_rows).to_csv(work_dir / "val.csv", index=False, encoding="utf-8")

    stats = {
        "train": len(train_rows),
        "val": len(val_rows),
        "chars": len(chars),
        "max_len": int(labels["value"].map(len).max()),
        "train_img_dir": str(train_img_dir),
        "test_img_dir": str(test_img_dir),
    }
    (work_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


class ResizeKeepRatio:
    def __init__(self, height=48, width=320):
        self.height = height
        self.width = width

    def __call__(self, img):
        img = img.convert("L")
        w, h = img.size
        new_w = min(self.width, max(1, int(math.ceil(w * self.height / h))))
        img = img.resize((new_w, self.height), Image.BICUBIC)
        canvas = Image.new("L", (self.width, self.height), 255)
        canvas.paste(img, (0, 0))
        return canvas


class ToTensorNormalize:
    def __call__(self, img):
        data = torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes()))
        data = data.view(img.size[1], img.size[0]).float().div(255.0)
        data = data.unsqueeze(0)
        return (data - 0.5) / 0.5


class Compose:
    def __init__(self, ops):
        self.ops = ops

    def __call__(self, x):
        for op in self.ops:
            x = op(x)
        return x


class OCRDataset(Dataset):
    def __init__(self, csv_path, image_dir, vocab, image_h=48, image_w=320, augment=False):
        self.df = pd.read_csv(csv_path)
        self.image_dir = Path(image_dir)
        self.vocab = vocab
        self.augment = augment
        self.tf = Compose([ResizeKeepRatio(image_h, image_w), ToTensorNormalize()])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(self.image_dir / str(row["name"]))
        x = self.tf(img)
        text = str(row["value"])
        y = torch.tensor([self.vocab[c] for c in text if c in self.vocab], dtype=torch.long)
        return x, y, text, str(row["name"])


def collate(batch):
    xs, ys, texts, names = zip(*batch)
    xs = torch.stack(xs)
    lengths = torch.tensor([len(y) for y in ys], dtype=torch.long)
    ys = torch.cat(ys) if ys else torch.empty(0, dtype=torch.long)
    return xs, ys, lengths, texts, names


class CRNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 2, 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),
        )
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.mean(dim=2).permute(0, 2, 1)
        out, _ = self.rnn(feat)
        out = self.fc(out)
        return out


class TransformerOCR(nn.Module):
    def __init__(self, num_classes, d_model=512, nhead=8, num_layers=8, dim_feedforward=2048, max_seq_len=256):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.GELU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.GELU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.GELU(),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.GELU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, d_model, 3, 1, 1), nn.BatchNorm2d(d_model), nn.GELU(),
            nn.Conv2d(d_model, d_model, 3, 1, 1), nn.BatchNorm2d(d_model), nn.GELU(), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(d_model, d_model, 2, 1, 0), nn.BatchNorm2d(d_model), nn.GELU(),
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.pos = nn.Parameter(torch.zeros(1, max_seq_len, d_model))
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.fc = nn.Linear(d_model, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        feat = self.cnn(x)
        feat = feat.mean(dim=2).permute(0, 2, 1)
        if feat.size(1) > self.pos.size(1):
            raise RuntimeError(f"Sequence length {feat.size(1)} exceeds positional table {self.pos.size(1)}")
        feat = feat + self.pos[:, :feat.size(1), :]
        feat = self.encoder(feat)
        return self.fc(self.norm(feat))


def build_model(name, num_classes, image_w):
    if name == "crnn":
        return CRNN(num_classes)
    if name == "transformer":
        max_seq_len = max(256, image_w // 2)
        return TransformerOCR(num_classes, max_seq_len=max_seq_len)
    raise ValueError(f"Unknown model: {name}")


def ctc_decode(logits, inv_vocab):
    pred = logits.argmax(-1).cpu().numpy()
    out = []
    for seq in pred:
        chars, last = [], 0
        for i in seq:
            i = int(i)
            if i != 0 and i != last:
                chars.append(inv_vocab.get(i, ""))
            last = i
        out.append("".join(chars))
    return out


@torch.no_grad()
def evaluate(model, loader, device, inv_vocab):
    model.eval()
    total = correct = 0
    for xs, _, _, texts, _ in loader:
        xs = xs.to(device, non_blocking=True)
        preds = ctc_decode(model(xs), inv_vocab)
        for p, t in zip(preds, texts):
            total += 1
            correct += int(p == t)
    return correct / max(total, 1)


def train(args):
    distributed = "RANK" in os.environ
    if distributed:
        dist.init_process_group("nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        rank = dist.get_rank()
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        rank = 0

    work_dir = Path(args.work_dir)
    vocab = json.loads((work_dir / "vocab.json").read_text(encoding="utf-8"))
    inv_vocab = {v: k for k, v in vocab.items()}
    stats = json.loads((work_dir / "stats.json").read_text(encoding="utf-8"))
    train_ds = OCRDataset(work_dir / "train.csv", stats["train_img_dir"], vocab, args.image_h, args.image_w, True)
    val_ds = OCRDataset(work_dir / "val.csv", stats["train_img_dir"], vocab, args.image_h, args.image_w, False)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if distributed else None
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=train_sampler is None, sampler=train_sampler,
                              num_workers=args.workers, pin_memory=True, collate_fn=collate, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
                            num_workers=args.workers, pin_memory=True, collate_fn=collate)

    model = build_model(args.model, len(vocab), args.image_w).to(device)
    if distributed:
        model = DDP(model, device_ids=[local_rank])
    criterion = nn.CTCLoss(blank=0, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
    out_dir = Path(args.output_dir)
    if rank == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for epoch in range(1, args.epochs + 1):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        model.train()
        total_loss = 0.0
        start = time.time()
        iterator = tqdm(train_loader, disable=rank != 0, desc=f"epoch {epoch}")
        for xs, ys, y_lengths, _, _ in iterator:
            xs = xs.to(device, non_blocking=True)
            ys = ys.to(device, non_blocking=True)
            y_lengths = y_lengths.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=args.amp and device.type == "cuda"):
                logits = model(xs)
                log_probs = F.log_softmax(logits.float(), -1).permute(1, 0, 2)
                input_lengths = torch.full((xs.size(0),), logits.size(1), dtype=torch.long, device=device)
                loss = criterion(log_probs, ys, input_lengths, y_lengths)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            total_loss += float(loss.item())
            iterator.set_postfix(loss=f"{loss.item():.4f}")
        sched.step()
        acc = evaluate(model.module if distributed else model, val_loader, device, inv_vocab)
        if rank == 0:
            print(f"epoch={epoch} loss={total_loss/len(train_loader):.5f} val_exact={acc:.5f} time={time.time()-start:.1f}s", flush=True)
            state = {
                "model": (model.module if distributed else model).state_dict(),
                "vocab": vocab,
                "args": vars(args),
                "model_name": args.model,
                "val_exact": acc,
                "epoch": epoch,
            }
            torch.save(state, out_dir / "last.pt")
            if acc >= best:
                best = acc
                torch.save(state, out_dir / "best.pt")
    if distributed:
        dist.destroy_process_group()


@torch.no_grad()
def predict(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    vocab = ckpt["vocab"]
    inv_vocab = {v: k for k, v in vocab.items()}
    model_name = ckpt.get("model_name", ckpt.get("args", {}).get("model", "crnn"))
    model = build_model(model_name, len(vocab), args.image_w)
    model.load_state_dict(ckpt["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    tf = Compose([ResizeKeepRatio(args.image_h, args.image_w), ToTensorNormalize()])
    img_dir = Path(args.test_image_dir)
    paths = sorted([p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
                   key=lambda p: int(p.stem) if p.stem.isdigit() else p.name)
    rows = []
    for start in tqdm(range(0, len(paths), args.batch_size), desc="predict"):
        batch = paths[start:start + args.batch_size]
        xs = torch.stack([tf(Image.open(p)) for p in batch]).to(device)
        preds = ctc_decode(model(xs), inv_vocab)
        for p, text in zip(batch, preds):
            rows.append({"new_name": p.name, "value": text or "无"})
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    print(f"wrote {out} rows={len(rows)}")


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("prepare")
    pp.add_argument("--data-dir", default="data")
    pp.add_argument("--work-dir", default="work_torch")
    pp.add_argument("--val-ratio", type=float, default=0.05)
    pp.set_defaults(func=lambda a: prepare_data(a.data_dir, a.work_dir, a.val_ratio))
    tr = sub.add_parser("train")
    tr.add_argument("--work-dir", default="work_torch")
    tr.add_argument("--output-dir", default="output_torch_crnn")
    tr.add_argument("--epochs", type=int, default=80)
    tr.add_argument("--batch-size", type=int, default=128)
    tr.add_argument("--workers", type=int, default=8)
    tr.add_argument("--lr", type=float, default=1e-3)
    tr.add_argument("--image-h", type=int, default=48)
    tr.add_argument("--image-w", type=int, default=640)
    tr.add_argument("--model", default="transformer", choices=["transformer", "crnn"])
    tr.add_argument("--amp", action="store_true", default=True)
    tr.set_defaults(func=train)
    pr = sub.add_parser("predict")
    pr.add_argument("--checkpoint", default="output_torch_crnn/best.pt")
    pr.add_argument("--test-image-dir", required=True)
    pr.add_argument("--output", default="results/submission_torch_crnn.csv")
    pr.add_argument("--batch-size", type=int, default=512)
    pr.add_argument("--image-h", type=int, default=48)
    pr.add_argument("--image-w", type=int, default=640)
    pr.set_defaults(func=predict)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
