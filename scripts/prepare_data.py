import argparse
import json
import os
import random
import shutil
import zipfile
from pathlib import Path

import pandas as pd


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
    best_n, best_dir = candidates[0]
    if best_n < expected_count * 0.95:
        raise RuntimeError(f"Best image dir {best_dir} has only {best_n} images")
    return best_dir


def link_or_copy(src, dst, copy_images):
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() or dst.is_file():
            dst.unlink()
        else:
            shutil.rmtree(dst)
    if copy_images:
        shutil.copytree(src, dst)
    else:
        os.symlink(src.resolve(), dst, target_is_directory=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--package-root", default=".", help="Directory containing data/train_images.zip etc.")
    parser.add_argument("--work-dir", default="work", help="Output work directory.")
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--copy-images", action="store_true", help="Copy images instead of symlink.")
    args = parser.parse_args()

    package_root = Path(args.package_root).resolve()
    data_dir = package_root / "data"
    work_dir = Path(args.work_dir).resolve()
    extract_dir = work_dir / "extracted"
    train_data_dir = work_dir / "train_data"
    test_data_dir = work_dir / "test_data"
    train_data_dir.mkdir(parents=True, exist_ok=True)
    test_data_dir.mkdir(parents=True, exist_ok=True)

    unzip_if_needed(data_dir / "train_images.zip", extract_dir / "train")
    unzip_if_needed(data_dir / "test_images.zip", extract_dir / "test")

    train_img_src = find_image_dir(extract_dir / "train", 50000)
    test_img_src = find_image_dir(extract_dir / "test", 10000)
    link_or_copy(train_img_src, train_data_dir / "train_images", args.copy_images)
    link_or_copy(test_img_src, test_data_dir / "test_images", args.copy_images)

    labels = pd.read_csv(data_dir / "train_label.csv", encoding="gbk")
    labels["name"] = labels["name"].astype(str)
    labels["value"] = labels["value"].astype(str).str.strip()
    labels = labels[labels["value"].str.len() > 0].copy()

    existing = set(p.name for p in (train_data_dir / "train_images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    labels = labels[labels["name"].isin(existing)].copy()

    rng = random.Random(args.seed)
    idx = list(range(len(labels)))
    rng.shuffle(idx)
    val_n = max(1, int(len(idx) * args.val_ratio))
    val_idx = set(idx[:val_n])

    train_rows, val_rows = [], []
    for i, row in enumerate(labels.itertuples(index=False)):
        line = f"train_images/{row.name}\t{row.value}"
        if i in val_idx:
            val_rows.append(line)
        else:
            train_rows.append(line)

    (train_data_dir / "train_list.txt").write_text("\n".join(train_rows) + "\n", encoding="utf-8")
    (train_data_dir / "val_list.txt").write_text("\n".join(val_rows) + "\n", encoding="utf-8")

    test_images = sorted(
        [p.name for p in (test_data_dir / "test_images").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=lambda x: int(Path(x).stem) if Path(x).stem.isdigit() else x,
    )
    (test_data_dir / "test_images.txt").write_text("\n".join(test_images) + "\n", encoding="utf-8")

    stats = {
        "train_images": len(existing),
        "test_images": len(test_images),
        "train_labels": len(train_rows),
        "val_labels": len(val_rows),
        "max_label_len": int(labels["value"].map(len).max()),
        "num_chars": len(set("".join(labels["value"].tolist()))),
        "train_data_dir": str(train_data_dir),
        "test_data_dir": str(test_data_dir),
    }
    (work_dir / "data_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
