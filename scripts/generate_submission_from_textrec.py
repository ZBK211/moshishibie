import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def sort_key(path):
    stem = path.stem
    return int(stem) if stem.isdigit() else path.name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True, help="Exported PaddleOCR recognition inference model directory.")
    parser.add_argument("--test-image-dir", default="work/test_data/test_images")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--score-output", default="recognition_scores.csv")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from paddleocr import TextRecognition

    image_paths = sorted(
        [p for p in Path(args.test_image_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
        key=sort_key,
    )
    if len(image_paths) != 10000:
        print(f"Warning: expected 10000 test images, found {len(image_paths)}")

    score_path = Path(args.score_output)
    done = set()
    rows = []
    if args.resume and score_path.exists():
        old = pd.read_csv(score_path)
        rows = old.to_dict("records")
        done = set(old["new_name"].astype(str))

    recognizer = TextRecognition(model_dir=args.model_dir)
    for start in tqdm(range(0, len(image_paths), args.batch_size), desc="predict"):
        batch = [p for p in image_paths[start:start + args.batch_size] if p.name not in done]
        if not batch:
            continue
        results = recognizer.predict([str(p) for p in batch])
        by_name = {Path(r["input_path"]).name: r for r in results}
        for path in batch:
            r = by_name.get(path.name)
            text = "" if r is None else str(r.get("rec_text", ""))
            score = 0.0 if r is None else float(r.get("rec_score", 0.0))
            rows.append({"new_name": path.name, "value": text or " ", "rec_score": score})
            done.add(path.name)
        pd.DataFrame(rows).to_csv(score_path, index=False, encoding="utf-8")

    df = pd.DataFrame(rows)
    order = {p.name: i for i, p in enumerate(image_paths)}
    df["sort_key"] = df["new_name"].map(order)
    df = df.sort_values("sort_key").drop(columns=["sort_key"])
    df[["new_name", "value"]].to_csv(args.output, index=False, encoding="utf-8")
    print(f"Wrote {args.output}: {len(df)} rows")
    print(f"Mean confidence: {df['rec_score'].mean():.6f}")


if __name__ == "__main__":
    main()
