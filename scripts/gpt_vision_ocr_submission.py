import argparse
import base64
import csv
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


API_URL = "https://api.openai.com/v1/responses"


def sort_key(path):
    stem = path.stem
    return int(stem) if stem.isdigit() else path.name


def image_data_url(path):
    mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


def extract_text_from_response(data):
    if isinstance(data, dict) and isinstance(data.get("output_text"), str):
        return data["output_text"]

    chunks = []

    def walk(obj):
        if isinstance(obj, dict):
            if obj.get("type") in {"output_text", "text"} and isinstance(obj.get("text"), str):
                chunks.append(obj["text"])
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(data.get("output", data))
    return "\n".join(chunks)


def parse_json_output(text):
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", text, re.S)
        if not match:
            raise
        obj = json.loads(match.group(1))
    if isinstance(obj, dict):
        if "results" in obj:
            obj = obj["results"]
        else:
            obj = [{"new_name": k, "value": v} for k, v in obj.items()]
    return obj


def clean_value(value):
    value = "" if value is None else str(value)
    value = value.replace("\t", " ").replace("\r", " ").replace("\n", " ")
    return value.strip()


def read_existing(path):
    done = {}
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = csv.excel_tab if "\t" in sample.splitlines()[0] else csv.excel
        reader = csv.DictReader(f, dialect=dialect)
        for row in reader:
            name = row.get("new_name")
            value = clean_value(row.get("value", ""))
            if name and value:
                done[name] = value
    return done


def write_submission(path, image_paths, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["new_name", "value"])
        for image_path in image_paths:
            if image_path.name in rows:
                writer.writerow([image_path.name, rows[image_path.name]])


def write_final_submission(path, image_paths, rows):
    missing = [p.name for p in image_paths if p.name not in rows]
    if missing:
        raise RuntimeError(f"Submission is incomplete: {len(missing)} missing, first={missing[:10]}")
    write_submission(path, image_paths, rows)


def call_openai(batch, model, api_key, detail, timeout, retries):
    content = [
        {
            "type": "input_text",
            "text": (
                "You are an OCR engine for a Chinese scene text recognition benchmark. "
                "For each following image, read the visible text exactly. "
                "Return strict JSON only, as an array of objects with keys new_name and value. "
                "Preserve each filename exactly. Do not add explanations. "
                "If no text is readable, use an empty string."
            ),
        }
    ]
    for path in batch:
        content.append({"type": "input_text", "text": f"new_name: {path.name}"})
        content.append({"type": "input_image", "image_url": image_data_url(path), "detail": detail})

    body = {
        "model": model,
        "input": [{"role": "user", "content": content}],
        "temperature": 0,
    }
    payload = json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    last_error = None
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(API_URL, data=payload, headers=headers, method="POST")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
            parsed = parse_json_output(extract_text_from_response(data))
            result = {}
            for item in parsed:
                name = str(item.get("new_name", "")).strip()
                if name:
                    result[name] = clean_value(item.get("value", ""))
            return result
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = exc
            sleep = min(60, 2 ** attempt)
            time.sleep(sleep)
    raise RuntimeError(f"OpenAI OCR failed for {[p.name for p in batch]}: {last_error}")


def preflight_openai(model, api_key, timeout):
    body = {"model": model, "input": "Reply with OK only."}
    payload = json.dumps(body).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request = urllib.request.Request(API_URL, data=payload, headers=headers, method="POST")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    text = extract_text_from_response(data).strip()
    if "OK" not in text.upper():
        raise RuntimeError(f"OpenAI preflight returned unexpected text: {text[:200]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-image-dir", required=True)
    parser.add_argument("--output", default="results/submission_gpt_ocr.txt")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--detail", choices=["low", "high", "auto"], default="high")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--preflight-timeout", type=int, default=20)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("OPENAI_API_KEY is required.")

    image_paths = sorted(
        [p for p in Path(args.test_image_dir).iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}],
        key=sort_key,
    )
    if args.limit:
        image_paths = image_paths[: args.limit]
    if not image_paths:
        raise SystemExit("No test images found.")

    out = Path(args.output)
    rows = read_existing(out)
    pending = [p for p in image_paths if p.name not in rows]
    batches = [pending[i : i + args.batch_size] for i in range(0, len(pending), args.batch_size)]

    print(f"images={len(image_paths)} done={len(rows)} pending={len(pending)} batches={len(batches)} model={args.model}", flush=True)
    if pending:
        try:
            preflight_openai(args.model, args.api_key, args.preflight_timeout)
            print("openai preflight ok", flush=True)
        except Exception as exc:
            raise SystemExit(
                "OpenAI API preflight failed. Check local proxy/network before running OCR. "
                f"Error: {exc}"
            )
    write_submission(out, image_paths, rows)

    def run_batch(batch):
        return batch, call_openai(batch, args.model, args.api_key, args.detail, args.timeout, args.retries)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_batch, batch) for batch in batches]
        for idx, future in enumerate(as_completed(futures), 1):
            batch, result = future.result()
            missing = []
            for path in batch:
                if path.name in result:
                    rows[path.name] = result[path.name]
                else:
                    rows[path.name] = ""
                    missing.append(path.name)
            write_submission(out, image_paths, rows)
            print(f"[{idx}/{len(batches)}] wrote {out} batch={[p.name for p in batch]} missing={missing}", flush=True)

    write_final_submission(out, image_paths, rows)


if __name__ == "__main__":
    main()
