import argparse
import csv
from pathlib import Path


def sniff_delimiter(path):
    first = Path(path).read_text(encoding="utf-8-sig").splitlines()[0]
    return "\t" if "\t" in first else ","


def read_rows(path):
    delimiter = sniff_delimiter(path)
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f, delimiter=delimiter)
        rows = []
        for row in reader:
            name = str(row.get("new_name", "")).strip()
            value = str(row.get("value", "")).strip()
            if name:
                rows.append({"new_name": name, "value": value})
        return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="Complete 10000-row base submission.")
    parser.add_argument("--patch", required=True, help="Higher quality partial or complete submission.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--output-delimiter", choices=["comma", "tab"], default="tab")
    args = parser.parse_args()

    base_rows = read_rows(args.base)
    patch_rows = read_rows(args.patch)
    patch = {row["new_name"]: row["value"] for row in patch_rows if row["value"]}

    fused = []
    replaced = 0
    for row in base_rows:
        name = row["new_name"]
        value = row["value"]
        if name in patch:
            value = patch[name]
            replaced += 1
        fused.append({"new_name": name, "value": value or " "})

    if len(fused) != 10000:
        raise RuntimeError(f"Expected 10000 rows, got {len(fused)}")
    if len({row["new_name"] for row in fused}) != 10000:
        raise RuntimeError("Duplicate new_name values found")

    delimiter = "\t" if args.output_delimiter == "tab" else ","
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["new_name", "value"], delimiter=delimiter, lineterminator="\n")
        writer.writeheader()
        writer.writerows(fused)

    print(f"base_rows={len(base_rows)}")
    print(f"patch_rows={len(patch_rows)}")
    print(f"patch_nonempty={len(patch)}")
    print(f"replaced={replaced}")
    print(f"wrote={out}")


if __name__ == "__main__":
    main()
