import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    args = parser.parse_args()
    df = pd.read_csv(args.csv_path)
    print(df.shape)
    print(df.head(10).to_string(index=False))
    assert list(df.columns) == ["new_name", "value"], df.columns
    assert len(df) == 10000, len(df)
    assert df["new_name"].is_unique
    empty = df["value"].isna() | (df["value"].astype(str).str.len() == 0)
    assert int(empty.sum()) == 0, int(empty.sum())
    print("submission ok")


if __name__ == "__main__":
    main()
