"""
Preprocess raw CSV(s) in data/raw into a single processed parquet file.
Usage:
  python src/preprocessing.py --in data/raw --out data/processed/dataset.parquet
"""
from pathlib import Path
import argparse
import pandas as pd

def load_raw(folder: Path) -> pd.DataFrame:
    parts = [pd.read_csv(f) for f in folder.glob("*.csv")]
    if not parts:
        raise SystemExit(f"No CSV files found in {folder}")
    return pd.concat(parts, ignore_index=True)

def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates().copy()
    for col in df.select_dtypes(include=["object"]).columns:
        if col.startswith(("amt","loan","income","rate","term")):
            df[col] = pd.to_numeric(df[col], errors="ignore")
    return df.fillna(0)

def add_features(df: pd.DataFrame) -> pd.DataFrame:
    if {"loan_amount","income"}.issubset(df.columns):
        df["loan_to_income"] = df["loan_amount"] / (df["income"] + 1e-9)
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_dir", required=True)
    ap.add_argument("--out", dest="out_file", required=True)
    args = ap.parse_args()
    in_dir = Path(args.in_dir)
    out_file = Path(args.out_file); out_file.parent.mkdir(parents=True, exist_ok=True)
    df = load_raw(in_dir)
    df = add_features(basic_clean(df))
    df.to_parquet(out_file, index=False)
    print(f"Wrote {out_file} rows={len(df)}")

if __name__ == "__main__":
    main()
