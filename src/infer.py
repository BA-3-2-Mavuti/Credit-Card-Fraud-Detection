"""
Run inference on a CSV or parquet of new records (no target column required).
Usage:
  python src/infer.py --model models/latest.joblib --input new.csv --out predictions.csv
"""
from pathlib import Path
import argparse, joblib, pandas as pd

def load_table(path: Path):
    return pd.read_parquet(path) if path.suffix.lower()==".parquet" else pd.read_csv(path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    model = joblib.load(args.model)
    df = load_table(Path(args.input))
    proba = model.predict_proba(df)[:,1]
    preds = (proba >= 0.5).astype(int)
    out_df = df.copy()
    out_df["prob_fraud"] = proba
    out_df["prediction"] = preds
    out_path = Path(args.out)
    out_df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} rows={len(out_df)}")

if __name__ == "__main__":
    main()
