"""
Train a simple model from processed parquet.
Usage:
  python src/train.py --data data/processed/dataset.parquet --target is_fraud
"""
from pathlib import Path
import argparse, time, joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--models_dir", default="models")
    args = ap.parse_args()
    df = pd.read_parquet(Path(args.data))
    if args.target not in df.columns: raise SystemExit(f"Target {args.target} missing")
    y = df[args.target]; X = df.drop(columns=[args.target])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42,
                                          stratify=y if y.nunique()==2 else None)
    model = LogisticRegression(max_iter=1000)
    model.fit(Xtr, ytr)
    proba = model.predict_proba(Xte)[:,1]
    auc = roc_auc_score(yte, proba)
    print(f"AUC: {auc:.4f}")
    models_dir = Path(args.models_dir); models_dir.mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    vpath = models_dir / f"model_{ts}.joblib"
    lpath = models_dir / "latest.joblib"
    joblib.dump(model, vpath); joblib.dump(model, lpath)
    print(f"Saved {vpath.name} and latest.joblib")

if __name__ == "__main__":
    main()
