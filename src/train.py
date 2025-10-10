# src/integration/train_model.py
"""
Train and evaluate simple baseline models:
- Logistic Regression (class_weight='balanced')
- RandomForest (as a stronger baseline)
- Optionally: SMOTE experiment (only applied on training data)
Saves models and prints evaluation metrics.
"""

import os
import joblib
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from imblearn.over_sampling import SMOTE

MODELS_DIR = os.path.join("models")
PROCESSED_DIR = os.path.join("data", "processed")


def load_processed(train_path=None, test_path=None):
    if train_path is None:
        train_path = os.path.join(PROCESSED_DIR, "train_processed.csv")
    if test_path is None:
        test_path = os.path.join(PROCESSED_DIR, "test_processed.csv")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError("Processed train/test CSVs not found. Run eda_preprocess.py first.")

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    X_train = train.drop("Class", axis=1)
    y_train = train["Class"]
    X_test = test.drop("Class", axis=1)
    y_test = test["Class"]

    return X_train, X_test, y_train, y_test


def train_logistic(X_train, y_train, class_weight="balanced"):
    model = LogisticRegression(class_weight=class_weight, solver="liblinear", random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def train_with_smote(X_train, y_train):
    sm = SMOTE(random_state=42, n_jobs=-1)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"SMOTE: resampled from {len(y_train)} -> {len(y_res)} samples.")
    model = LogisticRegression(solver="liblinear", random_state=42)
    model.fit(X_res, y_res)
    return model


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = None
    try:
        y_proba = model.predict_proba(X_test)[:, 1]
    except Exception:
        pass

    print(f"\n--- Evaluation: {name} ---")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))
    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba) if y_proba is not None else None,
    }
    print("Summary metrics:", metrics)
    return metrics


def save_model(model, name, save_dir=MODELS_DIR):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{name}.joblib")
    joblib.dump(model, path)
    print(f"Saved model -> {path}")
    return path


def main(run_smote=False):
    X_train, X_test, y_train, y_test = load_processed()
    print("Training baseline logistic regression...")
    log_model = train_logistic(X_train, y_train)
    evaluate_model("Logistic (balanced)", log_model, X_test, y_test)
    save_model(log_model, "logistic_balanced")

    print("\nTraining Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    evaluate_model("RandomForest", rf_model, X_test, y_test)
    save_model(rf_model, "random_forest")

    if run_smote:
        print("\nTraining with SMOTE (on training data only)...")
        sm_model = train_with_smote(X_train, y_train)
        evaluate_model("Logistic (SMOTE)", sm_model, X_test, y_test)
        save_model(sm_model, "logistic_smote")

    print("\nTraining complete. Models saved to 'models/'.")
    return True


if __name__ == "__main__":
    # Change to True if you want SMOTE experiment
    main(run_smote=False)
