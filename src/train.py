"""
train.py
Trains baseline Logistic Regression and advanced Random Forest models for fraud detection.
Handles missing 'Time' column gracefully and saves figures/models for reports.
"""

from pathlib import Path
import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from data_loader import load_creditcard_csv
from preprocess import preprocess_dataframe
from models import train_baseline_logistic, balance_with_smote, train_random_forest, save_model
from evaluate import evaluate_model, create_comparison_plots

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT.parent / "data" / "creditcard.csv"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"

REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    logging.info("Starting model training pipeline...")
    
    # 1. Load Data
    df = load_creditcard_csv(DATA_PATH)
    logging.info(f"Class distribution:\n{df['Class'].value_counts(normalize=True)}")

    # 2. Preprocess Data
    logging.info("Preprocessing data...")
    X, y, scaler = preprocess_dataframe(df)
    save_model(scaler, MODELS_DIR / "scaler.joblib")
    logging.info("Scaler saved.")

    # 3. Split Data
    logging.info("Splitting data into training and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 4. Train Baseline Model
    logging.info("Training baseline Logistic Regression model...")
    baseline_model = train_baseline_logistic(X_train, y_train)
    base_metrics = evaluate_model(baseline_model, X_test, y_test, "baseline", REPORTS_DIR / "figures")

    # 5. Balance Data with SMOTE
    logging.info("Balancing training data with SMOTE...")
    X_train_bal, y_train_bal = balance_with_smote(X_train, y_train)

    # 6. Train Random Forest Model
    logging.info("Training Random Forest model on balanced data...")
    rf_model = train_random_forest(X_train_bal, y_train_bal)
    rf_metrics = evaluate_model(rf_model, X_test, y_test, "random_forest", REPORTS_DIR / "figures")
    
    # 7. Save the final model
    save_model(rf_model, MODELS_DIR / "fraud_detection_model.joblib")
    logging.info("Final Random Forest model saved.")

    # 8. Compare Models and Generate Final Reports
    logging.info("Generating comparison plots...")
    create_comparison_plots(base_metrics, rf_metrics, REPORTS_DIR / "figures")
    
    logging.info("\n" + "="*50)
    logging.info("Final Random Forest Classification Report:")
    y_pred_rf = rf_model.predict(X_test)
    logging.info("\n" + classification_report(y_test, y_pred_rf))
    logging.info("="*50)

    logging.info(f"Pipeline finished. Reports saved to: {REPORTS_DIR}")

if __name__ == "__main__":
    main()