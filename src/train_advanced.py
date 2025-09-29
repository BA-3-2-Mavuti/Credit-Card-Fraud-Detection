"""
Train an advanced Random Forest model on balanced data using SMOTE.
Usage:
  python src/train_advanced.py
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import os

def main():
    # Load data
    data_path = 'data/creditcard.csv'
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_csv(data_path)

    # Define features (exclude 'Class', 'Amount', 'Time')
    features = [c for c in df.columns if c not in ['Class', 'Amount', 'Time']]
    X = df[features]
    y = df['Class']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    print("Train/test split created:", X_train.shape, X_test.shape)

    # Apply SMOTE to balance training data
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    print("Balanced training data:", X_train_balanced.shape, y_train_balanced.shape)

    # Train Random Forest
    advanced_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    advanced_model.fit(X_train_balanced, y_train_balanced)

    print("Advanced model training complete.")

if __name__ == "__main__":
    main()
