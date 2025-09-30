from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path

def train_baseline_logistic(X_train, y_train):
    """
    Trains a baseline Logistic Regression model.
    """
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    return lr

def balance_with_smote(X_train, y_train):
    """
    Applies SMOTE to balance the training dataset.
    """
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    return X_res, y_res

def train_random_forest(X_train_bal, y_train_bal, n_estimators=200):
    """
    Trains a Random Forest classifier on the balanced dataset.
    """
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42, n_jobs=-1)
    rf.fit(X_train_bal, y_train_bal)
    return rf

def save_model(obj, path):
    """
    Saves a model or scaler to disk using joblib. Creates directories if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)