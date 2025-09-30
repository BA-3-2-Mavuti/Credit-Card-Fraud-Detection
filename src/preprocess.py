from sklearn.preprocessing import StandardScaler
import pandas as pd
import joblib
from pathlib import Path

FEATURE_ORDER = [f"V{i}" for i in range(1,29)] + ["scaled_Amount", "scaled_Time"]

def preprocess_dataframe(df: pd.DataFrame, save_scaler_to: str = None):
    """
    Preprocesses the credit card dataframe:
    - Drops duplicates & NA
    - Scales 'Amount' and 'Time' into 'scaled_Amount' and 'scaled_Time'
    - Returns:
        X: preprocessed features
        y: target variable
        scaler: fitted StandardScaler
    """
    df = df.copy()

    # Drop duplicates and NAs
    df = df.drop_duplicates().reset_index(drop=True)
    if df.isna().sum().sum() > 0:
        df = df.dropna().reset_index(drop=True)

    # Ensure required columns exist
    required = {f"V{i}" for i in range(1,29)} | {"Amount", "Time", "Class"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Input df missing required columns: {missing}")

    # Scale Amount and Time
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df[['Amount','Time']].values)
    df['scaled_Amount'] = scaled_values[:,0]
    df['scaled_Time'] = scaled_values[:,1]

    # Features and target
    X = df.drop(columns=['Amount','Time','Class'])
    y = df['Class']

    # Ensure consistent feature order
    cols = [c for c in FEATURE_ORDER if c in X.columns] + \
           [c for c in X.columns if c not in FEATURE_ORDER]
    X = X[cols]

    if save_scaler_to:
        path = Path(save_scaler_to)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(scaler, path)

    return X, y, scaler