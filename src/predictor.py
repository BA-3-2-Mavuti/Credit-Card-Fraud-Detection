from pathlib import Path
import joblib
import pandas as pd

# Feature names expected by the model
FEATURE_NAMES = [f'V{i}' for i in range(1, 29)] + ['scaled_Amount', 'scaled_Time']

# Paths to saved model and scaler
MODEL_PATH = Path(__file__).resolve().parents[1] / 'models' / 'fraud_detection_model.joblib'
SCALER_PATH = Path(__file__).resolve().parents[1] / 'models' / 'scaler.joblib'

# Internal cached artifacts
_model = None
_scaler = None
_top_features = None

def load_artifacts():
    """
    Loads the trained model and scaler from disk, and extracts top features.
    """
    global _model, _scaler, _top_features

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f'Model not found at {MODEL_PATH}. Run training first.')
        _model = joblib.load(MODEL_PATH)

    if _scaler is None:
        if not SCALER_PATH.exists():
            raise FileNotFoundError(f'Scaler not found at {SCALER_PATH}. Run training first.')
        _scaler = joblib.load(SCALER_PATH)

    try:
        # Use the actual features the model was trained on
        model_features = _model.feature_names_in_
        imps = _model.feature_importances_
        imp_df = pd.DataFrame({'feature': model_features, 'importance': imps})
        _top_features = imp_df.sort_values('importance', ascending=False).head(5)['feature'].tolist()
    except Exception:
        # Fallback if feature names not available
        _top_features = ['V17', 'V14', 'V12', 'V10', 'V11']  # Default from a typical run

def get_top_features():
    """Returns the top 5 most important features."""
    if _top_features is None:
        load_artifacts()
    return _top_features

def prepare_transaction(transaction_dict):
    """
    Prepares a transaction dictionary into a DataFrame suitable for the model.
    Handles both 1-feature (Amount only) and 2-feature (Amount + Time) scalers.
    """
    data = {feat: 0.0 for feat in FEATURE_NAMES}

    # Populate V1–V28
    for i in range(1, 29):
        key = f'V{i}'
        if key in transaction_dict:
            data[key] = float(transaction_dict[key])

    # Raw Amount & Time
    amt = float(transaction_dict.get('Amount', 0.0))
    t = float(transaction_dict.get('Time', 0.0))

    if _scaler is not None:
        try:
            # Case: scaler trained on Amount + Time
            scaled = _scaler.transform([[amt, t]])[0]
            data['scaled_Amount'] = float(scaled[0])
            data['scaled_Time'] = float(scaled[1])
        except ValueError:
            # Case: scaler trained only on Amount
            scaled_amt = _scaler.transform([[amt]])[0]
            data['scaled_Amount'] = float(scaled_amt[0])
            data['scaled_Time'] = float(t)  # keep raw Time
    else:
        # Fallback: no scaler available
        data['scaled_Amount'] = float(amt)
        data['scaled_Time'] = float(t)

    df = pd.DataFrame([data], columns=FEATURE_NAMES)
    return df

def make_prediction(transaction_dict):
    """
    Makes a fraud prediction on a single transaction dictionary.
    Returns a dictionary with prediction (0/1) and fraud probability (0-1).
    """
    df = prepare_transaction(transaction_dict)
    pred = _model.predict(df)[0]
    proba = None
    if hasattr(_model, "predict_proba"):
        proba = _model.predict_proba(df)[0, 1]

    return {
        'prediction': int(pred),
        'fraud_probability': float(proba) if proba is not None else None
    }

# Load artifacts on module import
load_artifacts()
