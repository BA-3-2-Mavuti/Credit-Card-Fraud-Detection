import joblib, os, pandas as pd

MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/fraud_detection_model.joblib')
MODEL = joblib.load(MODEL_PATH)

FEATURES = [f'V{i}' for i in range(1,29)] + ['scaled_time','scaled_amount']

def predict(transaction_dict):
    # Fill missing features with 0
    data = {f:0 for f in FEATURES}
    for f,v in transaction_dict.items():
        if f in data: data[f]=v
    df = pd.DataFrame([data])
    pred = MODEL.predict(df)[0]
    prob = MODEL.predict_proba(df)[0][1]
    return {"prediction": int(pred), "fraud_prob": float(prob)}
