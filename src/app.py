import gradio as gr
from predictor import predict

TOP_FEATURES = ['V17','V14','V12','V10','V11']

def make_prediction(*vals):
    feature_dict = dict(zip(TOP_FEATURES, vals))
    result = predict(feature_dict)
    if result["prediction"]==1:
        return f"🚨 FRAUD (prob={result['fraud_prob']:.2%})"
    else:
        return f"✅ LEGITIMATE (prob={result['fraud_prob']:.2%})"

inputs = [gr.Number(label=f) for f in TOP_FEATURES]

gr.Interface(
    fn=make_prediction,
    inputs=inputs,
    outputs=gr.Textbox(label="Prediction"),
    title="Credit Card Fraud Detection AI"
).launch()
