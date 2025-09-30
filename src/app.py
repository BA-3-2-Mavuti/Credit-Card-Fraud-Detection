"""
app.py
Fraud Detection Chatbot (Gradio)
- Accepts either:
  1) CSV-style row from dataset (Time,V1..V28,Amount,Class)
  2) or key=value format (Amount=..., Time=..., V1=..., ...)
- Uses trained model from predictor.py
- Handles both old scaler (1 feature) and new scaler (2 features)
"""

import gradio as gr
import sys
from pathlib import Path
import pandas as pd
import traceback
import random

# --- Setup imports ---
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

MODEL_AVAILABLE = True
try:
    from predictor import make_prediction, get_top_features
    TOP_FEATURES = get_top_features()
except Exception as e:
    MODEL_AVAILABLE = False
    print(f"⚠️ Model not found, running in demo mode: {e}")
    TOP_FEATURES = [f"V{i}" for i in range(14, 19)]  # dummy features

    def make_prediction(transaction_dict):
        prob = random.uniform(0.02, 0.98)
        pred = 1 if prob > 0.5 else 0
        return {"prediction": int(pred), "fraud_probability": float(prob)}

# CSV expected columns
CSV_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]

# --- Parse User Input ---
def parse_input_to_transaction(user_input: str):
    txt = user_input.strip()
    if not txt:
        raise ValueError("Empty input")

    # Case A: CSV row (no '=' signs)
    if "," in txt and "=" not in txt:
        parts = [p.strip().replace('"', "") for p in txt.split(",")]
        if len(parts) == len(CSV_COLS):
            nums = [float(p) for p in parts]
            tx = {k: v for k, v in zip(CSV_COLS, nums)}
            tx.pop("Class", None)
            return tx
        elif len(parts) == 30:
            # maybe missing Time column
            nums = [float(p) for p in parts]
            cols_no_time = [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
            tx = {k: v for k, v in zip(cols_no_time, nums)}
            tx["Time"] = 0.0
            tx.pop("Class", None)
            return tx
        else:
            raise ValueError(f"CSV row has {len(parts)} values, expected {len(CSV_COLS)}")

    # Case B: key=value format
    tx = {}
    for part in txt.split(","):
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        val = val.strip().replace('"', "")
        tx[key] = float(val)

    if "Amount" not in tx or "Time" not in tx:
        raise ValueError("Please include at least Amount and Time.")

    return tx

# --- Build Reply ---
def build_human_response(tx_dict, model_result):
    pred = model_result.get("prediction", 0)
    prob = model_result.get("fraud_probability", None)

    if prob is None:
        status, emoji, prob_text = "Unknown", "❔", "N/A"
    elif prob >= 0.8:
        status, emoji, prob_text = "HIGH RISK - FRAUD LIKELY", "🚨", f"{prob*100:.2f}%"
    elif prob >= 0.5:
        status, emoji, prob_text = "SUSPICIOUS - REVIEW", "⚠️", f"{prob*100:.2f}%"
    else:
        status, emoji, prob_text = "LOW RISK - LIKELY LEGIT", "✅", f"{prob*100:.2f}%"

    amount = tx_dict.get("Amount", 0.0)
    time = tx_dict.get("Time", 0.0)

    reply = f"""{emoji} **{status}**
- Fraud probability: **{prob_text}**
- Amount: **${amount:,.2f}**
- Time: **{time} seconds**"""

    if prob is not None and prob >= 0.5:
        reply += "\n\n📌 Recommended: *Hold transaction for manual review.*"
    else:
        reply += "\n\n📌 Recommended: *No immediate action required.*"

    return reply

# --- Gradio Chatbot Logic ---
def chatbot_predict(history, message: str):
    user_text = message.strip()
    try:
        tx = parse_input_to_transaction(user_text)
        result = make_prediction(tx)
        reply = build_human_response(tx, result)
    except Exception as e:
        reply = f"⚠️ Error: {e}\n\n{traceback.format_exc()}"

    history = history + [(user_text, reply)]
    return history, history

# --- Gradio UI ---
def make_demo():
    desc = (
        "Paste a CSV row or type key=value pairs.\n\n"
        "Examples:\n"
        "- CSV row:\n"
        "  2,-1.1582,0.8777,1.5487,...,69.99,0\n\n"
        "- Key=Value format:\n"
        "  Amount=69.99, Time=2, V1=-1.1582, V2=0.8777, V3=1.5487"
    )

    with gr.Blocks(theme=gr.themes.Soft(), title="Fraud Detection Chatbot") as demo:
        gr.Markdown("# 💬 Fraud Detection Chatbot")
        gr.Markdown(desc)

        chatbot = gr.Chatbot(height=380)
        state = gr.State([])

        with gr.Row():
            txt = gr.Textbox(show_label=False, placeholder="Paste CSV row or type Amount=..., Time=...")
            send = gr.Button("Analyze")

        # Quick sample row button
        sample_row = (
            '2,-1.15823309349523,0.877736754848451,1.548717846511,0.403033933955121,'
            '-0.407193377311653,0.0959214624684256,0.592940745385545,-0.270532677192282,'
            '0.817739308235294,0.753074431976354,-0.822842877946363,0.53819555014995,'
            '1.3458515932154,-1.11966983471731,0.175121130008994,-0.451449182813529,'
            '-0.237033239362776,-0.0381947870352842,0.803486924960175,0.408542360392758,'
            '-0.00943069713232919,0.79827849458971,-0.137458079619063,0.141266983824769,'
            '-0.206009587619756,0.502292224181569,0.219422229513348,0.215153147499206,69.99,"0"'
        )
        sample_btn = gr.Button("Use sample transaction")
        sample_btn.click(lambda: sample_row, None, txt)

        send.click(chatbot_predict, inputs=[state, txt], outputs=[chatbot, state])
        txt.submit(chatbot_predict, inputs=[state, txt], outputs=[chatbot, state])

        if MODEL_AVAILABLE:
            gr.Markdown("✅ Model loaded successfully.")
        else:
            gr.Markdown("⚠️ No trained model found. Running in demo mode.")

    return demo

if __name__ == "__main__":
    app = make_demo()
    print("🚀 Launching Fraud Detection Chatbot...")
    app.launch()
