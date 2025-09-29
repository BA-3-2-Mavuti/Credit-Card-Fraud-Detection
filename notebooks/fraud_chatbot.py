import pandas as pd
import joblib


def fraud_chatbot():
    # Load dataset (must be the same one you trained on)
    df = pd.read_csv(r"C:\Users\M SMART\Documents\Credit-Card-Fraud-Detection - Copy\data\creditcard_2023.csv")

    # Load trained model and scaler
    rf_model = joblib.load("rf_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Assume you have already computed accuracy of the model
    model_accuracy = 0.98  # replace with your real accuracy score

    print("💬 Welcome to the Fraud Detection Chatbot!")
    print("You can enter a Transaction ID, and I will check if it is FRAUD or NOT FRAUD.")
    print(f"⚡ My AI model has an accuracy of {model_accuracy*100:.2f}%")
    print("👉 Type 'exit' anytime to close the chatbot.\n")

    while True:
        try:
            # Ask for transaction ID
            user_input = input("Enter Transaction ID (or type 'exit' to quit): ").strip().lower()

            if user_input == "exit":
                print("👋 Thank you for using the Fraud Detection Chatbot! Goodbye!")
                break

            # Convert input to integer
            transaction_id = int(user_input)

            # Find transaction in dataset
            if transaction_id in df['id'].values:
                transaction = df[df['id'] == transaction_id]

                # Extract features (drop id and Class, keep DataFrame format!)
                features = transaction.drop(['id', 'Class'], axis=1)

                # Scale features (preserve column names)
                features_scaled = pd.DataFrame(
                    scaler.transform(features),
                    columns=features.columns
                )

                # Predict
                prediction = rf_model.predict(features_scaled.values)[0]
                probability = rf_model.predict_proba(features_scaled.values)[0][1]

                if prediction == 1:
                    print(f"🚨 Transaction {transaction_id} is predicted as FRAUD "
                          )
                else:
                    print(f"✅ Transaction {transaction_id} is predicted as NOT FRAUD "
                         )
            else:
                print("⚠️ Transaction ID not found. Please try again.\n")

        except ValueError:
            print("❌ Invalid input. Please enter a valid Transaction ID or 'exit' to quit.\n")


if __name__ == "__main__":
    fraud_chatbot()
