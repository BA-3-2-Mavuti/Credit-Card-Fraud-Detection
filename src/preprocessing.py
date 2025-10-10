# src/integration/eda_preprocess.py
"""
EDA + Preprocessing script.
- Loads creditcard.csv from common locations
- Generates a couple of quick plots and saves them in reports/figures
- Scales Time and Amount (StandardScaler)
- Splits into train/test (stratified)
- Saves processed train/test CSVs under data/processed/
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

sns.set_style("whitegrid")

POSSIBLE_PATHS = [
    os.path.join("..", "data", "creditcard.csv"),
    os.path.join(".", "data", "creditcard.csv"),
    "creditcard.csv",
    os.path.join("data", "raw", "creditcard.csv"),
]

FIGURES_DIR = os.path.join("reports", "figures")
PROCESSED_DIR = os.path.join("data", "processed")


def find_dataset():
    for p in POSSIBLE_PATHS:
        if os.path.exists(p):
            return p
    return None


def load_data(path=None):
    p = path or find_dataset()
    if p is None:
        raise FileNotFoundError(
            "Could not locate 'creditcard.csv'. Checked paths: "
            + ", ".join(POSSIBLE_PATHS)
        )
    df = pd.read_csv(p)
    print(f"Loaded dataset from: {p}  shape={df.shape}")
    return df


def plot_class_distribution(df, save_path):
    counts = df["Class"].value_counts()
    plt.figure(figsize=(7, 5))
    sns.barplot(x=counts.index, y=counts.values, palette="viridis")
    plt.title("Distribution of Transactions (0: Non-Fraud, 1: Fraud)")
    plt.xlabel("Class")
    plt.ylabel("Number of Transactions")
    plt.xticks([0, 1], ["Non-Fraud", "Fraud"])
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Saved class distribution chart -> {save_path}")


def plot_time_analysis(df, save_path):
    # If Time exists, compute hour-of-day (heuristic)
    if "Time" in df.columns:
        df2 = df.copy()
        # Keep it robust: Time in dataset is seconds since first transaction;
        # converting to hour modulo 24 gives a coarse view
        df2["Hour"] = ((df2["Time"] // 3600) % 24).astype(int)
        fraud_by_hour = df2.groupby("Hour")["Class"].mean() * 100
        transactions_by_hour = df2.groupby("Hour")["Class"].count()

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fraud_by_hour.plot(ax=axes[0], marker="o")
        axes[0].set_title("Fraud Rate by Hour (percent)")
        axes[0].set_xlabel("Hour")
        axes[0].set_ylabel("Fraud %")

        transactions_by_hour.plot(kind="bar", ax=axes[1], alpha=0.7)
        axes[1].set_title("Transactions by Hour")
        axes[1].set_xlabel("Hour")
        axes[1].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        print(f"Saved time analysis chart -> {save_path}")
    else:
        print("No 'Time' column found; skipping time analysis chart.")


def preprocess(df):
    df_proc = df.copy()

    # Scale Time (if present) and Amount
    scalers = {}
    if "Time" in df_proc.columns:
        time_scaler = StandardScaler()
        df_proc["scaled_Time"] = time_scaler.fit_transform(
            df_proc["Time"].values.reshape(-1, 1)
        )
        df_proc.drop(["Time"], axis=1, inplace=True)
        scalers["time_scaler"] = time_scaler
    elif "id" in df_proc.columns:
        time_scaler = StandardScaler()
        df_proc["scaled_Time"] = time_scaler.fit_transform(
            df_proc["id"].values.reshape(-1, 1)
        )
        df_proc.drop(["id"], axis=1, inplace=True)
        scalers["time_scaler"] = time_scaler

    if "Amount" in df_proc.columns:
        amount_scaler = StandardScaler()
        df_proc["scaled_Amount"] = amount_scaler.fit_transform(
            df_proc["Amount"].values.reshape(-1, 1)
        )
        df_proc.drop(["Amount"], axis=1, inplace=True)
        scalers["amount_scaler"] = amount_scaler

    # reorder so scaled_* upfront (not required, just tidy)
    cols = list(df_proc.columns)
    front = [c for c in ["scaled_Amount", "scaled_Time"] if c in cols]
    for c in front:
        cols.remove(c)
    df_proc = df_proc[front + cols]

    # Ensure 'Class' is present
    if "Class" not in df_proc.columns:
        raise KeyError("'Class' column is required in the dataset for supervised learning.")

    X = df_proc.drop("Class", axis=1)
    y = df_proc["Class"]

    return X, y, scalers, df_proc


def split_and_save(X, y, save_dir=PROCESSED_DIR):
    os.makedirs(save_dir, exist_ok=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    # Save small CSVs for later reproducibility
    train_df = X_train.copy()
    train_df["Class"] = y_train
    test_df = X_test.copy()
    test_df["Class"] = y_test

    train_path = os.path.join(save_dir, "train_processed.csv")
    test_path = os.path.join(save_dir, "test_processed.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)
    print(f"Saved processed train/test to: {train_path} and {test_path}")

    return train_path, test_path


def main():
    print("\n--- EDA & Preprocessing ---")
    df = load_data()
    os.makedirs(FIGURES_DIR, exist_ok=True)
    plot_class_distribution(df, os.path.join(FIGURES_DIR, "class_imbalance_chart.png"))
    plot_time_analysis(df, os.path.join(FIGURES_DIR, "time_series_analysis.png"))

    print("\nPreprocessing dataset...")
    X, y, scalers, df_proc = preprocess(df)
    train_path, test_path = split_and_save(X, y)
    print("\nDone. Preprocessed files are ready for modeling.")
    return {"train_path": train_path, "test_path": test_path, "scalers": scalers}


if __name__ == "__main__":
    main()
