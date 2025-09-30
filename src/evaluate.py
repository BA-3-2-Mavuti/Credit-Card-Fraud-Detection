from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
    recall_score,
    precision_score
)
import pandas as pd

sns.set(style='whitegrid')

def save_figure(fig, path):
    """
    Saves a matplotlib figure to the specified path, creating directories if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def evaluate_model(model, X_test, y_test, prefix, out_dir):
    """
    Generates confusion matrix, ROC and Precision-Recall curves for a model.
    Returns a metrics dictionary.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Predictions
    y_pred = model.predict(X_test)

    # Predicted probabilities or decision function
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_proba = model.decision_function(X_test)
        # Scale to 0-1 range for consistency
        y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())
    else:
        y_proba = y_pred # Fallback for models without proba

    # Metrics calculation
    roc_auc = roc_auc_score(y_test, y_proba)
    avg_precision = average_precision_score(y_test, y_proba)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{prefix} Confusion Matrix")
    save_figure(fig, out_dir / f"{prefix}_confusion_matrix.png")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    ax.plot([0,1],[0,1], linestyle='--', color='grey')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{prefix} ROC Curve")
    ax.legend()
    save_figure(fig, out_dir / f"{prefix}_roc.png")

    # Precision-Recall curve
    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(6,4))
    ax.plot(recall_vals, precision_vals, label=f"AP = {avg_precision:.4f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{prefix} Precision-Recall Curve")
    ax.legend()
    save_figure(fig, out_dir / f"{prefix}_precision_recall.png")

    return {
        'model': prefix,
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'roc_auc': roc_auc,
        'avg_precision': avg_precision
    }

def create_comparison_plots(base_metrics, rf_metrics, out_dir):
    """
    Creates bar plots comparing recall and roc_auc for baseline and random forest.
    """
    df = pd.DataFrame([base_metrics, rf_metrics])
    
    # Recall comparison
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x='model', y='recall', data=df, ax=ax, palette='viridis')
    ax.set_title("Fraud Recall Comparison")
    ax.set_ylim(0, 1)
    for i in ax.patches:
        ax.text(i.get_x() + i.get_width()/2, i.get_height()+0.02, f"{i.get_height():.2%}", ha='center')
    save_figure(fig, out_dir / "recall_comparison.png")

    # ROC AUC comparison
    fig, ax = plt.subplots(figsize=(6,4))
    sns.barplot(x='model', y='roc_auc', data=df, ax=ax, palette='plasma')
    ax.set_title("ROC AUC Score Comparison")
    ax.set_ylim(0, 1)
    for i in ax.patches:
        ax.text(i.get_x() + i.get_width()/2, i.get_height()+0.02, f"{i.get_height():.4f}", ha='center')
    save_figure(fig, out_dir / "roc_auc_comparison.png")