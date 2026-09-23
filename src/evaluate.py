"""Shared evaluation functions so every model uses identical definitions."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay,
)

def metric_row(name, y_true, probability, threshold=0.5):
    prediction = (probability >= threshold).astype(int)
    return {
        "model": name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, prediction),
        "precision": precision_score(y_true, prediction, zero_division=0),
        "recall": recall_score(y_true, prediction, zero_division=0),
        "f1": f1_score(y_true, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_true, probability),
    }

def save_evaluation(results, y_true, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([metric_row(n, y_true, p) for n, p in results.items()])
    metrics.to_csv(out / "metrics.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics.to_dict("records"), indent=2))

    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for name, proba in results.items():
        RocCurveDisplay.from_predictions(y_true, proba, name=name, ax=ax)
    ax.plot([0, 1], [0, 1], "--", color="#8f8a8c", linewidth=1)
    ax.set_title("ROC curves on the same test set")
    fig.tight_layout(); fig.savefig(out / "roc_curves.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, len(results), figsize=(4.1 * len(results), 3.8))
    if len(results) == 1: axes = [axes]
    for ax, (name, proba) in zip(axes, results.items()):
        cm = confusion_matrix(y_true, (proba >= 0.5).astype(int))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax)
        ax.set_title(name); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.suptitle("Confusion matrices at threshold 0.50", y=1.03)
    fig.tight_layout(); fig.savefig(out / "confusion_matrices.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    return metrics
