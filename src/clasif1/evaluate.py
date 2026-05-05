from __future__ import annotations
import json
from pathlib import Path
 
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
 
import config

def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    accuracy = accuracy_score(y_true, y_pred)

    p_pos, r_pos, f1_pos, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    p_mac, r_mac, f1_mac, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    target_names = [config.ID2LABEL[0], config.ID2LABEL[1]]
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()

    return {
        "accuracy": float(accuracy),
        "precision_pos": float(p_pos),
        "recall_pos": float(r_pos),
        "f1_pos": float(f1_pos),
        "precision_macro": float(p_mac),
        "recall_macro": float(r_mac),
        "f1_macro": float(f1_mac),
        "n_samples": int(len(y_true)),
        "confusion_matrix": cm,
        "classification_report": report,
    }


def compute_metrics_for_trainer(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    m = compute_metrics(labels, preds)
    return {
        "accuracy": m["accuracy"],
        "f1_pos": m["f1_pos"],
        "f1_macro": m["f1_macro"],
    }


def print_metrics(title, metrics):
    print(f"\n=== {title} ===")
    print(f"  n_samples : {metrics['n_samples']}")
    print(f"  accuracy  : {metrics['accuracy']:.4f}")
    print(f"  f1_pos    : {metrics['f1_pos']:.4f}  (precision={metrics['precision_pos']:.4f}, recall={metrics['recall_pos']:.4f})")
    print(f"  f1_macro  : {metrics['f1_macro']:.4f}  (precision={metrics['precision_macro']:.4f}, recall={metrics['recall_macro']:.4f})")
    print()
    print(metrics["classification_report"])
    cm = metrics["confusion_matrix"]
    print(f"  confusion matrix (rows=true, cols=pred, [0={config.ID2LABEL[0]}, 1={config.ID2LABEL[1]}]):")
    print(f"    [[{cm[0][0]:4d} {cm[0][1]:4d}]")
    print(f"     [{cm[1][0]:4d} {cm[1][1]:4d}]]")


def save_metrics(metrics, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"Salvat metrici la: {path}")


if __name__ == "__main__":
    y_true = [1, 1, 0, 0, 1, 0, 1, 1, 0, 1]
    y_pred = [1, 0, 0, 1, 1, 0, 1, 1, 0, 0]
    m = compute_metrics(y_true, y_pred)
    print_metrics("SMOKE TEST", m)