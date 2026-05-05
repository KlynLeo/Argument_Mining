from __future__ import annotations

import torch
from transformers import pipeline

import config
import data_utils
import evaluate


LABEL_FOR = config.ZERO_SHOT_LABELS[0]
LABEL_AGAINST = config.ZERO_SHOT_LABELS[1]


def main():
    _, _, test_df = data_utils.load_splits()

    device = 0 if torch.cuda.is_available() else -1
    clf = pipeline("zero-shot-classification", model=config.ZERO_SHOT_MODEL, device=device)

    results = clf(
        test_df["text"].tolist(),
        candidate_labels=[LABEL_FOR, LABEL_AGAINST],
        batch_size=8,
        hypothesis_template=config.ZERO_SHOT_HYPOTHESIS_TEMPLATE,
    )

    y_pred = [1 if r["labels"][0] == LABEL_AGAINST else 0 for r in results]
    y_true = test_df["label"].tolist()

    metrics = evaluate.compute_metrics(y_true, y_pred)
    evaluate.print_metrics("ZERO-SHOT (test)", metrics)
    evaluate.save_metrics(metrics, config.ZERO_SHOT_METRICS_PATH)

    test_df["pred_label"] = y_pred
    test_df.to_csv(config.ZERO_SHOT_PREDICTIONS_PATH, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
