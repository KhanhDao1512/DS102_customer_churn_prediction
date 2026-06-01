import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score


def find_best_threshold(
    y_true,
    y_prob,
    start=0.1,
    stop=0.9,
    step=0.01,
):
    """Find the probability threshold that maximizes F1 on validation data."""

    thresholds = np.arange(start, stop, step)

    threshold_results = []
    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)

        precision = precision_score(
            y_true,
            y_pred,
            zero_division=0,
        )
        recall = recall_score(
            y_true,
            y_pred,
            zero_division=0,
        )
        f1 = f1_score(
            y_true,
            y_pred,
            zero_division=0,
        )

        threshold_results.append({
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        })

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold

    return best_threshold, best_f1, pd.DataFrame(threshold_results)
