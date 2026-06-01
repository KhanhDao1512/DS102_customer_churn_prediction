import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_binary_metrics(
    y_true,
    y_pred,
    y_prob=None,
    model_name=None,
    threshold=None,
):
    """Compute core binary classification metrics for the churn class."""

    metrics = {}

    if model_name is not None:
        metrics["Model"] = model_name

    if threshold is not None:
        metrics["Threshold"] = threshold

    metrics.update({
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC_AUC": (
            roc_auc_score(y_true, y_prob)
            if y_prob is not None
            else np.nan
        ),
    })

    return metrics


def classification_report_to_frame(y_true, y_pred):
    """Return sklearn classification_report as a DataFrame."""

    return pd.DataFrame(
        classification_report(
            y_true,
            y_pred,
            output_dict=True,
            zero_division=0,
        )
    ).transpose()
