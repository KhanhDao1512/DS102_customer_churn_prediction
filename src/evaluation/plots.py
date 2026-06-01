import matplotlib.pyplot as plt
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def save_confusion_matrix(
    y_true,
    y_pred,
    output_path,
    threshold=None,
):
    """Save a confusion matrix plot."""

    ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=["No Churn", "Churn"],
        cmap="Blues",
    )

    title = "Baseline 05 Confusion Matrix"
    if threshold is not None:
        title = f"{title} (threshold={threshold:.2f})"

    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_roc_curve(
    y_true,
    y_prob,
    output_path,
    name="Final Logistic Regression",
):
    """Save a ROC curve plot."""

    RocCurveDisplay.from_predictions(
        y_true,
        y_prob,
        name=name,
    )
    plt.title("Baseline 05 ROC Curve")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def save_precision_recall_curve(
    y_true,
    y_prob,
    output_path,
    name="Final Logistic Regression",
):
    """Save a precision-recall curve plot."""

    PrecisionRecallDisplay.from_predictions(
        y_true,
        y_prob,
        name=name,
    )
    plt.title("Baseline 05 Precision-Recall Curve")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
