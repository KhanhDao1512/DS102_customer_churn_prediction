from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler


current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

sys.path.append(str(project_root))

from src.data.preprocessing import preprocess_pipeline
from src.data.split import make_train_val_test_split
from src.evaluation.metrics import (
    classification_report_to_frame,
    compute_binary_metrics,
)
from src.evaluation.plots import (
    save_confusion_matrix,
    save_precision_recall_curve,
    save_roc_curve,
)
from src.evaluation.threshold import find_best_threshold
from src.features.features import build_features
from src.models.modeling import (
    benchmark_models,
    fit_final_logistic_regression,
    get_baseline_model_configs,
    predict_with_threshold,
    save_model_bundle,
    tune_logistic_regression,
)


RANDOM_STATE = 42

data_path = (
    project_root
    / "data"
    / "raw"
    / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
)

reports_dir = project_root / "reports"
reports_dir.mkdir(exist_ok=True)

models_dir = project_root / "models"
models_dir.mkdir(exist_ok=True)


def print_metric_dict(title, metrics):
    print(f"\n=== {title} ===")

    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


# =====================================================
# LOAD, CLEAN, AND FEATURE ENGINEER
# =====================================================

df_cleaned = preprocess_pipeline(data_path)
df_final = build_features(df_cleaned)

print("Cleaned shape:", df_cleaned.shape)
print("Final shape after feature engineering:", df_final.shape)

X = df_final.drop(columns=["Churn"])
y = df_final["Churn"]

X_encoded = pd.get_dummies(X, drop_first=True)

print("Encoded feature count:", X_encoded.shape[1])
print("Target distribution:")
print(y.value_counts(normalize=True))

# =====================================================
# TRAIN / VALIDATION / TEST SPLIT
# =====================================================

(
    X_train,
    X_val,
    X_test,
    X_train_val,
    y_train,
    y_val,
    y_test,
    y_train_val,
) = make_train_val_test_split(
    X_encoded,
    y,
    random_state=RANDOM_STATE,
)

print("\n=== SPLIT SHAPES ===")
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)

print("\n=== TARGET RATIO ===")
print("Train:")
print(y_train.value_counts(normalize=True))
print("Validation:")
print(y_val.value_counts(normalize=True))
print("Test:")
print(y_test.value_counts(normalize=True))

# =====================================================
# SCALE DATA FOR LOGISTIC REGRESSION
# =====================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("\nScaling completed.")

# =====================================================
# BASELINE MODEL BENCHMARK
# =====================================================

model_configs = get_baseline_model_configs(
    random_state=RANDOM_STATE,
)

results_df, _ = benchmark_models(
    model_configs=model_configs,
    X_train=X_train,
    y_train=y_train,
    X_val=X_val,
    y_val=y_val,
    X_train_scaled=X_train_scaled,
    X_val_scaled=X_val_scaled,
)

print("\n=== VALIDATION BENCHMARK ===")
print(results_df.to_string(index=False))

benchmark_path = reports_dir / "baseline_05_validation_benchmark.csv"
results_df.to_csv(benchmark_path, index=False)

print(f"\nSaved validation benchmark to: {benchmark_path}")

best_model_name = results_df.iloc[0]["Model"]
print(f"\nBest validation model: {best_model_name}")

# =====================================================
# HYPERPARAMETER TUNING: LOGISTIC REGRESSION BALANCED
# =====================================================

print("\nTuning Logistic Regression Balanced...")

logistic_search = tune_logistic_regression(
    X_train_scaled,
    y_train,
    random_state=RANDOM_STATE,
)

best_logistic_model = logistic_search.best_estimator_

y_val_pred_tuned = best_logistic_model.predict(X_val_scaled)
y_val_prob_tuned = best_logistic_model.predict_proba(X_val_scaled)[:, 1]

val_tuned_core_metrics = compute_binary_metrics(
    y_val,
    y_val_pred_tuned,
    y_val_prob_tuned,
)

tuned_metrics = {
    "Model": "Logistic Regression Balanced Tuned",
    "Best Params": logistic_search.best_params_,
    "Best CV F1": logistic_search.best_score_,
    "Val Accuracy": val_tuned_core_metrics["Accuracy"],
    "Val Precision": val_tuned_core_metrics["Precision"],
    "Val Recall": val_tuned_core_metrics["Recall"],
    "Val F1": val_tuned_core_metrics["F1"],
    "Val ROC_AUC": val_tuned_core_metrics["ROC_AUC"],
}

print("\n=== TUNED LOGISTIC VALIDATION RESULT ===")
for key, value in tuned_metrics.items():
    print(f"{key}: {value}")

tuned_result_path = reports_dir / "baseline_05_tuned_logistic_validation.csv"
pd.DataFrame([tuned_metrics]).to_csv(
    tuned_result_path,
    index=False,
)

print(f"\nSaved tuned logistic result to: {tuned_result_path}")

# =====================================================
# THRESHOLD TUNING ON VALIDATION SET
# =====================================================

best_threshold, best_threshold_f1, threshold_df = find_best_threshold(
    y_val,
    y_val_prob_tuned,
)

threshold_df = threshold_df.sort_values(
    by="F1",
    ascending=False,
)

print("\n=== BEST VALIDATION THRESHOLD ===")
print(f"Best threshold: {best_threshold:.2f}")
print(f"Best validation F1: {best_threshold_f1:.4f}")

print("\nTop 10 thresholds:")
print(threshold_df.head(10).to_string(index=False))

threshold_path = reports_dir / "baseline_05_threshold_tuning.csv"
threshold_df.to_csv(threshold_path, index=False)

print(f"\nSaved threshold tuning result to: {threshold_path}")

y_val_pred_default = (y_val_prob_tuned >= 0.5).astype(int)
y_val_pred_best_threshold = (
    y_val_prob_tuned >= best_threshold
).astype(int)

default_val_metrics = compute_binary_metrics(
    y_val,
    y_val_pred_default,
    y_val_prob_tuned,
    threshold=0.5,
)
tuned_val_metrics = compute_binary_metrics(
    y_val,
    y_val_pred_best_threshold,
    y_val_prob_tuned,
    threshold=best_threshold,
)

print("\n=== VALIDATION: DEFAULT VS TUNED THRESHOLD ===")
print("Default threshold 0.50")
print(f"Precision: {default_val_metrics['Precision']:.4f}")
print(f"Recall: {default_val_metrics['Recall']:.4f}")
print(f"F1: {default_val_metrics['F1']:.4f}")

print(f"\nTuned threshold {best_threshold:.2f}")
print(f"Precision: {tuned_val_metrics['Precision']:.4f}")
print(f"Recall: {tuned_val_metrics['Recall']:.4f}")
print(f"F1: {tuned_val_metrics['F1']:.4f}")

# =====================================================
# FINAL REFIT ON TRAIN + VALIDATION
# =====================================================

final_model, final_scaler = fit_final_logistic_regression(
    X_train_val,
    y_train_val,
    logistic_search.best_params_,
    random_state=RANDOM_STATE,
)

y_test_pred, y_test_prob = predict_with_threshold(
    final_model,
    final_scaler,
    X_test,
    best_threshold,
)
y_test_pred_default, _ = predict_with_threshold(
    final_model,
    final_scaler,
    X_test,
    0.5,
)

# =====================================================
# FINAL TEST EVALUATION
# =====================================================

test_metrics = compute_binary_metrics(
    y_test,
    y_test_pred,
    y_test_prob,
    model_name="Final Logistic Regression Balanced Tuned",
    threshold=best_threshold,
)

default_test_metrics = compute_binary_metrics(
    y_test,
    y_test_pred_default,
    y_test_prob,
    model_name="Final Logistic Regression Balanced Tuned",
    threshold=0.5,
)

print_metric_dict(
    "FINAL TEST EVALUATION: TUNED THRESHOLD",
    test_metrics,
)

print("\nClassification report:")
print(classification_report(y_test, y_test_pred))

print_metric_dict(
    "FINAL TEST EVALUATION: DEFAULT THRESHOLD 0.50",
    default_test_metrics,
)

final_metrics_path = reports_dir / "baseline_05_final_test_metrics.csv"

pd.DataFrame([
    default_test_metrics,
    test_metrics,
]).to_csv(final_metrics_path, index=False)

print(f"\nSaved final test metrics to: {final_metrics_path}")

# =====================================================
# FINAL REPORT ARTIFACTS
# =====================================================

classification_report_path = (
    reports_dir / "baseline_05_classification_report.csv"
)
classification_report_df = classification_report_to_frame(
    y_test,
    y_test_pred,
)
classification_report_df.to_csv(classification_report_path)

print(
    "Saved classification report to: "
    f"{classification_report_path}"
)

confusion_matrix_path = (
    reports_dir / "baseline_05_confusion_matrix.png"
)
save_confusion_matrix(
    y_test,
    y_test_pred,
    confusion_matrix_path,
    threshold=best_threshold,
)
print(f"Saved confusion matrix to: {confusion_matrix_path}")

roc_curve_path = reports_dir / "baseline_05_roc_curve.png"
save_roc_curve(
    y_test,
    y_test_prob,
    roc_curve_path,
)
print(f"Saved ROC curve to: {roc_curve_path}")

pr_curve_path = reports_dir / "baseline_05_pr_curve.png"
save_precision_recall_curve(
    y_test,
    y_test_prob,
    pr_curve_path,
)
print(f"Saved precision-recall curve to: {pr_curve_path}")

# =====================================================
# SAVE FINAL MODEL BUNDLE
# =====================================================

model_bundle_path = models_dir / "baseline_05_final_logistic.joblib"
save_model_bundle(
    output_path=model_bundle_path,
    model=final_model,
    scaler=final_scaler,
    threshold=best_threshold,
    feature_columns=X_encoded.columns,
    best_params=logistic_search.best_params_,
    test_metrics=test_metrics,
)

print(f"Saved final model bundle to: {model_bundle_path}")

print("\n=== BASELINE 05 COMPLETE ===")
print(f"Final model: {test_metrics['Model']}")
print(f"Best params: {logistic_search.best_params_}")
print(f"Threshold: {best_threshold:.2f}")
print(f"Test F1: {test_metrics['F1']:.4f}")
print(f"Test Recall: {test_metrics['Recall']:.4f}")
print(f"Test Precision: {test_metrics['Precision']:.4f}")
print(f"Test ROC-AUC: {test_metrics['ROC_AUC']:.4f}")
