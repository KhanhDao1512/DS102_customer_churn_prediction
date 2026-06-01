# fix_threshold.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    cross_val_score,
    train_test_split
)
from sklearn.pipeline import Pipeline


# =========================================================
# PROJECT ROOT SETUP
# =========================================================

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

sys.path.append(str(project_root))


# =========================================================
# IMPORT MODULES
# =========================================================

from src.data.preprocessing import preprocess_pipeline
from src.models.models import MODELS
from src.pipeline.pipeline import create_preprocessing_pipeline


# =========================================================
# CONFIG
# =========================================================

LOGISTIC_PARAM_GRID = {

    'model__C': [
        0.01,
        0.05,
        0.1,
        0.5,
        1,
        2,
        5,
        10
    ],

    'model__solver': [
        'lbfgs',
        'liblinear'
    ],

    'model__class_weight': [
        None,
        'balanced'
    ]
}


BASELINE_MODELS = [
    'Logistic Regression',
    'Gradient Boosting',
    'LightGBM',
    'Random Forest',
    'XGBoost'
]


# =========================================================
# PIPELINE
# =========================================================

def build_pipeline_no_smote(X_train, model):

    preprocessor = create_preprocessing_pipeline(X_train)

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return pipeline


def compare_models_no_smote(X_train, y_train, scoring='f1'):

    results = []

    for model_name in BASELINE_MODELS:

        model = MODELS[model_name]

        pipeline = build_pipeline_no_smote(
            X_train,
            model
        )

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring=scoring,
            n_jobs=1
        )

        results.append({
            'Model': model_name,
            'Mean Score': scores.mean(),
            'Std Score': scores.std()
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by='Mean Score',
        ascending=False
    )

    print(results_df)

    return results_df


def tune_logistic_no_smote(
    X_train,
    y_train,
    scoring='f1',
    random_state=42
):

    model = LogisticRegression(max_iter=2000)

    pipeline = build_pipeline_no_smote(
        X_train,
        model
    )

    random_search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=LOGISTIC_PARAM_GRID,
        n_iter=10,
        scoring=scoring,
        cv=5,
        n_jobs=1,
        random_state=random_state
    )

    random_search.fit(
        X_train,
        y_train
    )

    print("\n=== BEST PARAMETERS ===")
    print(random_search.best_params_)

    print("\n=== BEST CV SCORE ===")
    print(random_search.best_score_)

    return random_search.best_estimator_


# =========================================================
# THRESHOLD
# =========================================================

def find_best_threshold(y_true, y_prob):

    thresholds = np.arange(
        0.1,
        0.9,
        0.01
    )

    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:

        y_pred = (
            y_prob >= threshold
        ).astype(int)

        score = f1_score(
            y_true,
            y_pred
        )

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1


def evaluate_with_threshold(model, X_test, y_test, threshold):

    y_prob = model.predict_proba(X_test)[:, 1]

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    metrics = {
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1': f1_score(y_test, y_pred),
        'ROC-AUC': roc_auc_score(y_test, y_prob),
        'PR-AUC': average_precision_score(y_test, y_prob)
    }

    print("\n=== TEST PERFORMANCE ===")
    print(f"Threshold: {threshold:.2f}")

    for metric, score in metrics.items():
        print(f"{metric}: {score:.4f}")

    return y_pred, y_prob, metrics


# =========================================================
# MAIN WORKFLOW
# =========================================================

def main():

    # =====================================================
    # LOAD DATA
    # =====================================================

    file_path = (
        project_root
        / 'data'
        / 'raw'
        / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    )

    print("\n[1] Loading and preprocessing data...")

    df_clean = preprocess_pipeline(file_path)

    print("Dataset shape:", df_clean.shape)


    # =====================================================
    # RAW CLEANED BASELINE
    # =====================================================

    print("\n[2] Skipping feature engineering...")

    df_final = df_clean

    print("Raw cleaned shape:", df_final.shape)


    # =====================================================
    # TRAIN / VALIDATION / TEST SPLIT
    # =====================================================

    print("\n[3] Splitting dataset into train / validation / test...")

    X = df_final.drop(columns=['Churn'])
    y = df_final['Churn']

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        stratify=y_train_full,
        random_state=42
    )

    print("Train shape     :", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape      :", X_test.shape)


    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    print("\n[4] Comparing models without SMOTE...")

    comparison_results = compare_models_no_smote(
        X_train,
        y_train,
        scoring='f1'
    )

    print("\nModel Ranking:")
    print(comparison_results)


    # =====================================================
    # LOGISTIC REGRESSION TUNING
    # =====================================================

    print("\n[5] Tuning Logistic Regression without SMOTE...")

    best_model = tune_logistic_no_smote(
        X_train,
        y_train,
        scoring='f1'
    )


    # =====================================================
    # VALIDATION THRESHOLD
    # =====================================================

    print("\n[6] Selecting threshold on validation set...")

    val_prob = best_model.predict_proba(X_val)[:, 1]

    best_threshold, val_f1 = find_best_threshold(
        y_val,
        val_prob
    )

    print("\n=== BEST VALIDATION THRESHOLD ===")
    print(f"Threshold: {best_threshold:.2f}")
    print(f"Validation F1: {val_f1:.4f}")


    # =====================================================
    # FINAL TEST EVALUATION
    # =====================================================

    print("\n[7] Evaluating on test set with validation threshold...")

    evaluate_with_threshold(
        best_model,
        X_test,
        y_test,
        best_threshold
    )


    # =====================================================
    # DONE
    # =====================================================

    print("\nPipeline completed successfully!")


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":
    main()
