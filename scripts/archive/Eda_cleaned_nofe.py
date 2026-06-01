# main.py

import sys
from pathlib import Path


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

from src.features.features import (
    build_features
)

from src.models.models import (
    split_data,
    compare_models,
    tune_model,
    evaluate_model
)

from src.evaluation.evaluation import (
    plot_confusion_matrix,
    plot_roc_curve
)


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
        / 'processed'
        / 'cleaned_telco_churn.csv'
    )

    print("\n[1] Loading and preprocessing data...")

    df_clean = preprocess_pipeline(file_path)

    print("Dataset shape:", df_clean.shape)


    # =====================================================
    # FEATURE ENGINEERING
    # =====================================================

    print("\n[2] Skipping feature engineering for raw cleaned baseline...")

    df_final = df_clean

    print("Feature engineered shape:", df_final.shape)


    # =====================================================
    # TRAIN TEST SPLIT
    # =====================================================

    print("\n[3] Splitting dataset...")

    X_train, X_test, y_train, y_test = split_data(
        df_final,
        target_col='Churn'
    )

    print("Train shape:", X_train.shape)
    print("Test shape :", X_test.shape)


    # =====================================================
    # MODEL COMPARISON
    # =====================================================

    print("\n[5] Comparing multiple models...")

    comparison_results = compare_models(
        X_train,
        y_train,
        scoring='f1'
    )

    print("\nModel Ranking:")
    print(comparison_results)


    # =====================================================
    # HYPERPARAMETER TUNING
    # =====================================================

    print("\n[6] Hyperparameter tuning...")

    best_model = tune_model(
        X_train,
        y_train,
        model_name='Random Forest',
        scoring='f1'
    )


    # =====================================================
    # FINAL EVALUATION
    # =====================================================

    print("\n[7] Evaluating best model on test set...")

    y_pred = evaluate_model(
        best_model,
        X_test,
        y_test
    )
    

    # =====================================================
    # ROC CURVE + CONFUSION MATRIX
    # =====================================================

    print("\n[8] Generating evaluation plots...")

    y_prob = best_model.predict_proba(X_test)[:, 1]

    plot_confusion_matrix(
        y_test,
        y_pred
    )

    plot_roc_curve(
        y_test,
        y_prob
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