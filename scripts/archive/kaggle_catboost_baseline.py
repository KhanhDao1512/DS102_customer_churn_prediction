# kaggle_catboost_baseline.py

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.model_selection import train_test_split


# =========================================================
# PROJECT ROOT SETUP
# =========================================================

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent

sys.path.append(str(project_root))


# =========================================================
# CONFIG
# =========================================================

RANDOM_STATE = 42

SERVICE_COLUMNS = [
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies'
]

DROP_COLUMNS = [
    'customerID',
    'gender',
    'PhoneService'
]


# =========================================================
# DATA
# =========================================================

def load_kaggle_style_data(file_path):

    data = pd.read_csv(file_path)

    data = data.drop(
        columns=[
            col for col in DROP_COLUMNS
            if col in data.columns
        ]
    )

    if 'TotalCharges' in data.columns:
        data['TotalCharges'] = pd.to_numeric(
            data['TotalCharges'],
            errors='coerce'
        )

        data['TotalCharges'] = data['TotalCharges'].fillna(
            data['TotalCharges'].median()
        )

    if 'Churn' in data.columns:
        data['Churn'] = data['Churn'].map({
            'Yes': 1,
            'No': 0
        })

    for col in SERVICE_COLUMNS:
        if col in data.columns:
            data[col] = data[col].replace(
                'No internet service',
                'No'
            )

    if 'MultipleLines' in data.columns:
        data['MultipleLines'] = data['MultipleLines'].replace(
            'No phone service',
            'No'
        )

    return data


def split_train_validation_test(df, target_col='Churn'):

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        stratify=y_train_full,
        random_state=RANDOM_STATE
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def get_cat_features(X):

    return [
        index
        for index, dtype in enumerate(X.dtypes)
        if dtype == 'object'
    ]


# =========================================================
# HYPOTHESIS TESTING
# =========================================================

def run_hypothesis_tests(df, target_col='Churn'):

    print("\n[2] Hypothesis testing against Churn...")

    try:
        from scipy.stats import chi2_contingency, mannwhitneyu
    except ImportError:
        print("scipy is not installed. Skipping hypothesis tests.")
        return

    results = []

    categorical_cols = (
        df
        .drop(columns=[target_col])
        .select_dtypes(include=['object'])
        .columns
        .tolist()
    )

    numerical_cols = (
        df
        .drop(columns=[target_col])
        .select_dtypes(include=['int64', 'float64'])
        .columns
        .tolist()
    )

    for col in categorical_cols:

        contingency_table = pd.crosstab(
            df[col],
            df[target_col]
        )

        _, p_value, _, _ = chi2_contingency(
            contingency_table
        )

        results.append({
            'Feature': col,
            'Test': 'Chi-square',
            'P-value': p_value
        })

    churn_yes = df[target_col] == 1
    churn_no = df[target_col] == 0

    for col in numerical_cols:

        _, p_value = mannwhitneyu(
            df.loc[churn_yes, col],
            df.loc[churn_no, col],
            alternative='two-sided'
        )

        results.append({
            'Feature': col,
            'Test': 'Mann-Whitney U',
            'P-value': p_value
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by='P-value',
        ascending=True
    )

    print(results_df.to_string(index=False))


# =========================================================
# MODEL
# =========================================================

def make_catboost(scale_pos_weight):

    return CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=4,
        l2_leaf_reg=5,
        loss_function='Logloss',
        eval_metric='AUC',
        scale_pos_weight=scale_pos_weight,
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False
    )


def calculate_scale_pos_weight(y_train):

    negative_count = (y_train == 0).sum()
    positive_count = (y_train == 1).sum()

    return negative_count / positive_count


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


def evaluate_predictions(y_true, y_prob, threshold):

    y_pred = (
        y_prob >= threshold
    ).astype(int)

    return {
        'Threshold': threshold,
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred),
        'Recall': recall_score(y_true, y_pred),
        'F1': f1_score(y_true, y_pred),
        'ROC-AUC': roc_auc_score(y_true, y_prob),
        'PR-AUC': average_precision_score(y_true, y_prob)
    }


def train_weight_candidates(X_train, X_val, y_train, y_val, cat_features):

    ratio_weight = calculate_scale_pos_weight(y_train)

    weight_candidates = [
        1,
        ratio_weight,
        3,
        5
    ]

    results = []
    fitted_models = {}

    for weight in weight_candidates:

        model = make_catboost(
            scale_pos_weight=weight
        )

        train_pool = Pool(
            X_train,
            y_train,
            cat_features=cat_features
        )

        val_pool = Pool(
            X_val,
            y_val,
            cat_features=cat_features
        )

        model.fit(
            train_pool,
            eval_set=val_pool,
            use_best_model=True,
            early_stopping_rounds=50
        )

        val_prob = model.predict_proba(X_val)[:, 1]

        best_threshold, best_val_f1 = find_best_threshold(
            y_val,
            val_prob
        )

        metrics = evaluate_predictions(
            y_val,
            val_prob,
            best_threshold
        )

        metrics['Scale Pos Weight'] = weight
        metrics['Validation F1'] = best_val_f1
        metrics['Best Iteration'] = model.get_best_iteration()

        results.append(metrics)
        fitted_models[weight] = model

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by='Validation F1',
        ascending=False
    )

    best_weight = results_df.iloc[0]['Scale Pos Weight']
    best_threshold = results_df.iloc[0]['Threshold']
    best_model = fitted_models[best_weight]

    return best_model, best_weight, best_threshold, results_df


# =========================================================
# SHAP
# =========================================================

def save_catboost_shap_importance(model, X_test, y_test, cat_features):

    print("\n[7] Calculating CatBoost SHAP importance...")

    output_dir = project_root / 'reports'
    output_dir.mkdir(exist_ok=True)

    test_pool = Pool(
        X_test,
        y_test,
        cat_features=cat_features
    )

    shap_values = model.get_feature_importance(
        data=test_pool,
        type='ShapValues'
    )

    feature_shap_values = shap_values[:, :-1]

    shap_importance = pd.DataFrame({
        'Feature': X_test.columns,
        'Mean Abs SHAP': np.abs(feature_shap_values).mean(axis=0)
    })

    shap_importance = shap_importance.sort_values(
        by='Mean Abs SHAP',
        ascending=False
    )

    output_path = output_dir / 'catboost_shap_importance.csv'

    shap_importance.to_csv(
        output_path,
        index=False
    )

    print(shap_importance.head(15).to_string(index=False))
    print(f"\nSaved SHAP importance to: {output_path}")

    return shap_importance


# =========================================================
# MAIN
# =========================================================

def main():

    file_path = (
        project_root
        / 'data'
        / 'raw'
        / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    )

    print("\n[1] Loading Kaggle-style cleaned data...")

    df = load_kaggle_style_data(file_path)

    print("Dataset shape:", df.shape)
    print("Churn distribution:")
    print(df['Churn'].value_counts(normalize=True))

    run_hypothesis_tests(df)

    print("\n[3] Splitting train / validation / test...")

    X_train, X_val, X_test, y_train, y_val, y_test = (
        split_train_validation_test(df)
    )

    cat_features = get_cat_features(X_train)

    print("Train shape     :", X_train.shape)
    print("Validation shape:", X_val.shape)
    print("Test shape      :", X_test.shape)
    print("Categorical features:", list(X_train.columns[cat_features]))

    print("\n[4] Training CatBoost scale_pos_weight candidates...")

    best_model, best_weight, best_threshold, validation_results = (
        train_weight_candidates(
            X_train,
            X_val,
            y_train,
            y_val,
            cat_features
        )
    )

    print("\n=== VALIDATION RESULTS ===")
    print(validation_results.to_string(index=False))

    print("\n=== SELECTED MODEL ===")
    print(f"Scale pos weight: {best_weight:.4f}")
    print(f"Threshold: {best_threshold:.2f}")

    print("\n[5] Evaluating selected model on test set...")

    test_prob = best_model.predict_proba(X_test)[:, 1]

    test_metrics = evaluate_predictions(
        y_test,
        test_prob,
        best_threshold
    )

    print("\n=== TEST PERFORMANCE ===")

    for metric, score in test_metrics.items():
        print(f"{metric}: {score:.4f}")

    print("\n[6] Comparing with default threshold 0.50...")

    default_metrics = evaluate_predictions(
        y_test,
        test_prob,
        threshold=0.5
    )

    print("\n=== TEST PERFORMANCE AT THRESHOLD 0.50 ===")

    for metric, score in default_metrics.items():
        print(f"{metric}: {score:.4f}")

    save_catboost_shap_importance(
        best_model,
        X_test,
        y_test,
        cat_features
    )

    print("\nPipeline completed successfully!")


if __name__ == "__main__":
    main()
