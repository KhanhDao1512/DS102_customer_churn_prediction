import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.evaluation.metrics import compute_binary_metrics


LOGISTIC_PARAM_GRID = {
    "C": [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
    "solver": ["lbfgs", "liblinear"],
    "class_weight": ["balanced"],
}


def get_baseline_model_configs(random_state=42):
    """Return baseline model configs and whether each model needs scaled data."""

    return {
        "Logistic Regression": {
            "model": LogisticRegression(
                max_iter=3000,
                random_state=random_state,
            ),
            "use_scaled": True,
        },
        "Logistic Regression Balanced": {
            "model": LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=random_state,
            ),
            "use_scaled": True,
        },
        "Naive Bayes": {
            "model": GaussianNB(),
            "use_scaled": False,
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(
                random_state=random_state,
            ),
            "use_scaled": False,
        },
        "Random Forest": {
            "model": RandomForestClassifier(
                random_state=random_state,
            ),
            "use_scaled": False,
        },
        "AdaBoost": {
            "model": AdaBoostClassifier(
                random_state=random_state,
            ),
            "use_scaled": False,
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(
                random_state=random_state,
            ),
            "use_scaled": False,
        },
        "XGBoost": {
            "model": XGBClassifier(
                random_state=random_state,
                eval_metric="logloss",
            ),
            "use_scaled": False,
        },
        "LightGBM": {
            "model": LGBMClassifier(
                random_state=random_state,
                verbose=-1,
            ),
            "use_scaled": False,
        },
    }


def benchmark_models(
    model_configs,
    X_train,
    y_train,
    X_val,
    y_val,
    X_train_scaled,
    X_val_scaled,
):
    """Fit configured models and evaluate them on the validation set."""

    results = []
    trained_models = {}

    for model_name, config in model_configs.items():
        print(f"\nTraining: {model_name}")

        model = clone(config["model"])

        if config["use_scaled"]:
            X_fit = X_train_scaled
            X_eval = X_val_scaled
        else:
            X_fit = X_train
            X_eval = X_val

        model.fit(X_fit, y_train)

        y_val_pred = model.predict(X_eval)

        if hasattr(model, "predict_proba"):
            y_val_prob = model.predict_proba(X_eval)[:, 1]
        else:
            y_val_prob = None

        metrics = compute_binary_metrics(
            y_val,
            y_val_pred,
            y_val_prob,
            model_name=model_name,
        )

        if y_val_prob is None:
            metrics["ROC_AUC"] = np.nan

        results.append(metrics)
        trained_models[model_name] = model

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="F1",
        ascending=False,
    )

    return results_df, trained_models


def tune_logistic_regression(
    X_train_scaled,
    y_train,
    random_state=42,
    scoring="f1",
    cv=5,
    n_iter=20,
    n_jobs=1,
):
    """Tune balanced logistic regression on scaled training data."""

    search = RandomizedSearchCV(
        estimator=LogisticRegression(
            max_iter=5000,
            random_state=random_state,
        ),
        param_distributions=LOGISTIC_PARAM_GRID,
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    search.fit(X_train_scaled, y_train)

    return search


def fit_final_logistic_regression(
    X_train,
    y_train,
    best_params,
    random_state=42,
):
    """Fit final logistic regression with a scaler on the provided data."""

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=5000,
        random_state=random_state,
        **best_params,
    )
    model.fit(X_train_scaled, y_train)

    return model, scaler


def predict_with_threshold(model, scaler, X, threshold):
    """Predict probabilities and labels with a custom threshold."""

    X_scaled = scaler.transform(X)
    y_prob = model.predict_proba(X_scaled)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    return y_pred, y_prob


def save_model_bundle(
    output_path,
    model,
    scaler,
    threshold,
    feature_columns,
    best_params,
    test_metrics,
):
    """Save model, scaler, threshold, and metadata as a reusable bundle."""

    model_bundle = {
        "model": model,
        "scaler": scaler,
        "threshold": threshold,
        "feature_columns": list(feature_columns),
        "best_params": best_params,
        "test_metrics": test_metrics,
    }

    joblib.dump(model_bundle, output_path)

    return model_bundle
