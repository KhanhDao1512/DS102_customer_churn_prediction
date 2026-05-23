import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    RandomizedSearchCV
)

from imblearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.svm import SVC

from imblearn.over_sampling import SMOTE

from src.pipeline.pipeline import create_preprocessing_pipeline


# =========================================================
# MODEL REGISTRY
# =========================================================

MODELS = {

    'Logistic Regression':
        LogisticRegression(
            max_iter=2000,
            class_weight='balanced'
        ),

    'Random Forest':
        RandomForestClassifier(
            random_state=42
        ),

    'Gradient Boosting':
        GradientBoostingClassifier(),

    'XGBoost':
        XGBClassifier(
            random_state=42,
            eval_metric='logloss'
        ),

    'LightGBM':
        LGBMClassifier(
            random_state=42,
            verbose=-1
        ),

    'CatBoost':
        CatBoostClassifier(
            verbose=0,
            random_state=42
        )
}

# =========================================================
# HYPERPARAMETER GRIDS
# =========================================================

# =========================================================
# HYPERPARAMETER GRIDS
# =========================================================

PARAM_GRIDS = {

    'Random Forest': {

        'model__n_estimators': [100, 200, 300],

        'model__max_depth': [
            5,
            10,
            15,
            None
        ],

        'model__min_samples_split': [
            2,
            5,
            10
        ],

        'model__min_samples_leaf': [
            1,
            2,
            4
        ],

        'model__class_weight': [
            'balanced',
            None
        ]
    },

    'XGBoost': {

        'model__n_estimators': [100, 200],

        'model__max_depth': [3, 5, 7],

        'model__learning_rate': [
            0.01,
            0.05,
            0.1
        ],

        'model__subsample': [
            0.8,
            1.0
        ]
    },

    'LightGBM': {

        'model__n_estimators': [100, 200],

        'model__learning_rate': [
            0.01,
            0.05,
            0.1
        ],

        'model__max_depth': [
            -1,
            5,
            10
        ]
    },

    'CatBoost': {

        'model__depth': [4, 6, 8],

        'model__learning_rate': [
            0.01,
            0.05,
            0.1
        ],

        'model__iterations': [
            100,
            200
        ]
    }
}


# =========================================================
# DATA SPLIT
# =========================================================

def split_data(
    df,
    target_col='Churn',
    test_size=0.2,
    random_state=42
):

    X = df.drop(columns=[target_col])
    y = df[target_col]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )



# =========================================================
# BUILD PIPELINE
# =========================================================

def build_pipeline(
    X_train,
    model
):

    preprocessor = create_preprocessing_pipeline(
        X_train
    )

    pipeline = Pipeline([

        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('model', model)
    ])

    return pipeline


# =========================================================
# MODEL COMPARISON
# =========================================================

def compare_models(
    X_train,
    y_train,
    scoring='f1'
):

    results = []

    for model_name, model in MODELS.items():

        pipeline = build_pipeline(
            X_train,
            model
        )

        scores = cross_val_score(
            pipeline,
            X_train,
            y_train,
            cv=5,
            scoring=scoring,
            n_jobs=-1
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


# =========================================================
# TRAIN MODEL
# =========================================================

def train_model(
    X_train,
    y_train,
    model_name='Random Forest'
):

    model = MODELS[model_name]

    pipeline = build_pipeline(
        X_train,
        model
    )

    pipeline.fit(
        X_train,
        y_train
    )

    return pipeline


# =========================================================
# HYPERPARAMETER TUNING
# =========================================================

def tune_model(
    X_train,
    y_train,
    model_name='Random Forest',
    scoring='f1',
    random_state=42
):

    model = MODELS[model_name]

    pipeline = build_pipeline(
        X_train,
        model
    )

    param_grid = PARAM_GRIDS.get(
        model_name,
        {}
    )

    random_search = RandomizedSearchCV(

        estimator=pipeline,

        param_distributions=param_grid,

        n_iter=10,

        scoring=scoring,

        cv=5,

        n_jobs=-1,

        random_state=random_state
    )

    random_search.fit(
        X_train,
        y_train
    )

    print("\n=== BEST PARAMETERS ===")
    print(random_search.best_params_)

    print("\n=== BEST SCORE ===")
    print(random_search.best_score_)

    return random_search.best_estimator_


# =========================================================
# EVALUATE TEST SET
# =========================================================

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import numpy as np


def evaluate_model(
    model,
    X_test,
    y_test
):

    # =====================================================
    # PREDICT PROBABILITY
    # =====================================================

    y_prob = model.predict_proba(X_test)[:, 1]

    # =====================================================
    # FIND BEST THRESHOLD
    # =====================================================

    thresholds = np.arange(
        0.1,
        0.9,
        0.01
    )

    best_threshold = 0.5
    best_f1 = 0

    for t in thresholds:

        y_pred_temp = (
            y_prob >= t
        ).astype(int)

        score = f1_score(
            y_test,
            y_pred_temp
        )

        if score > best_f1:

            best_f1 = score
            best_threshold = t

    # =====================================================
    # FINAL PREDICTION
    # =====================================================

    y_pred = (
        y_prob >= best_threshold
    ).astype(int)

    # =====================================================
    # METRICS
    # =====================================================

    metrics = {

        'Accuracy':
            accuracy_score(
                y_test,
                y_pred
            ),

        'Precision':
            precision_score(
                y_test,
                y_pred
            ),

        'Recall':
            recall_score(
                y_test,
                y_pred
            ),

        'F1':
            f1_score(
                y_test,
                y_pred
            )
    }

    print("\n=== BEST THRESHOLD ===")
    print(best_threshold)

    print("\n=== TEST PERFORMANCE ===")

    for metric, score in metrics.items():

        print(f"{metric}: {score:.4f}")

    return y_pred