from pathlib import Path
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
file_path = (
    project_root
    / 'data'
    / 'raw'
    / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
)

import sys
sys.path.append(str(project_root))

import pandas as pd
import numpy as np
from src.data.load_data import load_data
from src.data.preprocessing import preprocess_pipeline
from src.features.features import build_features
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
#===Preprocessing===
df_cleaned = preprocess_pipeline(file_path)
df_final = build_features(df_cleaned)

#===Split train/test===
X = df_final.drop(columns=['Churn'])
y = df_final['Churn']

X = pd.get_dummies(X, drop_first=True)

X_train_val, X_test, y_train_val, y_test = train_test_split(X,y,test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val,test_size=0.25, stratify=y_train_val, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print('Shape train:', X_train.shape)
print('Shape val:', X_val.shape)
print('Shape test:', X_test.shape)

tuning_configs = {
    "Logistic Regression Balanced": {
        "model": LogisticRegression(max_iter=3000),
        "params": {
            "C": [0.01, 0.05, 0.1, 0.5, 1, 2, 5, 10],
            "solver": ["lbfgs", "liblinear"],
            "class_weight": ["balanced"]
        },
        "use_scaled": True
    },

    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "max_depth": [2, 3, 4],
            "min_samples_leaf": [1, 2, 5],
            "subsample": [0.8, 1.0]
        },
        "use_scaled": False
    },

    "XGBoost": {
        "model": XGBClassifier(random_state=42, eval_metric="logloss"),
        "params": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 4, 5],
            "learning_rate": [0.01, 0.05, 0.1],
            "subsample": [0.8, 1.0],
            "colsample_bytree": [0.8, 1.0],
            "scale_pos_weight": [1, 3, 5]
        },
        "use_scaled": False
    },

    "LightGBM": {
        "model": LGBMClassifier(random_state=42, verbose=-1),
        "params": {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.01, 0.05, 0.1],
            "num_leaves": [15, 31, 63],
            "max_depth": [-1, 3, 5],
            "min_child_samples": [10, 20, 30],
            "scale_pos_weight": [1, 3, 5]
        },
        "use_scaled": False
    }
}

def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.1, 0.9, 0.01)

    best_threshold = 0.5
    best_f1 = 0

    for threshold in thresholds:
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred)

        if score > best_f1:
            best_f1 = score
            best_threshold = threshold

    return best_threshold, best_f1

#=== Train & Evaluation ===
results = []
best_estimators = {}
for model_name, config in tuning_configs.items():
    
    print(f"Tuning: {model_name}")

    model = config["model"]
    params = config["params"]
    use_scaled = config["use_scaled"]

    if use_scaled:
        X_fit = X_train_scaled
        X_eval = X_val_scaled
    else:
        X_fit = X_train
        X_eval = X_val

    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=params,
        n_iter=10,
        scoring="f1",
        cv=5,
        random_state=42,
        n_jobs=1
    )

    search.fit(X_fit, y_train)

    best_model = search.best_estimator_

    y_pred = best_model.predict(X_eval)
    y_prob = best_model.predict_proba(X_eval)[:, 1]

    results.append({
        "Model": model_name,
        "Best Params": search.best_params_,
        "Best CV F1": search.best_score_,
        "Val Accuracy": accuracy_score(y_val, y_pred),
        "Val Precision": precision_score(y_val, y_pred),
        "Val Recall": recall_score(y_val, y_pred),
        "Val F1": f1_score(y_val, y_pred),
        "Val ROC-AUC": roc_auc_score(y_val, y_prob)
    })

    best_estimators[model_name] = best_model

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='Val F1',ascending=False)
print("\n=== TUNING RESULTS ON VALIDATION ===")
print(results_df.to_string(index=False))

best_model_name = results_df.iloc[0]['Model']
print('Best model:', best_model_name)

best_params = results_df.iloc[0]["Best Params"]
print("Best params:", best_params)

best_config = tuning_configs[best_model_name]

#=== Select threshold on validation set ===
threshold_model = clone(best_config["model"])
threshold_model.set_params(**best_params)

if best_config["use_scaled"]:
    threshold_model.fit(X_train_scaled, y_train)
    val_prob = threshold_model.predict_proba(X_val_scaled)[:, 1]
else:
    threshold_model.fit(X_train, y_train)
    val_prob = threshold_model.predict_proba(X_val)[:, 1]

best_threshold, best_val_f1 = find_best_threshold(y_val, val_prob)

print("\n=== BEST VALIDATION THRESHOLD ===")
print(f"Threshold: {best_threshold:.2f}")
print(f"Validation F1: {best_val_f1:.4f}")

#=== Refit best model on train + validation ===
final_model = clone(best_config["model"])
final_model.set_params(**best_params)

scaler_final = StandardScaler()
X_train_val_scaled = scaler_final.fit_transform(X_train_val)
X_test_scaled = scaler_final.transform(X_test)

if best_config["use_scaled"]:
    final_model.fit(X_train_val_scaled, y_train_val)
    y_test_prob = final_model.predict_proba(X_test_scaled)[:, 1]
else:
    final_model.fit(X_train_val, y_train_val)
    y_test_prob = final_model.predict_proba(X_test)[:, 1]

y_test_pred = (y_test_prob >= best_threshold).astype(int)
default_test_pred = (y_test_prob >= 0.5).astype(int)

print("\n=== TEST EVALUATION WITH TUNED THRESHOLD ===")
print(f"Model: {best_model_name}")
print(f"Threshold: {best_threshold:.2f}")
print(f"Accuracy: {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall: {recall_score(y_test, y_test_pred):.4f}")
print(f"F1: {f1_score(y_test, y_test_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_test_prob):.4f}")

print(classification_report(y_test, y_test_pred))

print("\n=== TEST EVALUATION WITH DEFAULT THRESHOLD 0.50 ===")
print(f"Model: {best_model_name}")
print("Threshold: 0.50")
print(f"Accuracy: {accuracy_score(y_test, default_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, default_test_pred):.4f}")
print(f"Recall: {recall_score(y_test, default_test_pred):.4f}")
print(f"F1: {f1_score(y_test, default_test_pred):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_test_prob):.4f}")
