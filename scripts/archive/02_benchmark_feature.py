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
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier 
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, recall_score, precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
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
X_test_scaled = scaler.transform(X_test)

print('Shape train:', X_train.shape)
print('Shape val:', X_val.shape)
print('Shape test:', X_test.shape)

#===Model===
models = {
    "Logistic Regression": LogisticRegression(max_iter=3000),
    "Logistic Regression Balanced": LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
    "LightGBM": LGBMClassifier(random_state=42, verbose=-1),
    "CatBoost": CatBoostClassifier(random_state=42, verbose=0, allow_writing_files=False)
}

#=== Train & Evaluation ===
results = []
for model_name, model in models.items():
    print(f"\nTraining: {model_name}")
    if "Logistic" in model_name:
        X_fit = X_train_scaled
        X_eval = X_val_scaled
    else:
        X_fit = X_train
        X_eval = X_val

    model.fit(X_fit, y_train)

    y_pred = model.predict(X_eval)
    y_prob = model.predict_proba(X_eval)[:,1]

    results.append({
        'Model':model_name,
        'Accuracy':accuracy_score(y_val, y_pred),
        'Recall':recall_score(y_val, y_pred),
        'Precision':precision_score(y_val, y_pred),
        'F1':f1_score(y_val, y_pred),
        'Roc_Auc':roc_auc_score(y_val, y_prob)
    })

results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by='F1',ascending=False)
print('Kết quả trên validation:', results_df)

best_model_name = results_df.iloc[0]['Model']
print('Best model:', best_model_name)

best_model = models[best_model_name]

scaler_final = StandardScaler()
X_train_val_scaled = scaler_final.fit_transform(X_train_val)
X_test_scaled = scaler_final.transform(X_test)

if "Logistic" in best_model_name:
    best_model.fit(X_train_val_scaled, y_train_val)
    y_test_pred = best_model.predict(X_test_scaled)
    y_test_prob = best_model.predict_proba(X_test_scaled)[:, 1]
else:
    best_model.fit(X_train_val, y_train_val)
    y_test_pred = best_model.predict(X_test)
    y_test_prob = best_model.predict_proba(X_test)[:, 1]

test_accuracy = accuracy_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)
test_roc_auc = roc_auc_score(y_test, y_test_prob)

print("\n=== TEST EVALUATION ===")
print(f"Model: {best_model_name}")
print(f"Accuracy: {test_accuracy:.4f}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1: {test_f1:.4f}")
print(f"ROC-AUC: {test_roc_auc:.4f}")
print(classification_report(y_test, y_test_pred))
print(results_df)