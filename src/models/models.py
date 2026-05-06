# src/model/model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def split_data(df: pd.DataFrame, target_col: str = 'Churn', test_size: float = 0.2, random_state: int = 42):
    """
    Chia dữ liệu thành tập huấn luyện (Train) và tập kiểm thử (Test).
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train, random_state: int = 42):
    """
    Huấn luyện mô hình Random Forest.
    """
    rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=random_state)
    rf_model.fit(X_train, y_train)
    return rf_model

def train_logistic_regression(X_train, y_train, random_state: int = 42):
    """
    Huấn luyện mô hình Logistic Regression.
    """
    lr_model = LogisticRegression(max_iter=1000, random_state=random_state)
    lr_model.fit(X_train, y_train)
    return lr_model