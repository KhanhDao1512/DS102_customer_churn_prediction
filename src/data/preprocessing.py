# src/data/preprocessing.py

import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic data cleaning.
    """

    data = df.copy()

    # Drop customerID
    if 'customerID' in data.columns:
        data.drop(columns=['customerID'], inplace=True)

    # Convert TotalCharges to numeric
    if 'TotalCharges' in data.columns:
        data['TotalCharges'] = pd.to_numeric(
            data['TotalCharges'],
            errors='coerce'
        )

    # Handle missing values using the business relation:
    # total charges are approximately monthly charges times tenure.
    if 'TotalCharges' in data.columns:
        data["TotalCharges"] = data["TotalCharges"].fillna(
            data["MonthlyCharges"] * data["tenure"]
        )

    # Encode target
    if 'Churn' in data.columns:
        data['Churn'] = data['Churn'].map({
            'Yes': 1,
            'No': 0
        })

    return data


def preprocess_pipeline(file_path: str) -> pd.DataFrame:
    df = load_data(file_path)
    df_cleaned = clean_data(df)

    return df_cleaned
