# src/features/feature_engineering.py

import pandas as pd


def create_custom_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    data = df.copy()

    # =====================================================
    # TENURE FEATURES
    # =====================================================

    if 'tenure' in data.columns:

        data['is_new_risk_customer'] = (
            data['tenure'] <= 12
        ).astype(int)

        data['is_loyal_customer'] = (
            data['tenure'] >= 60
        ).astype(int)

        data['tenure_group'] = pd.cut(
            data['tenure'],
            bins=[0, 6, 12, 24, 48, 72],
            labels=[
                '0_6',
                '6_12',
                '12_24',
                '24_48',
                '48_72'
            ]
        )

    # =====================================================
    # CHARGE FEATURES
    # =====================================================

    if (
        'MonthlyCharges' in data.columns
        and 'tenure' in data.columns
    ):

        data['avg_monthly_spent'] = (
            data['MonthlyCharges'] /
            (data['tenure'] + 1)
        )

        data['high_monthly_charge'] = (
            data['MonthlyCharges'] >= 80
        ).astype(int)

    if (
        'TotalCharges' in data.columns
        and 'tenure' in data.columns
    ):

        data['charge_per_month'] = (
            data['TotalCharges'] /
            (data['tenure'] + 1)
        )

        data['high_value_customer'] = (
            data['TotalCharges'] >= 5000
        ).astype(int)

    # =====================================================
    # CONTRACT FEATURES
    # =====================================================

    if 'Contract' in data.columns:

        data['is_month_to_month'] = (
            data['Contract'] == 'Month-to-month'
        ).astype(int)

    # =====================================================
    # INTERNET FEATURES
    # =====================================================

    if 'InternetService' in data.columns:

        data['is_fiber_optic'] = (
            data['InternetService'] == 'Fiber optic'
        ).astype(int)

    # =====================================================
    # SECURITY FEATURES
    # =====================================================

    if 'OnlineSecurity' in data.columns:

        data['has_online_security'] = (
            data['OnlineSecurity'] == 'Yes'
        ).astype(int)

    if 'TechSupport' in data.columns:

        data['has_tech_support'] = (
            data['TechSupport'] == 'Yes'
        ).astype(int)

    # =====================================================
    # PAYMENT RISK FEATURE
    # =====================================================

    if (
        'PaperlessBilling' in data.columns
        and 'PaymentMethod' in data.columns
    ):

        data['high_risk_payment'] = (
            (
                data['PaperlessBilling'] == 'Yes'
            ) &
            (
                data['PaymentMethod']
                == 'Electronic check'
            )
        ).astype(int)

    # =====================================================
    # SERVICES COUNT
    # =====================================================

    service_columns = [

        'PhoneService',
        'OnlineSecurity',
        'OnlineBackup',
        'DeviceProtection',
        'TechSupport',
        'StreamingTV',
        'StreamingMovies'
    ]

    available_services = [

        col for col in service_columns
        if col in data.columns
    ]

    if available_services:

        data['services_count'] = (
            data[available_services] == 'Yes'
        ).sum(axis=1)

    # =====================================================
    # INTERACTION FEATURE
    # =====================================================

    if (
        'InternetService' in data.columns
        and 'Contract' in data.columns
    ):

        data['fiber_monthly_contract'] = (
            (
                data['InternetService']
                == 'Fiber optic'
            ) &
            (
                data['Contract']
                == 'Month-to-month'
            )
        ).astype(int)

    return data


def build_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    return create_custom_features(df)