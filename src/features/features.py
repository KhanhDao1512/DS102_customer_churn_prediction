# src/features/features.py

import pandas as pd


def create_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()

    # =====================================================
    # TENURE FEATURES
    # =====================================================

    if "tenure" in data.columns:
        data["is_new_risk_customer"] = (
            data["tenure"] <= 12
        ).astype(int)

        data["is_loyal_customer"] = (
            data["tenure"] >= 60
        ).astype(int)

        data["tenure_group"] = pd.cut(
            data["tenure"],
            bins=[-1, 6, 12, 24, 48, 72],
            labels=[
                "0_6",
                "6_12",
                "12_24",
                "24_48",
                "48_72",
            ],
        )

    # =====================================================
    # CHARGE FEATURES
    # =====================================================

    if "MonthlyCharges" in data.columns:
        data["high_monthly_charge"] = (
            data["MonthlyCharges"] >= 80
        ).astype(int)

    if (
        "TotalCharges" in data.columns
        and "tenure" in data.columns
    ):
        data["charge_per_month"] = (
            data["TotalCharges"] / (data["tenure"] + 1)
        )

        data["high_value_customer"] = (
            data["TotalCharges"] >= 5000
        ).astype(int)

    # =====================================================
    # CONTRACT FEATURES
    # =====================================================

    if "Contract" in data.columns:
        data["is_month_to_month"] = (
            data["Contract"] == "Month-to-month"
        ).astype(int)

        data["is_long_term_contract"] = (
            data["Contract"].isin(["One year", "Two year"])
        ).astype(int)

    # =====================================================
    # INTERNET FEATURES
    # =====================================================

    if "InternetService" in data.columns:
        data["is_fiber_optic"] = (
            data["InternetService"] == "Fiber optic"
        ).astype(int)

        data["has_internet_service"] = (
            data["InternetService"] != "No"
        ).astype(int)

    # =====================================================
    # SECURITY / SUPPORT FEATURES
    # =====================================================

    protection_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
    ]

    available_protection_cols = [
        col for col in protection_cols
        if col in data.columns
    ]

    if available_protection_cols:
        data["protection_services_count"] = (
            data[available_protection_cols] == "Yes"
        ).sum(axis=1)

        data["has_no_protection_service"] = (
            data["protection_services_count"] == 0
        ).astype(int)

    if "OnlineSecurity" in data.columns:
        data["has_online_security"] = (
            data["OnlineSecurity"] == "Yes"
        ).astype(int)

    if "TechSupport" in data.columns:
        data["has_tech_support"] = (
            data["TechSupport"] == "Yes"
        ).astype(int)

    # =====================================================
    # STREAMING FEATURES
    # =====================================================

    streaming_cols = [
        "StreamingTV",
        "StreamingMovies",
    ]

    available_streaming_cols = [
        col for col in streaming_cols
        if col in data.columns
    ]

    if available_streaming_cols:
        data["streaming_services_count"] = (
            data[available_streaming_cols] == "Yes"
        ).sum(axis=1)

        data["has_any_streaming"] = (
            data["streaming_services_count"] > 0
        ).astype(int)

    # =====================================================
    # PAYMENT FEATURES
    # =====================================================

    if "PaymentMethod" in data.columns:
        data["is_auto_payment"] = (
            data["PaymentMethod"].isin([
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ])
        ).astype(int)

        data["is_electronic_check"] = (
            data["PaymentMethod"] == "Electronic check"
        ).astype(int)

    if (
        "PaperlessBilling" in data.columns
        and "PaymentMethod" in data.columns
    ):
        data["high_risk_payment"] = (
            (data["PaperlessBilling"] == "Yes")
            & (data["PaymentMethod"] == "Electronic check")
        ).astype(int)

    # =====================================================
    # SERVICES COUNT
    # =====================================================

    service_count = pd.Series(0, index=data.index)

    if "PhoneService" in data.columns:
        service_count += (data["PhoneService"] == "Yes").astype(int)

    if "InternetService" in data.columns:
        service_count += (data["InternetService"] != "No").astype(int)

    extra_service_cols = [
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    available_extra_service_cols = [
        col for col in extra_service_cols
        if col in data.columns
    ]

    if available_extra_service_cols:
        service_count += (
            data[available_extra_service_cols] == "Yes"
        ).sum(axis=1)

    data["services_count"] = service_count

    # =====================================================
    # INTERACTION FEATURES
    # =====================================================

    if (
        "InternetService" in data.columns
        and "Contract" in data.columns
    ):
        data["fiber_monthly_contract"] = (
            (data["InternetService"] == "Fiber optic")
            & (data["Contract"] == "Month-to-month")
        ).astype(int)

    if (
        "tenure" in data.columns
        and "Contract" in data.columns
    ):
        data["new_monthly_contract"] = (
            (data["tenure"] <= 12)
            & (data["Contract"] == "Month-to-month")
        ).astype(int)

    return data


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    return create_custom_features(df)