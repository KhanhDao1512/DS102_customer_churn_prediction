# src/features/feature_engineering.py
import pandas as pd

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mã hóa các biến phân loại.
    - Biến có 2 giá trị (Binary): Label Encoding (0/1).
    - Biến có >2 giá trị (Nominal): One-Hot Encoding.
    """
    data = df.copy()
    
    # Danh sách các cột nhị phân (Yes/No hoặc Male/Female)
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    
    for col in binary_cols:
        if col in data.columns:
            if col == 'gender':
                data[col] = data[col].map({'Male': 1, 'Female': 0})
            else:
                data[col] = data[col].map({'Yes': 1, 'No': 0})
                
    # One-Hot Encoding cho các biến phân loại nhiều lớp
    multi_class_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity', 
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
    ]
    
    # Chỉ lấy những cột thực sự tồn tại trong dataframe
    existing_multi_cols = [col for col in multi_class_cols if col in data.columns]
    
    data = pd.get_dummies(data, columns=existing_multi_cols, drop_first=True, dtype=int)
    
    return data

def create_custom_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tạo các đặc trưng mới dựa trên insight từ EDA.
    """
    data = df.copy()
    
    # Insight 1: Khách hàng rời đi cực nhiều ở giai đoạn 0-10 tháng (Phân tích tenure)
    if 'tenure' in data.columns:
        data['is_new_risk_customer'] = data['tenure'].apply(lambda x: 1 if x <= 10 else 0)
        data['is_loyal_customer'] = data['tenure'].apply(lambda x: 1 if x >= 70 else 0)
        
    # Insight 2: Combo dịch vụ gia tăng giữ chân khách hàng (OnlineSecurity & TechSupport)
    # Giả sử sau khi One-hot encoding, ta có các cột 'OnlineSecurity_Yes' và 'TechSupport_Yes'
    if 'OnlineSecurity_Yes' in data.columns and 'TechSupport_Yes' in data.columns:
        data['has_value_added_services'] = (
            (data['OnlineSecurity_Yes'] == 1) | (data['TechSupport_Yes'] == 1)
        ).astype(int)
        
    return data

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Pipeline tổng hợp xây dựng đặc trưng cho mô hình."""
    # 1. Mã hóa biến phân loại
    df_encoded = encode_categorical_features(df)
    
    # 2. Tạo đặc trưng mới từ insight
    df_featured = create_custom_features(df_encoded)
    
    return df_featured