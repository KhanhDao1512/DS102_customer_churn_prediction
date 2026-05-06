# src/data/preprocessing.py
import pandas as pd
import numpy as np

def load_data(file_path: str) -> pd.DataFrame:
    """Đọc dữ liệu từ file CSV."""
    return pd.read_csv(file_path)

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Làm sạch dữ liệu dựa trên các insight từ EDA.
    
    Các bước thực hiện:
    1. Xóa cột customerID vì không có giá trị học máy.
    2. Chuyển TotalCharges sang dạng số và điền 0 cho 11 dòng NaN.
    3. Chuyển biến mục tiêu 'Churn' sang dạng nhị phân (0, 1).
    4. Xóa cột TotalCharges do tương quan mạnh với tenure và MonthlyCharges.
    """
    data = df.copy()
    
    # 1. Xóa cột không cần thiết
    if 'customerID' in data.columns:
        data = data.drop(columns=['customerID'])
        
    # 2. Xử lý TotalCharges (Mặc dù sẽ drop, nhưng nên xử lý chuẩn trước khi đánh giá)
    if 'TotalCharges' in data.columns:
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        data['TotalCharges'] = data['TotalCharges'].fillna(0)
        
    # 3. Chuyển đổi biến mục tiêu Churn
    if 'Churn' in data.columns:
        data['Churn'] = data['Churn'].map({'Yes': 1, 'No': 0})
        
    # 4. Loại bỏ TotalCharges theo insight từ phân tích đa biến (Heatmap)
    # Giữ lại tenure và MonthlyCharges để tránh đa cộng tuyến
    if 'TotalCharges' in data.columns:
        data = data.drop(columns=['TotalCharges'])
        
    return data

def preprocess_pipeline(file_path: str) -> pd.DataFrame:
    """Pipeline tổng hợp đọc và làm sạch dữ liệu."""
    df = load_data(file_path)
    df_cleaned = clean_data(df)
    return df_cleaned