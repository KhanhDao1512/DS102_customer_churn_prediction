# main.py
import sys
from pathlib import Path

# 1. Lấy đường dẫn tuyệt đối của thư mục chứa file main.py hiện tại (thư mục scripts)
current_dir = Path(__file__).resolve().parent

# 2. Lùi ra ngoài 1 cấp để lấy đường dẫn thư mục gốc của dự án (Ds102_customer_churn_prediction)
project_root = current_dir.parent

# 3. Thêm thư mục gốc này vào danh sách các đường dẫn mà Python sẽ tìm kiếm module
sys.path.append(str(project_root))

from pathlib import Path
from src.data.preprocessing import preprocess_pipeline
from src.features.features import build_features
from src.models.models import split_data, train_random_forest
from src.evaluation.evaluation import evaluate_model, plot_confusion_matrix, plot_roc_curve

def main():
    # 1. Định nghĩa đường dẫn file
    # Lấy thư mục hiện tại (chính là thư mục scripts)
    current_dir = Path.cwd() 
    
    # Dùng .parent để lùi ra ngoài 1 cấp (về Ds102_customer_churn_prediction), 
    # sau đó mới nối với 'data/raw/...'
    file_path = current_dir.parent / 'data' / 'raw' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'
    
    # (Nếu ở cách trước bạn đã định nghĩa biến project_root ở đầu file, 
    # bạn cũng có thể dùng luôn: file_path = project_root / 'data' / ...)
    
    print("1. Đang tải và làm sạch dữ liệu...")
    df_clean = preprocess_pipeline(file_path)
    
    print("2. Đang tạo đặc trưng (Feature Engineering)...")
    df_final = build_features(df_clean)
    
    print("3. Đang chia tập dữ liệu (Train/Test Split)...")
    # Giả định biến mục tiêu của bạn sau bước làm sạch tên là 'Churn'
    X_train, X_test, y_train, y_test = split_data(df_final, target_col='Churn')
    
    print("4. Đang huấn luyện mô hình (Random Forest)...")
    model = train_random_forest(X_train, y_train)
    
    print("5. Đang đánh giá mô hình...")
    y_pred, y_prob = evaluate_model(model, X_test, y_test)
    
    print("6. Đang vẽ biểu đồ đánh giá...")
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_prob)
    
    print("Hoàn tất Pipeline!")

if __name__ == "__main__":
    main()