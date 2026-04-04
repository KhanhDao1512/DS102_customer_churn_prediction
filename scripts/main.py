from pathlib import Path
import pandas as pd

# Bước 1: Lấy đường dẫn của thư mục chứa file code hiện tại
current_dir = Path(__file__).parent

# Bước 2: Dùng dấu gạch chéo (/) để tự động nối đường dẫn. 
# pathlib sẽ tự động đổi dấu gạch này thành \ nếu nó phát hiện đang chạy trên Windows.
# Giả sử bạn muốn lùi ra một cấp (..) rồi chui vào thư mục 'data'
file_path = current_dir.parent / 'data' / 'raw' / 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

# Bước 3: Đọc file
df = pd.read_csv(file_path)

print(df.head())