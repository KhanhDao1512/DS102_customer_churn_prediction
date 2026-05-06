# src/evaluation/evaluation.py
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

def evaluate_model(model, X_test, y_test):
    """
    Đánh giá mô hình bằng các chỉ số phân loại và in ra báo cáo.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]  # Xác suất cho lớp 1 (Churn)
    
    print("=== BÁO CÁO PHÂN LOẠI (CLASSIFICATION REPORT) ===")
    print(classification_report(y_test, y_pred))
    
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc_score:.4f}")
    
    return y_pred, y_prob

def plot_confusion_matrix(y_test, y_pred):
    """
    Vẽ ma trận nhầm lẫn (Confusion Matrix).
    """
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
    plt.ylabel('Thực tế')
    plt.xlabel('Dự đoán')
    plt.show()

def plot_roc_curve(y_test, y_prob):
    """
    Vẽ đường cong ROC.
    """
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    auc_score = roc_auc_score(y_test, y_prob)
    
    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='orange', label=f'ROC curve (area = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.show()