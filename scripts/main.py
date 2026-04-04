import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '/home/huyy-giaa/Develop/DS102_customer_churn_prediction')))

from src.data.load_data import load_data

if __name__ == "__main__":
    data = load_data("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

    print(data.head())