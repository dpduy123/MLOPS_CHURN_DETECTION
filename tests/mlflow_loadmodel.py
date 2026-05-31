import mlflow
import json
import pandas as pd
from pathlib import Path


def test_model_with_json():
    mlflow.set_tracking_uri("http://localhost:5100")
    MODEL_URI = "models:/Churn_Predict@champion" # hoặc "models:/<Model-ID>" hoặc "models:/Churn_Predict/<version>"

    print(f"--- Đang tải model từ: {MODEL_URI} ---")
    model = mlflow.pyfunc.load_model(MODEL_URI)

    # Data gốc, chưa encode — đúng với những gì wrapper expect
    sample_json = {
        "dataframe_split": {
            "columns": [
                "CustomerID", "Age", "Gender", "Tenure", "Usage Frequency",
                "Support Calls", "Payment Delay", "Subscription Type",
                "Contract Length", "Total Spend", "Last Interaction"
            ],
            "data": [
                [2.0, 30.0, "Female", 39.0, 14.0, 5.0, 18.0, "Standard", "Annual",  932.0, 17.0],
                [3.0, 65.0, "Female", 49.0,  1.0, 10.0, 8.0, "Basic",    "Monthly", 557.0,  6.0],
                [4.0, 55.0, "Female", 14.0,  4.0,  6.0, 18.0, "Basic",   "Quarterly",185.0, 3.0]
            ]
        }
    }

    data_content = sample_json["dataframe_split"]
    df = pd.DataFrame(data_content["data"], columns=data_content["columns"], index=data_content.get("index"))
    print("--- Dữ liệu đã chuẩn bị ---")
    print(df)

    predictions = model.predict(df)
    print(f"\n--- Kết quả dự đoán ---")
    print(type(predictions))
    for i, pred in enumerate(predictions):
        print(f"Row {i+1}: Churn = {pred} ({'Có churn' if pred == 1 else 'Không churn'})")

    assert predictions is not None
    print("\n✅ Test thành công!")

if __name__ == "__main__":
    test_model_with_json()