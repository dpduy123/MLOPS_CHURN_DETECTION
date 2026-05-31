import joblib, pandas as pd
from pathlib import Path
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.copy()
        # Tính toán các tỷ lệ
        X['Tenure_Age_Ratio'] = X['Tenure'] / (X['Age'] + 1)
        X['Spend_per_Usage'] = X['Total Spend'] / (X['Usage Frequency'] + 1)
        X['Support_Calls_per_Tenure'] = X['Support Calls'] / (X['Tenure'] + 1)
        
        # Tính toán các nhóm
        X['Spending_Group'] = pd.qcut(X['Total Spend'], q=4, labels=False, duplicates='drop').fillna(0).astype(int)
        X['Tenure_Group'] = pd.cut(X['Tenure'], bins=[0, 12, 24, 36, 100], labels=False).fillna(0).astype(int)
        
        return X

def create_fe_processor_structure():
    """Tạo cấu trúc Pipeline FE (chưa cần fit)."""
    return Pipeline([
        ('feature_engineer', FeatureEngineer())
    ])

def get_fe_processor(data=None, save_path=None, debug=False):
    """Khởi tạo, fit (nếu có data) và lưu Pipeline FE."""
    fe_pipeline = create_fe_processor_structure()
    
    if debug:
        print("DEBUG: Feature Engineer Pipeline đã được khởi tạo.")

    if data is not None:
        if isinstance(data, (str, Path)):
            data = pd.read_csv(data)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("Dữ liệu đầu vào phải là đường dẫn đến file CSV hoặc một DataFrame.")
        fe_pipeline.fit(data)

        if debug:
            print("DEBUG: Feature Engineer Pipeline đã được fit thành công.")

    # Lưu Pipeline
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(fe_pipeline, path)
        if debug:
            print(f"DEBUG: Thành công - Đã lưu FE Pipeline tại {path}")

    return fe_pipeline

if __name__ == "__main__":
    import os
    # Test chạy FE Pipeline trên một phần dữ liệu đã được preprocess
    # Đường dẫn dữ liệu
    indata_path = "data/unit_tests/preprocess/period_1.csv"
    fe_processor_save_path = "outputs/transform_weights/fe_processor.pkl"
    outdata_path = "data/unit_tests/feature_engineering/period_1.csv"
    
    # Gọi hàm và truyền data vào để fit
    fe_processor = get_fe_processor(save_path=fe_processor_save_path, debug=True)

    loaded_fe_processor = joblib.load(fe_processor_save_path)

    # Load dữ liệu đã được preprocess và áp dụng FE Pipeline
    df = pd.read_csv(indata_path)
    transformed_df = loaded_fe_processor.transform(df)

    print("Feature Engineering đã được áp dụng thành công.")
    print(f"Shape dữ liệu sau khi FE: {transformed_df.shape}")

    print("\nDanh sách các cột sau khi transform:")
    print(transformed_df.columns.tolist())

    print("Kiểu dữ liệu của từng cột sau khi transform:")
    print(transformed_df.dtypes)

    # Lưu kết quả
    os.makedirs(os.path.dirname(outdata_path), exist_ok=True)
    transformed_df.to_csv(outdata_path, index=False)
    print(f"Đã lưu dữ liệu sau FE tại: {outdata_path}")
