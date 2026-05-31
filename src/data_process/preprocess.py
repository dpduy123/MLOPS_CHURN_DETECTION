import os, pandas as pd, joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin, _SetOutputMixin
from src.utils.config_loader import load_data_schema 

# Lưu ý: Trong file này, Schema đã được load rồi chứ không phải là directory. Hãy đảm bảo rằng bạn đã load schema trước khi gọi hàm.
# Định nghĩa Class ép kiểu Int ở ngoài cùng (Cấp module)
class IntCaster(BaseEstimator, TransformerMixin, _SetOutputMixin):
    def __init__(self, int_cols):
        self.int_cols = int_cols
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        for col in self.int_cols:
            if col in X.columns: X[col] = X[col].fillna(0).astype(int)
        return X
    
def create_pre_processor_structure(schema=None, debug=False):
    if schema is None:schema = load_data_schema()
    cat_cols = schema['features']['categorical']
    id_cols = schema['features'].get('id', [])
    int_cols = schema['features']['numerical'].get('discrete', []) + id_cols

    if debug:
        print(f"Các cột categorical: {cat_cols}")
        print(f"Các cột numerical (discrete): {int_cols}")

    cat_encoder = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, dtype=int, handle_unknown='ignore'), cat_cols)
        ],
        remainder='passthrough',
        verbose_feature_names_out=False
    ).set_output(transform="pandas")

    pipeline = Pipeline([
        ('cat_encoder', cat_encoder),
        ('int_caster', IntCaster(int_cols=int_cols)) 
    ])

    
    return pipeline

def get_pre_processor(data, schema=None, save_path=None, debug=False):
    if isinstance(data, (str, Path)):
        data = pd.read_csv(data)
    elif not isinstance(data, pd.DataFrame):
        raise ValueError("Dữ liệu đầu vào phải là đường dẫn đến file CSV hoặc một DataFrame.")
    
    if schema is None: schema = load_data_schema()

    clean_processor = create_pre_processor_structure(schema, debug=debug)
    clean_processor.fit(data)


    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clean_processor, path)
        if debug:
            print(f"Thành công: Đã lưu Pipeline tại {path}")

    return clean_processor

if __name__ == "__main__":
    # Code dùng để test nhanh khi chạy file này trực tiếp, không dùng trong pipeline chính
    # Đường dẫn dữ liệu
    indata_path = "data/raw/period_1.csv"
    processor_save_path = "outputs/transform_weights/pre_processor.pkl"
    outdata_path = "data/unit_tests/preprocess/period_1.csv"
    
    # Gọi hàm và truyền data vào để fit
    indata = pd.read_csv(indata_path)
    indata.drop(columns=["CustomerID", "Churn"], inplace=True)
    processor = get_pre_processor(data=indata, save_path=processor_save_path, debug=True)
    print("Pipeline đã sẵn sàng và đã được fit.")
    
    # Load lại
    loaded_processor = joblib.load(processor_save_path)
        
    # Transform
    df = pd.read_csv(indata_path)
    transformed_df = loaded_processor.transform(df)
    transformed_df[["CustomerID", "Churn"]] = df[["CustomerID", "Churn"]]
    
    # Lưu kết quả
    os.makedirs(os.path.dirname(outdata_path), exist_ok=True)
    transformed_df.to_csv(outdata_path, index=False)
    
    print(f"Đã lưu dữ liệu sạch tại: {outdata_path}")
    print(f"Shape dữ liệu sau khi transform: {transformed_df.shape}")

    # Kiểm tra xem có cột nào bị thay đổi giá trị ngoài mong muốn không
    print("\nDanh sách các cột sau khi transform:")
    print(transformed_df.columns.tolist())

    print("Kiểu dữ liệu của từng cột sau khi transform:")
    print(transformed_df.dtypes)