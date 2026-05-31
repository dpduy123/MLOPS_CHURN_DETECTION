import joblib, pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from src.data_process.preprocess import create_pre_processor_structure
from src.data_process.feature_engineer import create_fe_processor_structure
from src.utils.config_loader import load_data_schema


def create_struct_process_pipeline(schema=None):
    if schema is None: 
        schema  = load_data_schema()

    pre_proc    = create_pre_processor_structure(schema)
    fe_proc     = create_fe_processor_structure()

    full_pipeline = Pipeline([
        ('pre_process', pre_proc),
        ('fe_process', fe_proc)
    ])
    
    return full_pipeline

def get_process_pipeline(data, schema=None, save_path=None, debug=False):
    full_pipeline = create_struct_process_pipeline(schema)
    
    if debug:
        print("DEBUG: Đang bắt đầu fit full pipeline...")

    if data is not None:
        if isinstance(data, (str, Path)):
            data = pd.read_csv(data)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("Dữ liệu đầu vào phải là đường dẫn đến file CSV hoặc một DataFrame.")

        if debug:
            print("DEBUG: Feature Engineer Pipeline đã được fit thành công.")

    full_pipeline.fit(data)
    
    if debug:
        print("DEBUG: Full pipeline đã được fit thành công.")

    # 3. Lưu Pipeline
    if save_path:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(full_pipeline, path)
        if debug:
            print(f"DEBUG: Thành công - Đã lưu Full Pipeline tại {path}")

    return full_pipeline

if __name__ == "__main__":
    import os
    indata_path             = "data/raw/period_1.csv"
    full_pipeline_save_path = "outputs/transform_weights/full_pipeline.pkl"
    outdata_path            = "data/unit_tests/pipeline/period_1.csv"
    
    # Gọi hàm và truyền data vào để fit
    full_pipeline = get_process_pipeline(indata_path, save_path=full_pipeline_save_path, debug=True)

    loaded_full_pipeline = joblib.load(full_pipeline_save_path)

    # Load dữ liệu thô và áp dụng Full Pipeline
    df = pd.read_csv(indata_path)
    transformed_df = loaded_full_pipeline.transform(df)

    # Lưu kết quả
    os.makedirs(os.path.dirname(outdata_path), exist_ok=True)
    transformed_df.to_csv(outdata_path, index=False)

    print("Full Pipeline đã được áp dụng thành công.")
    print(f"Shape dữ liệu sau khi transform: {transformed_df.shape}")

    print("\nDanh sách các cột sau khi transform:")
    print(transformed_df.columns.tolist())

    print("Kiểu dữ liệu của từng cột sau khi transform:")