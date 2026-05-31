import joblib, pandas as pd, yaml
from pathlib import Path
from typing import Optional

def load_periods(
    start: int, 
    end: int, 
    data_root: Path, 
    output: Optional[Path] = None
) -> pd.DataFrame:
    """
    Load các file CSV, gộp lại và tùy chọn lưu ra file output.
    """
    frames = []
    for i in range(start, end + 1):
        file_path = data_root / f"period_{i}.csv"
        if file_path.exists():
            frames.append(pd.read_csv(file_path))
        else:
            print(f"Bỏ qua file không tồn tại: {file_path}")

    if not frames:
        raise FileNotFoundError("Không tìm thấy file nào để load.")

    # Gộp dữ liệu
    df = pd.concat(frames, ignore_index=True)
    df = df.dropna().reset_index(drop=True)
    # Nếu có chỉ định file output, lưu lại
    if output:
        # Đảm bảo thư mục đích tồn tại
        output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output, index=False)
        print(f"Đã lưu dữ liệu gộp vào: {output}")

    return df


class ChurnDataset:
    LABEL_COL = "Churn"
    ID_COL = "CustomerID"

    def get_data(self, data):
        if isinstance(data, str):
            df = pd.read_csv(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError("get_data expects a pandas DataFrame or CSV file path")

        if self.LABEL_COL not in df.columns:
            raise ValueError(f"Missing required label column: {self.LABEL_COL}")
        
        if df[self.LABEL_COL].isnull().any():
            print(f"DEBUG: Tìm thấy {df[self.LABEL_COL].isnull().sum()} giá trị NaN trong cột {self.LABEL_COL}")
            df = df.dropna(subset=[self.LABEL_COL])

        y = df[self.LABEL_COL].astype(int).to_numpy()
        drop_cols = [self.LABEL_COL]
        if self.ID_COL in df.columns:
            drop_cols.append(self.ID_COL)

        X = df.drop(columns=drop_cols)
        return X, y
    

if __name__ == "__main__":
    import os
    from src.data_process.orchestrator import get_process_pipeline
    data_raw = load_periods(1, 10, data_root=Path("data/raw"), output=Path("data/processed/period_1_10_raw.csv"))

    indata_path             = "data/processed/period_1_10_raw.csv"
    full_pipeline_save_path = "outputs/transform_weights/full_pipeline_1_10.pkl"
    outdata_path            = "data/processed/period_1_10_processed.csv"

    indata = pd.read_csv(indata_path)
    print("Dữ liệu thô đã tải:")
    print(indata.head())

    indata.drop(columns=["CustomerID", "Churn"], inplace=True)  
    
    # Gọi hàm và truyền data vào để fit
    full_pipeline = get_process_pipeline(indata, save_path=full_pipeline_save_path, debug=True)

    loaded_full_pipeline = joblib.load(full_pipeline_save_path)

    # Load dữ liệu thô và áp dụng Full Pipeline
    df = pd.read_csv(indata_path)
    transformed_df = loaded_full_pipeline.transform(df)
    transformed_df[["CustomerID", "Churn"]] = df[["CustomerID", "Churn"]]

    os.makedirs(os.path.dirname(outdata_path), exist_ok=True)
    transformed_df.to_csv(outdata_path, index=False)

    print("Full Pipeline đã được áp dụng thành công.")
    print(f"Shape dữ liệu sau khi transform: {transformed_df.shape}")

    print("\nDanh sách các cột sau khi transform:")
    print(transformed_df.columns.tolist())

    print("Kiểu dữ liệu của từng cột sau khi transform:")
    print(transformed_df.dtypes)
