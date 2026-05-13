import pandas as pd


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

        y = df[self.LABEL_COL].astype(int).to_numpy()
        drop_cols = [self.LABEL_COL]
        if self.ID_COL in df.columns:
            drop_cols.append(self.ID_COL)

        X = df.drop(columns=drop_cols).to_numpy(dtype=float)
        return X, y