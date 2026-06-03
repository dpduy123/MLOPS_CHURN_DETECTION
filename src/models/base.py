import joblib, mlflow, pandas as pd, yaml

class BaseModel:
    def fit(self, x, y):
        self.model.fit(x, y)

    def predict_proba(self, x):
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(x)[:, 1]
        else:
            return self.model.predict(x)


    def predict(self, x):
        return self.model.predict(x)


    def save(self, path):
        # Chỉ dump Sklearn Model chứ không phải BaseModel
        joblib.dump(self.model, path)
        # Khi load lên, hãy treat như một Sklearn model thông thường

    def load(self, path):
        self.model = joblib.load(path)

class BaseSklearnWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import os
        import joblib
        import yaml

        def fix_path(path):
            return os.path.normpath(path.replace("\\", "/")) if path else None

        # Nạp artifacts
        self.preprocess = joblib.load(fix_path(context.artifacts.get("preprocess_path")))
        self.model      = joblib.load(fix_path(context.artifacts.get("model_path")))

        with open(fix_path(context.artifacts.get("data_schema")), "r") as f:
            self.schema = yaml.safe_load(f)
            
        self.feature_cols = self._extract_features(self.schema)

    def _ensure_loaded(self, context):
        """Kiểm tra xem các thuộc tính đã được load chưa, nếu chưa thì load ngay"""
        if not hasattr(self, 'preprocess'):
            self.load_context(context)

    def _extract_features(self, schema):
        """Hàm helper để gom tất cả các cột input từ file YAML"""

        features = []
        num_cols = schema['features'].get('numerical', {})

        features.extend(schema['features'].get('categorical', []))
        features.extend(num_cols.get('continuous', []))
        features.extend(num_cols.get('discrete', []))
        
        return features

    def _validate_input(self, model_input):
        if not isinstance(model_input, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame.")
        
        # Kiểm tra các cột bắt buộc (feature_cols)
        missing_cols = set(self.feature_cols) - set(model_input.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in input data: {missing_cols}")
        
        # Chỉ trả về các cột cần thiết, đúng thứ tự
        return model_input[self.feature_cols]

    def predict(self, context, model_input: pd.DataFrame) -> pd.DataFrame:
        self._ensure_loaded(context)
        validated_input = self._validate_input(model_input)
        processed_input = self.preprocess.transform(validated_input)
        
        # Kiểm tra xem đối tượng model có predict_proba không
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(processed_input)[:, 1]
        else:
            probs = self.model.predict(processed_input).astype(float)
            
        return pd.DataFrame({"churn_probability": probs})