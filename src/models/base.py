import joblib, mlflow, pandas as pd, yaml

class BaseModel:
    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)

class BaseSklearnWrapper(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.preprocess     = joblib.load(context.artifacts["preprocess_path"])
        self.model          = joblib.load(context.artifacts["model_path"])
        
        # Load và parse file YAML
        with open(context.artifacts["data_schema"], "r") as f:
            self.schema = yaml.safe_load(f)
            
        # Tự động trích xuất tất cả các cột input cần thiết (không bao gồm target)
        self.feature_cols = self._extract_features(self.schema)

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

    def predict(self, context, model_input):
        validated_input = self._validate_input(model_input)
        processed_input = self.preprocess.transform(validated_input)
        return self.model.predict(processed_input)