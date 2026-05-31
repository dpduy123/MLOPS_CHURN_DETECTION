from sklearn.ensemble import RandomForestClassifier
from src.models.base import BaseModel, BaseSklearnWrapper

# 1. Class mô hình kế thừa từ BaseModel
class RandomForestModel(BaseModel):
    def __init__(self, **kwargs):
        params = {
            "n_estimators": 100, 
            "max_depth": 5, 
            "random_state": 42, 
            "n_jobs": -1
        }
        params.update(kwargs)
        self.model = RandomForestClassifier(**params)

# 2. MLflow Wrapper
class RFModelWrapper(BaseSklearnWrapper):
    pass