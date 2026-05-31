from lightgbm import LGBMClassifier
from src.models.base import BaseModel, BaseSklearnWrapper

class LGBMModel(BaseModel):
    def __init__(self, **kwargs):
        # Các tham số tối ưu cho LightGBM
        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "random_state": 42,
            "verbose": -1,  # Giảm log thừa khi fit
            "n_jobs": 1     # Giữ n_jobs=1 để tránh xung đột tài nguyên
        }
        params.update(kwargs)
        
        # LightGBM không nhất thiết phải cần StandardScaler (do là tree-based model),
        # nên ở đây ta gọi trực tiếp Classifier.
        self.model = LGBMClassifier(**params)

class LGBMModelWrapper(BaseSklearnWrapper):
    """
    Kế thừa hoàn toàn logic từ BaseSklearnWrapper.
    """
    pass