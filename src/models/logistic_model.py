from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.models.base import BaseModel, BaseSklearnWrapper

# 1. Class mô hình chính
class LogisticModel(BaseModel):
    def __init__(self, **kwargs):
        # Thiết lập các tham số mặc định
        params = {"max_iter": 1000, "random_state": 42}
        params.update(kwargs)
        
        # Xây dựng Pipeline xử lý đặc trưng cho Logistic Regression
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(**params)
        )

# 2. MLflow Wrapper cho Logistic Regression
class LogisticModelWrapper(BaseSklearnWrapper):
    pass