from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from src.models.base import BaseModel, BaseSklearnWrapper

# 1. Class mô hình chính
class PolyModel(BaseModel):
    def __init__(self, degree=2, **kwargs):
        # Thiết lập các tham số mặc định cho Logistic Regression
        params = {"max_iter": 1000, "random_state": 42}
        params.update(kwargs)
        
        # Xây dựng Pipeline: Scale -> Sinh đặc trưng đa thức -> Logistic
        self.model = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=degree, include_bias=False),
            LogisticRegression(**params)
        )

# 2. MLflow Wrapper
class PolyModelWrapper(BaseSklearnWrapper):
    pass