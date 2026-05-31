from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from src.models.base import BaseModel, BaseSklearnWrapper

class KNNModel(BaseModel):
    def __init__(self, **kwargs):
        # Thiết lập các tham số mặc định cho KNN
        params = {
            "n_neighbors": 5,
            "n_jobs": -1
        }
        params.update(kwargs)
        
        # Pipeline: Chuẩn hóa dữ liệu trước khi đưa vào KNN
        self.model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(**params)
        )

class KNNModelWrapper(BaseSklearnWrapper):
    """
    Kế thừa trực tiếp logic load và predict từ BaseSklearnWrapper.
    """
    pass