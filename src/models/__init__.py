from .base              import BaseSklearnWrapper
from .logistic_model    import LogisticModel, LogisticModelWrapper
from .knn_model         import KNNModel, KNNModelWrapper
from .lgbm_model        import LGBMModel, LGBMModelWrapper
from .poly_model        import PolyModel, PolyModelWrapper
from .rf_model          import RandomForestModel, RFModelWrapper

# Dictionary để gọi Wrapper nhanh trong train.py
WRAPPER_MAP = {
    "base": BaseSklearnWrapper,
    "logistic": LogisticModelWrapper,
    "knn": KNNModelWrapper,
    "lgbm": LGBMModelWrapper,
    "poly": PolyModelWrapper,
    "random_forest": RFModelWrapper,
}