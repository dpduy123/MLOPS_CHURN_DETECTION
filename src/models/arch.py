from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
import joblib


class LogisticModel:
    def __init__(self, **kwargs):
        params = {"max_iter": 1000, "random_state": 42}
        params.update(kwargs)
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(**params)
        )

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


class PolyModel:
    def __init__(self, degree=2, **kwargs):
        params = {"max_iter": 1000, "random_state": 42}
        params.update(kwargs)
        self.model = make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=degree, include_bias=False),
            LogisticRegression(**params)
        )

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


class RandomForestModel:
    def __init__(self, **kwargs):
        params = {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42,
            "n_jobs": -1
        }
        params.update(kwargs)
        self.model = RandomForestClassifier(**params)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


class KNNModel:
    def __init__(self, **kwargs):
        params = {
            "n_neighbors": 5,
            "n_jobs": -1
        }
        params.update(kwargs)
        self.model = make_pipeline(
            StandardScaler(),
            KNeighborsClassifier(**params)
        )

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)


class LGBMModel:
    def __init__(self, **kwargs):
        params = {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 5,
            "random_state": 42,
            "verbose": -1,  # Giảm bớt hiển thị log thừa của LightGBM khi chạy tuning
            "n_jobs": 1  # Đổi từ mặc định -1 thành 1 để bảo vệ CPU khỏi bị quá tải
        }
        params.update(kwargs)
        self.model = LGBMClassifier(**params)

    def fit(self, x, y):
        self.model.fit(x, y)

    def predict(self, x):
        return self.model.predict(x)

    def save(self, path):
        joblib.dump(self.model, path)

    def load(self, path):
        self.model = joblib.load(path)