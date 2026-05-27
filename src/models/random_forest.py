import os
from typing import Any

import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder


BACKEND_ENV = "COFFEE_NIR_RF_BACKEND"


def has_cuda() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def select_random_forest():
    backend = os.getenv(BACKEND_ENV, "auto").lower()
    if backend not in {"auto", "gpu", "cpu"}:
        raise ValueError(f"{BACKEND_ENV} deve ser 'auto', 'gpu' ou 'cpu'.")

    if backend != "cpu" and has_cuda():
        try:
            from cuml.ensemble import RandomForestClassifier

            return RandomForestClassifier, "cuml"
        except ImportError:
            if backend == "gpu":
                raise RuntimeError("Backend GPU solicitado, mas cuML não está instalado.")

    if backend == "gpu":
        raise RuntimeError("Backend GPU solicitado, mas nenhuma GPU CUDA está acessível.")

    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier, "sklearn"


RandomForestClassifier, BACKEND = select_random_forest()


class RandomForestModel:
    def __init__(self, params: dict[str, Any] | None = None, random_state: int = 42):
        self.params = params or {}
        self.random_state = random_state
        self.encoder = None
        self.backend = BACKEND
        model_params = self.params.copy()

        if self.backend == "cuml":
            model_params.pop("class_weight", None)
        else:
            model_params.setdefault("n_jobs", -1)

        self.model = RandomForestClassifier(**model_params, random_state=random_state)

    @staticmethod
    def backend_name() -> str:
        return BACKEND

    def _features(self, X):
        X = X.to_numpy(dtype=np.float32, copy=False) if hasattr(X, "to_numpy") else np.asarray(X, dtype=np.float32)
        return np.asfortranarray(X) if self.backend == "cuml" else X

    @staticmethod
    def _numpy(values):
        if hasattr(values, "get"):
            return values.get()
        return values.to_numpy() if hasattr(values, "to_numpy") else np.asarray(values)

    def fit(self, X, y) -> None:
        y = np.asarray(y)
        if y.dtype.kind in {"O", "U", "S"}:
            self.encoder = LabelEncoder()
            y = self.encoder.fit_transform(y)
        if self.backend == "cuml":
            y = y.astype(np.int32)
        self.model.fit(self._features(X), y)

    def predict(self, X):
        y_pred = self._numpy(self.model.predict(self._features(X)))
        return self.encoder.inverse_transform(y_pred.astype(int)) if self.encoder is not None else y_pred

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self.__dict__, path)

    @staticmethod
    def load(path: str) -> "RandomForestModel":
        data = joblib.load(path)
        instance = object.__new__(RandomForestModel)
        if isinstance(data, dict):
            instance.__dict__.update(data)
        else:
            instance.model = data
            instance.encoder = None
            instance.params = {}
            instance.random_state = 42
            instance.backend = "cuml" if data.__class__.__module__.startswith("cuml") else "sklearn"
        return instance
