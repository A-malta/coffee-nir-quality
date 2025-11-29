from sklearn.ensemble import RandomForestClassifier
from typing import Dict, Any, Optional
import joblib
import os

class RandomForestModel:
    def __init__(self, params: Optional[Dict[str, Any]] = None, random_state: int = 42):
        self.params = params or {}
        self.random_state = random_state
        self.model = RandomForestClassifier(**self.params, random_state=self.random_state)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def save(self, filepath: str):
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    @staticmethod
    def load(filepath: str) -> 'RandomForestModel':
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Modelo não encontrado: {filepath}")
        
        loaded_model = joblib.load(filepath)
        instance = RandomForestModel()
        instance.model = loaded_model
        return instance
