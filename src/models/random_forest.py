from sklearn.ensemble import RandomForestRegressor
from typing import Dict, Any, Optional
import joblib
import os

class RandomForestModel:
    def __init__(self, params: Optional[Dict[str, Any]] = None, random_state: int = 42):
        """Cria um classificador Random Forest com hiperparâmetros opcionais.

        Args:
            params: Hiperparâmetros do ``RandomForestClassifier``.
            random_state: Semente para reprodutibilidade.
        """
        self.params = params or {}
        self.random_state = random_state
        self.model = RandomForestRegressor(**self.params, random_state=self.random_state)

    def fit(self, X, y):
        """Treina o modelo com atributos ``X`` e rótulos ``y``.

        Args:
            X: Matriz/DataFrame de features.
            y: Vetor/Série de classes alvo.

        Returns:
            None.
        """
        self.model.fit(X, y)

    def predict(self, X):
        """Realiza predição de classes para ``X``.

        Args:
            X: Matriz/DataFrame de features.

        Returns:
            Vetor de classes preditas.
        """
        return self.model.predict(X)

    def save(self, filepath: str):
        """Persiste o modelo em disco via joblib.

        Args:
            filepath: Caminho onde o modelo será salvo.

        Returns:
            None.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)

    @staticmethod
    def load(filepath: str) -> 'RandomForestModel':
        """Carrega um modelo salvo e retorna uma instância configurada.

        Args:
            filepath: Caminho do arquivo do modelo salvo.

        Returns:
            Instância de ``RandomForestModel`` com o modelo carregado.

        Raises:
            FileNotFoundError: Se o arquivo informado não existir.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Modelo não encontrado: {filepath}")
        
        loaded_model = joblib.load(filepath)
        instance = RandomForestModel()
        instance.model = loaded_model
        return instance
