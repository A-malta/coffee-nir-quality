from typing import Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    """Seleciona atributos por coeficientes de uma regressão logística L1."""

    def __init__(
        self,
        C: float,
        threshold: float,
        max_iter: int,
        tol: float,
        standardize: bool = True,
        penalty: str = "l1",
        solver: str = "saga",
    ) -> None:
        """Inicializa o seletor de atributos.

        Args:
            C: Inverso da intensidade de regularização da regressão logística.
            threshold: Limiar absoluto mínimo para selecionar um coeficiente.
            max_iter: Número máximo de iterações do otimizador.
            tol: Tolerância usada pelo critério de parada.
            standardize: Indica se os atributos devem ser padronizados no ajuste.
            penalty: Penalização da regressão, limitada a ``l1``.
            solver: Algoritmo de otimização da regressão logística.
        """
        self.C = C
        self.threshold = threshold
        self.max_iter = max_iter
        self.tol = tol
        self.standardize = standardize
        self.penalty = penalty
        self.solver = solver

    def fit(self, X: ArrayLike, y: ArrayLike) -> Self:
        """Ajusta o modelo L1 e determina os atributos selecionados.

        Args:
            X: Matriz de atributos com uma amostra por linha.
            y: Rótulos correspondentes às amostras.

        Returns:
            A própria instância ajustada do seletor.

        Raises:
            ValueError: Se a penalização configurada for diferente de ``l1``.
        """
        if self.penalty != "l1":
            raise ValueError("O seletor do TCC aceita somente penalização L1.")

        X_array = np.asarray(X, dtype=np.float32)
        if self.standardize:
            self.scaler_ = StandardScaler()
            X_selection = self.scaler_.fit_transform(X_array)
        else:
            self.scaler_ = None
            X_selection = X_array

        self.estimator_ = LogisticRegression(
            C=self.C,
            l1_ratio=1.0,
            solver=self.solver,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self.estimator_.fit(X_selection, y)

        coefficients = np.abs(self.estimator_.coef_)
        importances = coefficients.max(axis=0) if coefficients.ndim == 2 else coefficients
        support = importances > self.threshold
        if not support.any():
            support[int(np.argmax(importances))] = True

        self.support_ = support
        self.n_features_in_ = X_array.shape[1]
        self.n_selected_features_ = int(support.sum())
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.float32]:
        """Mantém somente os atributos selecionados durante o ajuste.

        Args:
            X: Matriz de atributos que será transformada.

        Returns:
            Array em ponto flutuante contendo os atributos selecionados.

        Raises:
            NotFittedError: Se o seletor ainda não tiver sido ajustado.
        """
        check_is_fitted(self, "support_")
        X_array = np.asarray(X, dtype=np.float32)
        return X_array[:, self.support_]
