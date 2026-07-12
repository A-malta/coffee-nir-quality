import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


class LassoFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(
        self,
        C,
        threshold,
        max_iter,
        tol,
    ):
        self.C = C
        self.threshold = threshold
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y):
        X_array = np.asarray(X, dtype=np.float32)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_array)

        self.estimator_ = LogisticRegression(
            C=self.C,
            l1_ratio=1.0,
            solver="saga",
            max_iter=self.max_iter,
            tol=self.tol,
        )
        self.estimator_.fit(X_scaled, y)

        coefficients = np.abs(self.estimator_.coef_)
        importances = coefficients.max(axis=0) if coefficients.ndim == 2 else coefficients
        support = importances > self.threshold
        if not support.any():
            support[int(np.argmax(importances))] = True

        self.support_ = support
        self.n_features_in_ = X_array.shape[1]
        self.n_selected_features_ = int(support.sum())
        return self

    def transform(self, X):
        check_is_fitted(self, "support_")
        X_array = np.asarray(X, dtype=np.float32)
        return X_array[:, self.support_]
