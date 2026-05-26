import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Union

def evaluate_model(y_true: Union[np.ndarray, pd.Series], y_pred: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
    """Retorna métricas principais de regressão em um dicionário.

    Args:
        y_true: Valores reais.
        y_pred: Valores preditos.

    Returns:
        Dicionário com r2, rmse e mae.
    """
    return {
        'r2': float(r2_score(y_true, y_pred)),
        'rmse': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': float(mean_absolute_error(y_true, y_pred)),
    }
