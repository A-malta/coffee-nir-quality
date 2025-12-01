import pandas as pd
import numpy as np
from typing import Optional, Union

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

def check_scipy():
    if savgol_filter is None:
        raise ImportError("A biblioteca 'scipy' é necessária para esta função. Instale-a com 'pip install scipy'.")

def snv(X: pd.DataFrame) -> pd.DataFrame:

    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1).replace(0.0, np.nan)
    return ((X - mu) / sigma).fillna(0.0)

def mean_centering(X: pd.DataFrame, mean_vector: Optional[pd.Series] = None) -> pd.DataFrame:

    if mean_vector is None:
        mu = X.mean(axis=1)
    else:
        mu = mean_vector
    return X.sub(mu, axis=0)

def savitzky_golay(X: pd.DataFrame, window_length: int, polyorder: int) -> pd.DataFrame:
    check_scipy()
    return pd.DataFrame({
        col: savgol_filter(X[col].to_numpy(), window_length, polyorder)
        for col in X.columns
    }, index=X.index)

def savitzky_derivative(
    X: pd.DataFrame,
    window_length: int,
    polyorder: int,
    deriv: int = 1,
    delta: Optional[float] = None,
    wavelengths: Optional[Union[np.ndarray, list]] = None,
) -> pd.DataFrame:
    check_scipy()
    if deriv not in (1, 2):
        raise ValueError("deriv deve ser 1 ou 2")
    
    if delta is None:
        if wavelengths is None:
            raise ValueError("Se delta não for fornecido, wavelengths devem ser informados.")
        wl_array = np.asarray(wavelengths, dtype=float)
        delta = float(np.mean(np.diff(wl_array)))
        
    return pd.DataFrame({
        col: savgol_filter(X[col].to_numpy(), window_length, polyorder, deriv=deriv, delta=delta)
        for col in X.columns
    }, index=X.index)

def moving_average(X: pd.DataFrame, window_length: int) -> pd.DataFrame:
    return X.rolling(window=window_length, axis=1, center=True).mean().fillna(method='bfill', axis=1).fillna(method='ffill', axis=1)

def sg_smoothing(X: pd.DataFrame, window_length: int, polyorder: int = 0) -> pd.DataFrame:

    return savitzky_golay(X, window_length, polyorder)
