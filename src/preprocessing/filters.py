import pandas as pd
import numpy as np
from typing import Optional, Union

try:
    from scipy.signal import savgol_filter
except ImportError:
    savgol_filter = None

def check_scipy():
    """Valida se ``scipy`` está disponível para filtros de Savitzky-Golay."""
    if savgol_filter is None:
        raise ImportError("A biblioteca 'scipy' é necessária para esta função. Instale-a com 'pip install scipy'.")

def snv(X: pd.DataFrame) -> pd.DataFrame:
    """Aplica normalização SNV em cada espectro.

    Args:
        X: DataFrame com espectros nas colunas.

    Returns:
        DataFrame normalizado por média e desvio-padrão de cada coluna.
    """

    mu = X.mean(axis=0)
    sigma = X.std(axis=0, ddof=1).replace(0.0, np.nan)
    return ((X - mu) / sigma).fillna(0.0)

def mean_centering(X: pd.DataFrame, mean_vector: Optional[pd.Series] = None) -> pd.DataFrame:
    """Centraliza cada linha subtraindo um vetor de média.

    Args:
        X: DataFrame com espectros.
        mean_vector: Vetor de médias por linha; se ``None``, é calculado de ``X``.

    Returns:
        DataFrame centrado na média.
    """

    if mean_vector is None:
        mu = X.mean(axis=1)
    else:
        mu = mean_vector
    return X.sub(mu, axis=0)

def savitzky_golay(X: pd.DataFrame, window_length: int, polyorder: int) -> pd.DataFrame:
    """Aplica suavização Savitzky-Golay em cada coluna.

    Args:
        X: DataFrame com espectros.
        window_length: Tamanho da janela do filtro.
        polyorder: Ordem do polinômio local.

    Returns:
        DataFrame suavizado.

    Raises:
        ImportError: Se ``scipy`` não estiver disponível.
    """
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
    """Calcula a 1ª ou 2ª derivada Savitzky-Golay dos espectros.

    Args:
        X: DataFrame com espectros.
        window_length: Tamanho da janela do filtro.
        polyorder: Ordem do polinômio local.
        deriv: Ordem da derivada (1 ou 2).
        delta: Espaçamento entre pontos; se ``None``, é inferido de ``wavelengths``.
        wavelengths: Vetor de comprimentos de onda para inferir ``delta``.

    Returns:
        DataFrame com derivadas calculadas.

    Raises:
        ImportError: Se ``scipy`` não estiver disponível.
        ValueError: Se ``deriv`` for inválido ou se ``delta`` e ``wavelengths`` faltarem.
    """
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
    """Aplica média móvel ao longo das colunas espectrais.

    Args:
        X: DataFrame com espectros.
        window_length: Tamanho da janela da média móvel.

    Returns:
        DataFrame suavizado por média móvel.
    """
    return X.rolling(window=window_length, axis=1, center=True).mean().fillna(method='bfill', axis=1).fillna(method='ffill', axis=1)

def sg_smoothing(X: pd.DataFrame, window_length: int, polyorder: int = 0) -> pd.DataFrame:
    """Alias para suavização por Savitzky-Golay.

    Args:
        X: DataFrame com espectros.
        window_length: Tamanho da janela do filtro.
        polyorder: Ordem do polinômio local.

    Returns:
        DataFrame suavizado.
    """

    return savitzky_golay(X, window_length, polyorder)
