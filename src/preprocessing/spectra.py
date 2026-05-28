from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def mean_centering(X: pd.DataFrame) -> pd.DataFrame:
    """Aplica mean centering aos espectros.

    Cada linha representa um comprimento de onda e cada coluna representa uma
    amostra. A funcao calcula, para cada comprimento de onda, a media entre as
    amostras do proprio conjunto e subtrai esse vetor de medias dos espectros
    correspondentes.

    Args:
        X: DataFrame com espectros.

    Returns:
        DataFrame centrado na média.
    """
    mu = X.mean(axis=1)
    return X.sub(mu, axis=0)


def savitzky_golay(
    X: pd.DataFrame,
    window_length: int,
    polyorder: int,
) -> pd.DataFrame:
    """Aplica suavização Savitzky-Golay aos espectros.

    Args:
        X: DataFrame com espectros em colunas.
        window_length: Tamanho da janela do filtro.
        polyorder: Ordem do polinômio local.

    Returns:
        DataFrame com os espectros suavizados.
    """
    return pd.DataFrame({
        col: savgol_filter(X[col].to_numpy(), window_length, polyorder)
        for col in X.columns
    }, index=X.index)


def spectral_derivative(X: pd.DataFrame, wavelengths: np.ndarray) -> pd.DataFrame:
    """Calcula a derivada numérica dos espectros.

    Args:
        X: DataFrame com espectros em colunas.
        wavelengths: Vetor de comprimentos de onda.

    Returns:
        DataFrame com a derivada calculada para cada espectro.
    """
    wl_array = np.asarray(wavelengths, dtype=float)
    return pd.DataFrame({
        col: np.gradient(X[col].to_numpy(), wl_array)
        for col in X.columns
    }, index=X.index)


@dataclass(frozen=True)
class PreprocessingVariant:
    name: str
    file_name: str


PREPROCESSING_VARIANTS = (
    PreprocessingVariant(
        name="SG_1D+2D+MeanCentering",
        file_name="SG_1D+2D+MeanCentering.xlsx",
    ),
)


def preprocess_spectra(
    X: pd.DataFrame,
    wavelengths: np.ndarray,
) -> pd.DataFrame:
    """Aplica Savitzky-Golay, 1ª derivada e 2ª derivada aos espectros."""
    X = savitzky_golay(X, window_length=15, polyorder=2)
    X = spectral_derivative(X, wavelengths)
    return spectral_derivative(X, wavelengths)
