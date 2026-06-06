import pandas as pd
from scipy.signal import savgol_filter


def mean_centering(X):
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


def savitzky_golay(X, window_length, polyorder, deriv):
    """Aplica Savitzky-Golay aos espectros.

    Args:
        X: DataFrame com espectros em colunas.
        window_length: Tamanho da janela do filtro.
        polyorder: Ordem do polinômio local.
        deriv: Ordem da derivada.
    Returns:
        DataFrame com os espectros filtrados.
    """
    return pd.DataFrame({
        col: savgol_filter(X[col].to_numpy(), window_length, polyorder, deriv=deriv)
        for col in X.columns
    }, index=X.index)


class PreprocessingVariant:
    def __init__(self, name, file_name):
        self.name = name
        self.file_name = file_name


PREPROCESSING_VARIANTS = (
    PreprocessingVariant(
        name="SG_1D+MeanCentering",
        file_name="SG_1D+MeanCentering.xlsx",
    ),
)

PREPROCESS_NAME = PREPROCESSING_VARIANTS[0].name
PREPROCESS_FILE = PREPROCESSING_VARIANTS[0].file_name


def preprocess_spectra(
    X,
):
    """Aplica Savitzky-Golay com 1ª derivada aos espectros."""
    return savitzky_golay(
        X,
        window_length=15,
        polyorder=2,
        deriv=1,
    )
