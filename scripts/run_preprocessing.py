from typing import Literal

import pandas as pd
from numpy.typing import ArrayLike

from src.config import PROCESSED_DIR, RAW_SPLIT_DIR, WAVELENGTH_COLUMN
from src.data.dataset import load_split_spectra
from src.preprocessing.spectra import PREPROCESS_FILE, mean_centering, preprocess_spectra


def save_processed_spectra(
    wavelengths: ArrayLike,
    X: pd.DataFrame,
    split: Literal["training", "validation"],
) -> None:
    """Salva os espectros pré-processados de uma partição.

    Args:
        wavelengths: Comprimentos de onda associados às linhas de ``X``.
        X: Espectros pré-processados organizados em colunas.
        split: Partição de destino, treinamento ou validação.
    """
    path = PROCESSED_DIR / split / PREPROCESS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([pd.Series(wavelengths, name=WAVELENGTH_COLUMN), X], axis=1).to_excel(path, index=False)


def preprocess_data(
    wavelengths: ArrayLike,
    X_train_raw: pd.DataFrame,
    X_val_raw: pd.DataFrame,
) -> None:
    """Pré-processa e salva os espectros de treino e validação.

    Args:
        wavelengths: Comprimentos de onda comuns aos dois conjuntos.
        X_train_raw: Espectros brutos de treinamento.
        X_val_raw: Espectros brutos de validação.
    """
    X_train = preprocess_spectra(X_train_raw.copy())
    X_val = preprocess_spectra(X_val_raw.copy())
    X_train = mean_centering(X_train)
    X_val = mean_centering(X_val)

    save_processed_spectra(wavelengths, X_train, "training")
    save_processed_spectra(wavelengths, X_val, "validation")


def run_preprocessing() -> None:
    """Executa o pré-processamento dos conjuntos espectrais divididos."""
    wavelengths, X_train_raw = load_split_spectra(RAW_SPLIT_DIR / "training_spectra.xlsx")
    _, X_val_raw = load_split_spectra(RAW_SPLIT_DIR / "validation_spectra.xlsx")

    preprocess_data(wavelengths, X_train_raw, X_val_raw)
