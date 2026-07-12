from collections.abc import Hashable, Sequence
from os import PathLike
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.config import (
    CLASS_TARGET_COLUMN,
    PROCESSED_DIR,
    QUALITY_SHEET,
    RAW_PREPROCESS_NAME,
    RAW_SPECTRA_SHEET,
    RAW_SPLIT_DIR,
    SCORE_TARGET_COLUMN,
    WAVELENGTH_COLUMN,
)


def normalize_wavelength_axis(spectra: pd.DataFrame) -> pd.DataFrame:
    """Normaliza e ordena o eixo de comprimentos de onda dos espectros.

    Converte o eixo espectral para valores numéricos e, quando ele está em
    número de onda, transforma-o em comprimento de onda antes da ordenação.

    Args:
        spectra: Tabela cuja primeira coluna contém o eixo espectral.

    Returns:
        Cópia da tabela com o eixo nomeado e ordenado por comprimento de onda.
    """
    spectra = spectra.copy()
    axis_column = spectra.columns[0]
    wavelengths = pd.to_numeric(spectra[axis_column])

    if wavelengths.max() > 3000:
        wavelengths = 10_000_000 / wavelengths

    spectra[axis_column] = wavelengths
    spectra = spectra.rename(columns={axis_column: WAVELENGTH_COLUMN})
    return spectra.sort_values(WAVELENGTH_COLUMN, kind="stable").reset_index(drop=True)


def load_raw_spectra(
    path: str | PathLike[str],
) -> tuple[pd.Series, pd.DataFrame]:
    """Carrega a planilha original de espectros.

    Args:
        path: Caminho da pasta de trabalho que contém os espectros brutos.

    Returns:
        Tupla com a série de comprimentos de onda e a tabela de espectros.
    """
    spectra = pd.read_excel(path, sheet_name=RAW_SPECTRA_SHEET)
    spectra = normalize_wavelength_axis(spectra)
    return spectra.iloc[:, 0], spectra.iloc[:, 1:]


def load_quality_table(path: str | PathLike[str]) -> pd.DataFrame:
    """Carrega e normaliza a tabela de qualidade sensorial.

    Args:
        path: Caminho da pasta de trabalho com os dados de qualidade.

    Returns:
        Tabela de qualidade com espaços externos removidos dos rótulos.
    """
    quality = pd.read_excel(path, sheet_name=QUALITY_SHEET)
    if CLASS_TARGET_COLUMN in quality:
        quality[CLASS_TARGET_COLUMN] = quality[CLASS_TARGET_COLUMN].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    return quality


def aligned_quality_scores(
    quality: pd.DataFrame,
    sample_ids: Sequence[Hashable],
) -> pd.Series:
    """Alinha os escores sensoriais à ordem informada de amostras.

    Args:
        quality: Tabela com identificadores e atributos de qualidade.
        sample_ids: Identificadores na ordem desejada para o resultado.

    Returns:
        Série de escores reindexada pela ordem das amostras.
    """
    return aligned_column(quality, sample_ids, SCORE_TARGET_COLUMN)


def aligned_quality_classes(
    quality: pd.DataFrame,
    sample_ids: Sequence[Hashable],
) -> pd.Series:
    """Alinha as classes de qualidade à ordem informada de amostras.

    Args:
        quality: Tabela com identificadores e atributos de qualidade.
        sample_ids: Identificadores na ordem desejada para o resultado.

    Returns:
        Série de classes reindexada pela ordem das amostras.
    """
    return aligned_column(quality, sample_ids, CLASS_TARGET_COLUMN)


def aligned_column(
    data: pd.DataFrame,
    sample_ids: Sequence[Hashable],
    column: Hashable,
) -> pd.Series:
    """Alinha uma coluna da tabela à sequência de identificadores.

    Args:
        data: Tabela cuja primeira coluna identifica as amostras.
        sample_ids: Identificadores na ordem desejada para o resultado.
        column: Nome da coluna que será extraída.

    Returns:
        Série selecionada e reindexada pelos identificadores fornecidos.
    """
    sample_col = data.columns[0]
    return data.set_index(sample_col)[column].reindex(sample_ids)


def load_processed_dataset(
    split: str,
    preprocess: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Carrega uma partição espectral previamente processada.

    Args:
        split: Nome da partição de dados, como ``training`` ou ``validation``.
        preprocess: Nome do pré-processamento armazenado em disco.

    Returns:
        Tupla com a matriz de atributos por amostra e os rótulos de classe.
    """
    X_raw = pd.read_excel(PROCESSED_DIR / split / f"{preprocess}.xlsx")
    y_raw = pd.read_excel(RAW_SPLIT_DIR / f"{split}_quality.xlsx")

    wavelengths = pd.to_numeric(X_raw.iloc[:, 0]).to_numpy()
    X = X_raw.iloc[:, 1:].T.astype(np.float32, copy=False)
    X.columns = wavelengths
    y = y_raw[CLASS_TARGET_COLUMN]

    return X.reset_index(drop=True), y.reset_index(drop=True)


def load_raw_split_dataset(split: str) -> tuple[pd.DataFrame, pd.Series]:
    """Carrega uma partição sem pré-processamento espectral.

    Args:
        split: Nome da partição de dados, como ``training`` ou ``validation``.

    Returns:
        Tupla com a matriz de atributos por amostra e os rótulos de classe.
    """
    wavelengths, spectra = load_split_spectra(RAW_SPLIT_DIR / f"{split}_spectra.xlsx")
    y_raw = pd.read_excel(RAW_SPLIT_DIR / f"{split}_quality.xlsx")

    X = spectra.T.astype(np.float32, copy=False)
    X.columns = wavelengths
    y = y_raw[CLASS_TARGET_COLUMN]

    return X.reset_index(drop=True), y.reset_index(drop=True)


def load_modeling_dataset(
    split: str,
    preprocess: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Carrega a variante de dados solicitada para modelagem.

    Args:
        split: Nome da partição de dados.
        preprocess: Nome do pré-processamento ou marcador para dados brutos.

    Returns:
        Tupla com a matriz de atributos por amostra e os rótulos de classe.
    """
    if preprocess == RAW_PREPROCESS_NAME:
        return load_raw_split_dataset(split)
    return load_processed_dataset(split, preprocess)


def load_split_spectra(
    path: str | PathLike[str],
) -> tuple[NDArray[Any], pd.DataFrame]:
    """Carrega e normaliza os espectros de uma partição salva.

    Args:
        path: Caminho da planilha com o eixo e os espectros da partição.

    Returns:
        Tupla com o eixo espectral como array e a tabela de espectros.
    """
    spectra = pd.read_excel(path)
    spectra = normalize_wavelength_axis(spectra)
    return spectra.iloc[:, 0].to_numpy(), spectra.iloc[:, 1:]
