from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CLASS_TARGET_COLUMN,
    PROCESSED_DIR,
    QUALITY_SHEET,
    RAW_PREPROCESS_NAME,
    RAW_SPECTRA_SHEET,
    RAW_SPLIT_DIR,
    SCORE_TARGET_COLUMN,
)


def load_raw_spectra(path: Path) -> tuple[pd.Series, pd.DataFrame]:
    spectra = pd.read_excel(path, sheet_name=RAW_SPECTRA_SHEET)
    return spectra.iloc[:, 0], spectra.iloc[:, 1:]


def load_quality_table(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=QUALITY_SHEET)


def aligned_quality_scores(quality: pd.DataFrame, sample_ids: list[str]) -> pd.Series:
    return aligned_column(quality, sample_ids, SCORE_TARGET_COLUMN)


def aligned_column(data: pd.DataFrame, sample_ids: list[str], column: str) -> pd.Series:
    sample_col = data.columns[0]
    return data.set_index(sample_col)[column].reindex(sample_ids)


def load_processed_dataset(split: str, preprocess: str) -> tuple[pd.DataFrame, pd.Series]:
    X_raw = pd.read_excel(PROCESSED_DIR / split / f"{preprocess}.xlsx")
    y_raw = pd.read_excel(RAW_SPLIT_DIR / f"{split}_quality.xlsx")

    X = X_raw.iloc[:, 1:].T.astype(np.float32, copy=False)
    y = y_raw[CLASS_TARGET_COLUMN]

    return X.reset_index(drop=True), y.reset_index(drop=True)


def load_raw_split_dataset(split: str) -> tuple[pd.DataFrame, pd.Series]:
    _, spectra = load_split_spectra(RAW_SPLIT_DIR / f"{split}_spectra.xlsx")
    y_raw = pd.read_excel(RAW_SPLIT_DIR / f"{split}_quality.xlsx")

    X = spectra.T.astype(np.float32, copy=False)
    y = y_raw[CLASS_TARGET_COLUMN]

    return X.reset_index(drop=True), y.reset_index(drop=True)


def load_modeling_dataset(split: str, preprocess: str) -> tuple[pd.DataFrame, pd.Series]:
    if preprocess == RAW_PREPROCESS_NAME:
        return load_raw_split_dataset(split)
    return load_processed_dataset(split, preprocess)


def load_split_spectra(path: Path) -> tuple[np.ndarray, pd.DataFrame]:
    spectra = pd.read_excel(path)
    return spectra.iloc[:, 0].to_numpy(), spectra.iloc[:, 1:]
