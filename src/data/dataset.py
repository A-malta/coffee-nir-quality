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


def load_raw_spectra(path):
    spectra = pd.read_excel(path, sheet_name=RAW_SPECTRA_SHEET)
    return spectra.iloc[:, 0], spectra.iloc[:, 1:]


def load_quality_table(path):
    return pd.read_excel(path, sheet_name=QUALITY_SHEET)


def aligned_quality_scores(quality, sample_ids):
    return aligned_column(quality, sample_ids, SCORE_TARGET_COLUMN)


def aligned_column(data, sample_ids, column):
    sample_col = data.columns[0]
    return data.set_index(sample_col)[column].reindex(sample_ids)


def load_processed_dataset(split, preprocess):
    X_raw = pd.read_excel(PROCESSED_DIR / split / f"{preprocess}.xlsx")
    y_raw = pd.read_excel(RAW_SPLIT_DIR / f"{split}_quality.xlsx")

    X = X_raw.iloc[:, 1:].T.astype(np.float32, copy=False)
    y = y_raw[CLASS_TARGET_COLUMN]

    return X.reset_index(drop=True), y.reset_index(drop=True)


def load_raw_split_dataset(split):
    _, spectra = load_split_spectra(RAW_SPLIT_DIR / f"{split}_spectra.xlsx")
    y_raw = pd.read_excel(RAW_SPLIT_DIR / f"{split}_quality.xlsx")

    X = spectra.T.astype(np.float32, copy=False)
    y = y_raw[CLASS_TARGET_COLUMN]

    return X.reset_index(drop=True), y.reset_index(drop=True)


def load_modeling_dataset(split, preprocess):
    if preprocess == RAW_PREPROCESS_NAME:
        return load_raw_split_dataset(split)
    return load_processed_dataset(split, preprocess)


def load_split_spectra(path):
    spectra = pd.read_excel(path)
    return spectra.iloc[:, 0].to_numpy(), spectra.iloc[:, 1:]
