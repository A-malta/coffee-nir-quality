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
    WAVELENGTH_COLUMN,
)


def normalize_wavelength_axis(spectra):
    spectra = spectra.copy()
    axis_column = spectra.columns[0]
    wavelengths = pd.to_numeric(spectra[axis_column])

    if wavelengths.max() > 3000:
        wavelengths = 10_000_000 / wavelengths

    spectra[axis_column] = wavelengths
    spectra = spectra.rename(columns={axis_column: WAVELENGTH_COLUMN})
    return spectra.sort_values(WAVELENGTH_COLUMN, kind="stable").reset_index(drop=True)


def load_raw_spectra(path):
    spectra = pd.read_excel(path, sheet_name=RAW_SPECTRA_SHEET)
    spectra = normalize_wavelength_axis(spectra)
    return spectra.iloc[:, 0], spectra.iloc[:, 1:]


def load_quality_table(path):
    quality = pd.read_excel(path, sheet_name=QUALITY_SHEET)
    if CLASS_TARGET_COLUMN in quality:
        quality[CLASS_TARGET_COLUMN] = quality[CLASS_TARGET_COLUMN].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
    return quality


def aligned_quality_scores(quality, sample_ids):
    return aligned_column(quality, sample_ids, SCORE_TARGET_COLUMN)


def aligned_quality_classes(quality, sample_ids):
    return aligned_column(quality, sample_ids, CLASS_TARGET_COLUMN)


def aligned_column(data, sample_ids, column):
    sample_col = data.columns[0]
    return data.set_index(sample_col)[column].reindex(sample_ids)


def load_processed_dataset(split, preprocess):
    X_raw = pd.read_excel(PROCESSED_DIR / split / f"{preprocess}.xlsx")
    y_raw = pd.read_excel(RAW_SPLIT_DIR / f"{split}_quality.xlsx")

    wavelengths = pd.to_numeric(X_raw.iloc[:, 0]).to_numpy()
    X = X_raw.iloc[:, 1:].T.astype(np.float32, copy=False)
    X.columns = wavelengths
    y = y_raw[CLASS_TARGET_COLUMN]

    return X.reset_index(drop=True), y.reset_index(drop=True)


def load_raw_split_dataset(split):
    wavelengths, spectra = load_split_spectra(RAW_SPLIT_DIR / f"{split}_spectra.xlsx")
    y_raw = pd.read_excel(RAW_SPLIT_DIR / f"{split}_quality.xlsx")

    X = spectra.T.astype(np.float32, copy=False)
    X.columns = wavelengths
    y = y_raw[CLASS_TARGET_COLUMN]

    return X.reset_index(drop=True), y.reset_index(drop=True)


def load_modeling_dataset(split, preprocess):
    if preprocess == RAW_PREPROCESS_NAME:
        return load_raw_split_dataset(split)
    return load_processed_dataset(split, preprocess)


def load_split_spectra(path):
    spectra = pd.read_excel(path)
    spectra = normalize_wavelength_axis(spectra)
    return spectra.iloc[:, 0].to_numpy(), spectra.iloc[:, 1:]
