import numpy as np
import pandas as pd

from src.config import CLASS_TARGET_COLUMN, RAW_SPLIT_DIR, WAVELENGTH_COLUMN
from src.data.dataset import aligned_column, load_quality_table, load_raw_spectra
from .kennard_stone import kennard_stone


def save_spectra(wavelengths, spectra, idx, name):
    wavelengths = pd.Series(wavelengths, name=WAVELENGTH_COLUMN)
    out = pd.concat([wavelengths, spectra[[spectra.columns[i] for i in idx]]], axis=1)
    out.to_excel(RAW_SPLIT_DIR / name, index=False)


def save_quality(quality, sample_ids, name):
    sample_col = quality.columns[0]
    out = pd.DataFrame({sample_col: sample_ids}).merge(quality, on=sample_col, how="left")
    out.to_excel(RAW_SPLIT_DIR / name, index=False)


def run_split(spectra_file, quality_file, validation_ratio=0.2):
    wavelengths, spectra = load_raw_spectra(spectra_file)
    quality = load_quality_table(quality_file)
    sample_ids = spectra.columns.tolist()
    labels = aligned_column(quality, sample_ids, CLASS_TARGET_COLUMN)
    features = np.asarray(spectra.T)
    labels = np.asarray(labels)

    val_idx = []
    for class_label in np.unique(labels):
        class_indices = np.flatnonzero(labels == class_label)
        n_validation = round(len(class_indices) * validation_ratio)
        # Aplica Kennard-Stone apenas nas amostras desta classe para manter
        # representatividade espectral dentro de cada grupo.
        selected_positions = kennard_stone(features[class_indices], n_validation)
        val_idx.extend(class_indices[selected_positions].tolist())

    val_idx = sorted(val_idx)
    validation_set = set(val_idx)
    train_idx = [idx for idx in range(len(features)) if idx not in validation_set]

    RAW_SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    save_spectra(wavelengths, spectra, train_idx, "training_spectra.xlsx")
    save_spectra(wavelengths, spectra, val_idx, "validation_spectra.xlsx")
    save_quality(quality, [sample_ids[i] for i in train_idx], "training_quality.xlsx")
    save_quality(quality, [sample_ids[i] for i in val_idx], "validation_quality.xlsx")
