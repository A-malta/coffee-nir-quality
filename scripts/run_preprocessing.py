import pandas as pd

from src.config import PROCESSED_DIR, RAW_SPLIT_DIR
from src.data.dataset import load_split_spectra
from src.preprocessing.spectra import PREPROCESS_FILE, mean_centering, preprocess_spectra


def save_processed_spectra(wavelengths, X, split):
    path = PROCESSED_DIR / split / PREPROCESS_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([pd.Series(wavelengths, name="Wavenumbers"), X], axis=1).to_excel(path, index=False)


def preprocess_data(wavelengths, X_train_raw, X_val_raw):
    X_train = preprocess_spectra(X_train_raw.copy())
    X_val = preprocess_spectra(X_val_raw.copy())
    X_train = mean_centering(X_train)
    X_val = mean_centering(X_val)

    save_processed_spectra(wavelengths, X_train, "training")
    save_processed_spectra(wavelengths, X_val, "validation")


def run_preprocessing():
    wavelengths, X_train_raw = load_split_spectra(RAW_SPLIT_DIR / "training_spectra.xlsx")
    _, X_val_raw = load_split_spectra(RAW_SPLIT_DIR / "validation_spectra.xlsx")

    preprocess_data(wavelengths, X_train_raw, X_val_raw)
