import pandas as pd

from src.config import PROCESSED_DIR, RAW_SPLIT_DIR
from src.data.dataset import load_split_spectra
from src.preprocessing.spectra import PREPROCESSING_VARIANTS, PreprocessingVariant, mean_centering, preprocess_spectra


def save_processed_spectra(wavelengths, X: pd.DataFrame, split: str, file_name: str) -> None:
    path = PROCESSED_DIR / split / file_name
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([pd.Series(wavelengths, name="Wavenumbers"), X], axis=1).to_excel(path, index=False)


def preprocess_variant(
    wavelengths,
    X_train_raw: pd.DataFrame,
    X_val_raw: pd.DataFrame,
    variant: PreprocessingVariant,
) -> None:
    X_train = preprocess_spectra(
        X_train_raw.copy(),
        wavelengths,
    )
    X_val = preprocess_spectra(
        X_val_raw.copy(),
        wavelengths,
    )
    X_train = mean_centering(X_train)
    X_val = mean_centering(X_val)

    save_processed_spectra(wavelengths, X_train, "training", variant.file_name)
    save_processed_spectra(wavelengths, X_val, "validation", variant.file_name)


def main() -> None:
    wavelengths, X_train_raw = load_split_spectra(RAW_SPLIT_DIR / "training_spectra.xlsx")
    _, X_val_raw = load_split_spectra(RAW_SPLIT_DIR / "validation_spectra.xlsx")

    for variant in PREPROCESSING_VARIANTS:
        preprocess_variant(wavelengths, X_train_raw, X_val_raw, variant)
