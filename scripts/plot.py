import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PLOTS_DIR, PROCESSED_DIR, RAW_SPLIT_DIR
from src.data.dataset import aligned_quality_scores, load_quality_table, load_raw_spectra, load_split_spectra
from src.preprocessing.spectra import PREPROCESS_FILE, PREPROCESS_NAME


def plot_spectra_by_score(
    wavelengths,
    spectra,
    scores,
    output_path,
    title,
    ylabel,
):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scores = np.asarray(scores)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(scores.min(), scores.max())

    fig, ax = plt.subplots(figsize=(8, 5))
    for idx, column in enumerate(spectra.columns):
        ax.plot(
            wavelengths,
            spectra[column],
            color=cmap(norm(scores[idx])),
            alpha=0.7,
            linewidth=0.8,
        )

    ax.set_title(title)
    ax.set_xlabel("Wavenumber")
    ax.set_ylabel(ylabel)
    ax.invert_xaxis()

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, ax=ax, label="Nota de Café (Pontos)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_raw_spectra(spectra_file, quality_file):
    wavelengths, X_raw = load_raw_spectra(spectra_file)
    quality_df = load_quality_table(quality_file)
    scores = aligned_quality_scores(quality_df, X_raw.columns.tolist())
    plot_spectra_by_score(
        wavelengths=wavelengths,
        spectra=X_raw,
        scores=scores,
        output_path=PLOTS_DIR / "espectros_nir_brutos.png",
        title="Espectros NIR Brutos",
        ylabel="Absorbância",
    )


def plot_processed_spectra():
    wavelengths, X_train = load_split_spectra(PROCESSED_DIR / "training" / PREPROCESS_FILE)
    _, X_val = load_split_spectra(PROCESSED_DIR / "validation" / PREPROCESS_FILE)

    quality_train = pd.read_excel(RAW_SPLIT_DIR / "training_quality.xlsx")
    quality_val = pd.read_excel(RAW_SPLIT_DIR / "validation_quality.xlsx")
    quality = pd.concat([quality_train, quality_val], ignore_index=True)

    spectra = pd.concat([X_train, X_val], axis=1)
    scores = aligned_quality_scores(quality, spectra.columns.tolist())

    plot_spectra_by_score(
        wavelengths=wavelengths,
        spectra=spectra,
        scores=scores,
        output_path=PLOTS_DIR / f"espectros_nir_preprocessados_{PREPROCESS_NAME}.png",
        title=f"Espectros NIR Pré-processados - {PREPROCESS_NAME}",
        ylabel="Sinal Normalizado",
    )


def run_plot(spectra_file, quality_file):
    plot_raw_spectra(spectra_file, quality_file)
    plot_processed_spectra()
