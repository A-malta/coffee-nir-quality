from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from src.config import PLOTS_DIR, PROCESSED_DIR, RAW_SPLIT_DIR
from src.data.dataset import (
    aligned_quality_classes,
    aligned_quality_scores,
    load_quality_table,
    load_raw_spectra,
    load_split_spectra,
)
from src.preprocessing.spectra import PREPROCESS_FILE, PREPROCESS_NAME


CLASS_COLORS = {
    "muito-bom": "#1f77b4",
    "muito bom": "#1f77b4",
    "very good": "#1f77b4",
    "excelente": "#d62728",
    "excellent": "#d62728",
}
DEFAULT_CLASS_COLOR = "#7f7f7f"
FIGSIZE = (10, 6)
TITLE_FONT_SIZE = 10
LABEL_FONT_SIZE = 9
TICK_FONT_SIZE = 8
LEGEND_FONT_SIZE = 8


def normalized_class_name(label: object) -> str:
    """Normaliza um rótulo de classe para uso como chave de cor.

    Args:
        label: Rótulo de classe a ser normalizado.

    Returns:
        Rótulo sem espaços externos, em minúsculas e com sublinhados
        substituídos por hífens.
    """
    return str(label).strip().casefold().replace("_", "-")


def plot_spectra_by_score(
    wavelengths: ArrayLike,
    spectra: pd.DataFrame,
    scores: ArrayLike,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    """Gera um gráfico de espectros coloridos pela pontuação sensorial.

    Args:
        wavelengths: Comprimentos de onda usados no eixo horizontal.
        spectra: Espectros organizados em colunas, um por amostra.
        scores: Pontuações sensoriais alinhadas às colunas de ``spectra``.
        output_path: Caminho do arquivo de imagem a ser gerado.
        title: Título do gráfico.
        ylabel: Rótulo do eixo vertical.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scores = np.asarray(scores)
    cmap = plt.get_cmap("viridis")
    norm = plt.Normalize(scores.min(), scores.max())

    fig, ax = plt.subplots(figsize=FIGSIZE)
    for idx, column in enumerate(spectra.columns):
        ax.plot(
            wavelengths,
            spectra[column],
            color=cmap(norm(scores[idx])),
            alpha=0.7,
            linewidth=0.8,
        )

    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("Wavelength (nm)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    colorbar = fig.colorbar(sm, ax=ax, label="Nota de Café (Pontos)", pad=0.02)
    colorbar.ax.yaxis.label.set_size(LABEL_FONT_SIZE)
    colorbar.ax.tick_params(labelsize=TICK_FONT_SIZE)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_spectra_by_class(
    wavelengths: ArrayLike,
    spectra: pd.DataFrame,
    classes: ArrayLike,
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    """Gera um gráfico de espectros coloridos pela classe sensorial.

    Args:
        wavelengths: Comprimentos de onda usados no eixo horizontal.
        spectra: Espectros organizados em colunas, um por amostra.
        classes: Classes sensoriais alinhadas às colunas de ``spectra``.
        output_path: Caminho do arquivo de imagem a ser gerado.
        title: Título do gráfico.
        ylabel: Rótulo do eixo vertical.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    classes = pd.Series(classes, index=spectra.columns)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    labels_by_key = {}
    for column in spectra.columns:
        label = classes[column]
        class_key = normalized_class_name(label)
        labels_by_key.setdefault(class_key, str(label))
        ax.plot(
            wavelengths,
            spectra[column],
            color=CLASS_COLORS.get(class_key, DEFAULT_CLASS_COLOR),
            alpha=0.7,
            linewidth=0.8,
        )

    ax.set_title(title, fontsize=TITLE_FONT_SIZE)
    ax.set_xlabel("Wavelength (nm)", fontsize=LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_FONT_SIZE)

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=CLASS_COLORS.get(class_key, DEFAULT_CLASS_COLOR),
            linewidth=1.5,
            label=label,
        )
        for class_key, label in labels_by_key.items()
    ]
    ax.legend(
        handles=handles,
        title="Classe",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=len(handles),
        frameon=False,
        fontsize=LEGEND_FONT_SIZE,
        title_fontsize=LEGEND_FONT_SIZE,
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_raw_spectra(spectra_file: Path, quality_file: Path) -> None:
    """Gera as visualizações dos espectros NIR brutos.

    Args:
        spectra_file: Planilha com os espectros brutos.
        quality_file: Planilha com pontuações e classes sensoriais.
    """
    wavelengths, X_raw = load_raw_spectra(spectra_file)
    quality_df = load_quality_table(quality_file)
    scores = aligned_quality_scores(quality_df, X_raw.columns.tolist())
    classes = aligned_quality_classes(quality_df, X_raw.columns.tolist())
    plot_spectra_by_score(
        wavelengths=wavelengths,
        spectra=X_raw,
        scores=scores,
        output_path=PLOTS_DIR / "espectros_nir_brutos.png",
        title="Espectros NIR Brutos",
        ylabel="Absorbância",
    )
    plot_spectra_by_class(
        wavelengths=wavelengths,
        spectra=X_raw,
        classes=classes,
        output_path=PLOTS_DIR / "espectros_nir_brutos_por_classe.png",
        title="Espectros NIR Brutos por Classe",
        ylabel="Absorbância",
    )


def plot_processed_spectra() -> None:
    """Gera as visualizações dos espectros NIR pré-processados."""
    wavelengths, X_train = load_split_spectra(PROCESSED_DIR / "training" / PREPROCESS_FILE)
    _, X_val = load_split_spectra(PROCESSED_DIR / "validation" / PREPROCESS_FILE)

    quality_train = pd.read_excel(RAW_SPLIT_DIR / "training_quality.xlsx")
    quality_val = pd.read_excel(RAW_SPLIT_DIR / "validation_quality.xlsx")
    quality = pd.concat([quality_train, quality_val], ignore_index=True)

    spectra = pd.concat([X_train, X_val], axis=1)
    scores = aligned_quality_scores(quality, spectra.columns.tolist())
    classes = aligned_quality_classes(quality, spectra.columns.tolist())

    plot_spectra_by_score(
        wavelengths=wavelengths,
        spectra=spectra,
        scores=scores,
        output_path=PLOTS_DIR / f"espectros_nir_preprocessados_{PREPROCESS_NAME}.png",
        title=f"Espectros NIR Pré-processados - {PREPROCESS_NAME}",
        ylabel="Sinal Normalizado",
    )
    plot_spectra_by_class(
        wavelengths=wavelengths,
        spectra=spectra,
        classes=classes,
        output_path=PLOTS_DIR / f"espectros_nir_preprocessados_{PREPROCESS_NAME}_por_classe.png",
        title=f"Espectros NIR Pré-processados por Classe - {PREPROCESS_NAME}",
        ylabel="Sinal Normalizado",
    )


def run_plot(spectra_file: Path, quality_file: Path) -> None:
    """Executa a geração de todas as visualizações espectrais.

    Args:
        spectra_file: Planilha com os espectros brutos.
        quality_file: Planilha com pontuações e classes sensoriais.
    """
    plot_raw_spectra(spectra_file, quality_file)
    plot_processed_spectra()
