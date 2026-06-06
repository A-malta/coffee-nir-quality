from pathlib import Path


DATA_DIR = Path("data")
RAW_SPLIT_DIR = DATA_DIR / "raw_split"
PROCESSED_DIR = DATA_DIR / "processed"

MODELS_DIR = Path("models")
PLOTS_DIR = Path("plots")
CONFUSION_MATRICES_DIR = Path("confusion_matrices")

RAW_SPECTRA_SHEET = "RawSpectra_RoastedCoffee"
QUALITY_SHEET = "Cup quality_RoastedCoffee"
CLASS_TARGET_COLUMN = "Class"
SCORE_TARGET_COLUMN = "Cup quality (points)"

RAW_PREPROCESS_NAME = "Raw"
BAYESIAN_SEARCH_RESULTS_FILE = Path("resultados_bayesian_search_treinamento.csv")
VALIDATION_RESULTS_FILE = Path("resultados_validacao_final.csv")
