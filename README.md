# Coffee NIR Quality Analysis

This project implements a machine learning pipeline for analyzing coffee quality using Near-Infrared (NIR) spectroscopy data. It includes modules for data splitting, preprocessing (various filters and normalizations), and model training/evaluation using Random Forest.

## Project Structure

- `src/`: Source code for data handling, preprocessing, models, and evaluation.
  - `data/`: Data loading and splitting logic (Kennard-Stone algorithm).
  - `preprocessing/`: Signal processing filters (Savitzky-Golay, SNV, MSC, etc.).
  - `models/`: Machine learning model wrappers (Random Forest).
  - `evaluation/`: Metric calculation utilities.
- `scripts/`: Executable scripts for running the pipeline.
  - `run_preprocessing.py`: Applies preprocessing chains to raw data.
  - `run_grid_search.py`: Performs grid search for hyperparameter tuning on processed data.
- `data/`: Directory for storing raw and processed datasets.

## Features

### Preprocessing
The pipeline supports a flexible preprocessing system with two groups of methods:
1.  **Row-wise / Independent**: Applied to each sample independently (e.g., Smoothing, Derivatives, SNV).
2.  **Column-wise / Dependent**: Depends on training set statistics (e.g., Mean Centering, Autoscaling).

Implemented filters:
- Savitzky-Golay Smoothing & Derivatives (1st and 2nd)
- Moving Average
- Standard Normal Variate (SNV)
- Multiplicative Scatter Correction (MSC)
- Mean Centering, Variance Scaling, Autoscaling

### Data Splitting
- Uses **Kennard-Stone** algorithm to ensure representative training, test, and validation sets.
- Split ratio: 80% Training, 10% Test, 10% Validation.

### Model Training
- **Random Forest** regressor/classifier (depending on target).
- Grid Search for hyperparameter optimization.
- Comprehensive metrics: Accuracy, Precision, Recall, Specificity (for classification).

## Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocessing
Run the preprocessing script to generate various processed datasets:
```bash
python scripts/run_preprocessing.py
```
This will create processed files in `data/processed/`.

### 3. Grid Search
Run the grid search to train and evaluate models on the processed data:
```bash
python scripts/run_grid_search.py
```
Results will be saved to `resultados_grid_search_validacao.csv` and `resultados_grid_search_teste.csv`.

## License
[MIT](LICENSE)
