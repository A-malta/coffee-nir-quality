# Coffee NIR Quality Pipeline

Pipeline para classificar a qualidade sensorial de café torrado a partir de espectros NIR.

## Etapas

Ao executar `python main.py`, o projeto roda três etapas:

1. **Split treino/validação (`scripts/run_split.py`)**
   - Lê espectros e qualidade sensorial.
   - Alinha as classes da coluna `Class` às amostras espectrais.
   - Separa aproximadamente 80% para treino e 20% para validação.
   - Mantém a validação estratificada por classe.
   - Seleciona amostras representativas dentro de cada classe com Kennard-Stone.
   - Salva:
     - `data/raw_split/training_spectra.xlsx`
     - `data/raw_split/validation_spectra.xlsx`
     - `data/raw_split/training_quality.xlsx`
     - `data/raw_split/validation_quality.xlsx`

2. **Pré-processamento (`scripts/run_preprocessing.py`)**
   - Aplica Savitzky-Golay smoothing.
   - Calcula 1ª e 2ª derivadas por Savitzky-Golay.
   - Aplica mean centering usando a média do treino.
   - Salva os arquivos processados em:
     - `data/processed/training/`
     - `data/processed/validation/`

3. **Grid search (`scripts/run_grid_search.py`)**
   - Treina `RandomForestClassifier` em cada combinação de hiperparâmetros.
   - Usa GPU via cuML quando CUDA está disponível.
   - Usa scikit-learn como fallback em CPU.
   - Avalia cada modelo no conjunto de validação.
   - Salva:
     - modelos em `models/*.joblib`
     - métricas em `resultados_grid_search_validacao.csv`

## Execução

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute com os caminhos padrão:

```bash
python main.py
```

Ou informe os arquivos manualmente:

```bash
python main.py \
  --spectra-file data/RawSpectra_RoastedCoffee.xlsx \
  --quality-file data/SensoryQuality_RoastedCoffee.xlsx
```

## Execução por etapa

```bash
python scripts/run_split.py
python scripts/run_preprocessing.py
python scripts/run_grid_search.py
```

O utilitário abaixo reavalia modelos salvos na validação e gera um consolidado ordenado:

```bash
python scripts/run_validation.py
```

## Backend da Random Forest

Por padrão, o backend é escolhido automaticamente:

- `cuml`, quando há GPU CUDA disponível.
- `sklearn`, quando não há GPU acessível.

Para forçar um backend:

```bash
COFFEE_NIR_RF_BACKEND=gpu python main.py
COFFEE_NIR_RF_BACKEND=cpu python main.py
```
