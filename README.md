# Coffee NIR Quality Pipeline

Este repositório executa um pipeline completo para **classificação da qualidade sensorial de café torrado** a partir de espectros NIR.

## O que o pipeline faz (passo a passo)

Ao executar `python main.py`, o projeto roda **4 etapas sequenciais**:

1. **Divisão de dados (`scripts/run_split.py`)**
   - Lê os arquivos brutos em `data/raw/`:
     - `RawSpectra_RoastedCoffee.xlsx` (espectros)
     - `SensoryQuality_RoastedCoffee.xlsx` (qualidade sensorial)
   - Aplica o algoritmo **Kennard-Stone** para selecionar amostras representativas.
   - Cria a divisão aproximada **80/10/10** em:
     - treino (`training`)
     - teste (`test`)
     - validação (`validation`)
   - Salva os arquivos em `data/raw_split/`:
     - `training_spectra.xlsx`, `test_spectra.xlsx`, `validation_spectra.xlsx`
     - `training_quality.xlsx`, `test_quality.xlsx`, `validation_quality.xlsx`

2. **Pré-processamento espectral (`scripts/run_preprocessing.py`)**
   - Lê os espectros de `data/raw_split/`.
   - Aplica, nesta ordem:
     1) **Savitzky-Golay smoothing**  
     2) **1ª derivada** (Savitzky-Golay)  
     3) **2ª derivada** (Savitzky-Golay)  
     4) **Mean centering** usando média calculada no conjunto de treino
   - Gera 3 arquivos processados (treino/validação/teste) em:
     - `data/processed/training/`
     - `data/processed/validation/`
     - `data/processed/test/`
   - Nome do arquivo gerado:
     - `SG_Smoothing+1D+2D+MeanCentering.xlsx`

3. **Treinamento + busca de hiperparâmetros (`scripts/run_grid_search.py`)**
   - Para cada arquivo de pré-processamento disponível em `data/processed/training/`, executa grid search de `RandomForestClassifier`.
   - Grade atual:
     - `n_estimators`: 50, 100, 200, 300
     - `max_depth`: 5, 10, 20
     - `min_samples_split`: 2, 5
     - `min_samples_leaf`: 1, 2
     - `max_features`: None
     - `bootstrap`: True, False
   - Para cada combinação:
     - treina com treino
     - avalia em validação e teste
     - salva o modelo em `models/*.joblib`
     - registra métricas em CSV:
       - `resultados_grid_search_validacao.csv`
       - `resultados_grid_search_teste.csv`

4. **Validação final dos modelos (`scripts/run_validation.py`)**
   - Lê todos os modelos em `models/*.joblib`.
   - Identifica o pré-processamento pelo nome do arquivo do modelo.
   - Reavalia cada modelo nos conjuntos de **teste** e **validação**.
   - Calcula métricas:
     - `accuracy`
     - `precision` (weighted)
     - `recall` (weighted)
     - `specificity` (com suporte para binário e multiclasse)
   - Salva o consolidado em:
     - `resultados_validacao_final.csv`

---

## Fluxo de execução

```text
data/raw
  ├─ RawSpectra_RoastedCoffee.xlsx
  └─ SensoryQuality_RoastedCoffee.xlsx
        │
        ▼
run_split.py
        │
        ▼
data/raw_split (treino/teste/validação)
        │
        ▼
run_preprocessing.py
        │
        ▼
data/processed (treino/teste/validação)
        │
        ▼
run_grid_search.py
        │
        ├─ models/*.joblib
        ├─ resultados_grid_search_validacao.csv
        └─ resultados_grid_search_teste.csv
        │
        ▼
run_validation.py
        │
        ▼
resultados_validacao_final.csv
```

## Como executar

1. Instale dependências:

```bash
pip install -r requirements.txt
```

2. Garanta que os dados brutos estejam em `data/raw/` com os nomes esperados.

3. Execute o pipeline completo:

```bash
python main.py
```

## Execução por etapa (opcional)

Se quiser rodar manualmente:

```bash
python scripts/run_split.py
python scripts/run_preprocessing.py
python scripts/run_grid_search.py
python scripts/run_validation.py
```
