# Coffee NIR Quality Pipeline

Pipeline para classificar a qualidade sensorial de café torrado a partir de espectros NIR.

## Etapas

Ao executar `python main.py`, o projeto roda cinco etapas:

1. **Split treino/validação (`src/data/splitter.py`)**
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
   - Gera uma versão de pré-processamento.
   - Aplica suavização por Savitzky-Golay.
   - Calcula a 1ª derivada numérica.
   - Aplica mean centering separadamente em treino e validação.
   - Salva os arquivos processados em:
     - `data/processed/training/`
     - `data/processed/validation/`

3. **Visualização de espectros (`scripts/plot.py`)**
   - Lê os arquivos brutos de espectros e qualidade sensorial.
   - Lê os espectros pré-processados gerados na etapa anterior.
   - Gera um gráfico dos espectros brutos e um gráfico para cada versão pré-processada.
   - Salva as figuras em:
     - `plots/espectros_nir_brutos.png`
     - `plots/espectros_nir_preprocessados_SG_1D+MeanCentering.png`

4. **Busca bayesiana (`scripts/run_bayesian_search.py`)**
   - Lê a recipe YAML informada na execução da pipeline.
   - Usa Optuna para procurar bons hiperparâmetros do `RandomForestClassifier`.
   - Avalia cada tentativa em um holdout estratificado separado a partir do conjunto de treino.
   - Usa como critério o menor recall entre as classes, isto é, a menor diagonal principal da matriz de confusão.
   - Seleciona o holdout interno com Kennard-Stone dentro de cada classe para manter representatividade espectral.
   - Aplica seleção de features com LASSO antes do Random Forest quando `feature_selection.lasso.enabled` estiver ativo na recipe.
   - Repete a busca para:
     - espectros brutos (`Raw`)
     - espectros pré-processados (`SG_1D+MeanCentering`)
   - Treina e salva a fração superior definida por `save_top_fraction`.
   - Salva:
     - modelos em `models/*.joblib`
     - métricas em `resultados_bayesian_search_treinamento.csv`
   - O CSV inclui `holdout_score`, seletor usado e quantidade de features selecionadas.
   - Quantidade salva por pré-processamento: `ceil(n_trials * save_top_fraction)`.

5. **Validação final (`scripts/run_validation.py`)**
   - Busca os modelos listados em `resultados_bayesian_search_treinamento.csv`.
   - Se esse arquivo não existir, usa os modelos salvos em `models/*.joblib`.
   - Avalia cada modelo no conjunto de validação correspondente ao pré-processamento indicado no nome do arquivo.
   - Gera matrizes de confusão normalizadas.
   - Salva:
     - métricas em `resultados_validacao_final.csv`
     - figuras em `confusion_matrices/*.png`

## Execução

Instale as dependências:

```bash
pip install -r requirements.txt
```

Execute informando os arquivos brutos:

```bash
python main.py \
  --spectra-file data/RawSpectra_RoastedCoffee.xlsx \
  --quality-file data/SensoryQuality_RoastedCoffee.xlsx \
  --recipe recipes/random_forest_bayesian_search.yaml
```
