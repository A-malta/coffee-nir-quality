# Coffee NIR Quality Pipeline

Pipeline para classificar a qualidade sensorial de café torrado a partir de espectros NIR.

O fluxo completo faz a divisão treino/validação, gera uma versão pré-processada dos espectros,
treina modelos Random Forest com busca bayesiana de hiperparâmetros e valida os melhores modelos
no conjunto reservado.

## Etapas

Ao executar `python main.py`, o projeto roda cinco etapas:

1. **Split treino/validação (`src/data/splitter.py`)**
   - Lê espectros e qualidade sensorial.
   - Converte o eixo espectral inicial para comprimento de onda em nm quando o arquivo vem em número de onda.
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
   - Gera a versão `SG_1D+MeanCentering`.
   - Aplica Savitzky-Golay com janela 15, polinômio de ordem 2 e 1ª derivada.
   - Aplica mean centering separadamente em treino e validação.
   - Salva os arquivos processados em:
     - `data/processed/training/SG_1D+MeanCentering.xlsx`
     - `data/processed/validation/SG_1D+MeanCentering.xlsx`

3. **Visualização de espectros (`scripts/plot.py`)**
   - Lê os arquivos brutos de espectros e qualidade sensorial.
   - Lê os espectros pré-processados gerados na etapa anterior.
   - Gera gráficos dos espectros brutos e pré-processados coloridos por nota e por classe.
   - Salva as figuras em:
     - `plots/espectros_nir_brutos.png`
     - `plots/espectros_nir_brutos_por_classe.png`
     - `plots/espectros_nir_preprocessados_SG_1D+MeanCentering.png`
     - `plots/espectros_nir_preprocessados_SG_1D+MeanCentering_por_classe.png`

4. **Busca bayesiana (`scripts/run_bayesian_search.py`)**
   - Lê a recipe YAML informada na execução da pipeline.
   - Usa Optuna para procurar bons hiperparâmetros do `RandomForestClassifier`.
   - Avalia cada tentativa com `StratifiedKFold`, usando o número de folds definido em `cv_folds`.
   - Usa como critério `min_class_recall`, isto é, o menor recall entre as classes.
   - Aplica seleção de features com LASSO antes do Random Forest.
   - Repete a busca para:
     - espectros brutos (`Raw`)
     - espectros pré-processados (`SG_1D+MeanCentering`)
   - Ordena todos os candidatos das duas versões de espectros e salva os melhores modelos conforme `save_top_models`.
   - Salva:
     - modelos em `models/*.joblib`
     - métricas em `resultados_bayesian_search_treinamento.csv`
     - seleção dos dados brutos em `data/lasso_features_raw.xlsx`
     - seleção dos dados processados em `data/lasso_features_processed_SG_1D+MeanCentering.xlsx`
   - Nas duas planilhas, cada comprimento de onda recebe `1` quando mantido e `0` quando removido pelo LASSO.
   - O CSV inclui `cv_score`, hiperparâmetros, métricas no treino, seletor usado, quantidade de features selecionadas e nome do arquivo do modelo.

5. **Validação final (`scripts/run_validation.py`)**
   - Busca os modelos listados em `resultados_bayesian_search_treinamento.csv`.
   - Avalia cada modelo no conjunto de validação correspondente ao pré-processamento indicado no nome do arquivo.
   - Ordena os resultados pelo menor recall de classe, recall médio por classe e acurácia.
   - Atualiza o ranking em `resultados_bayesian_search_treinamento.csv` para seguir a ordem da validação.
   - Gera matrizes de confusão normalizadas.
   - Salva:
     - métricas em `resultados_validacao_final.csv`
     - figuras em `confusion_matrices/*.png`

## Recipe

A recipe define o espaço de busca do Random Forest, o número de tentativas, o número de folds,
o critério de otimização e a configuração do LASSO.

A recipe disponível em `recipes/6.yaml` usa:

- `n_trials: 1000`
- `cv_folds: 5`
- `scoring: min_class_recall`
- `direction: maximize`
- `save_top_models: 20`
- hiperparâmetros buscados: `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features` e `bootstrap`
- seleção de features por LASSO com `C`, `threshold`, `max_iter` e `tol`

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
  --recipe recipes/6.yaml
```

## Entradas Esperadas

- O arquivo de espectros deve conter a aba `RawSpectra_RoastedCoffee`.
- O arquivo de qualidade sensorial deve conter a aba `Cup quality_RoastedCoffee`.
- A coluna usada como alvo de classificação é `Class`.

## Saídas

- `data/raw_split/`: espectros e qualidade separados em treino e validação.
- `data/processed/`: espectros pré-processados para treino e validação.
- `data/lasso_features_raw.xlsx`: features mantidas (`1`) e removidas (`0`) nos dados brutos.
- `data/lasso_features_processed_SG_1D+MeanCentering.xlsx`: features mantidas (`1`) e removidas (`0`) nos dados processados.
- `plots/`: gráficos dos espectros brutos e pré-processados.
- `models/`: modelos treinados em formato `.joblib`.
- `resultados_bayesian_search_treinamento.csv`: ranking e métricas dos modelos salvos após a busca.
- `resultados_validacao_final.csv`: métricas finais no conjunto de validação.
- `confusion_matrices/`: matrizes de confusão normalizadas dos modelos validados.

## Scripts Auxiliares

As etapas também podem ser chamadas individualmente pelos módulos em `scripts/`, desde que os
artefatos esperados pelas etapas anteriores já existam. O script `scripts/filter_results.py`
mantém em `resultados_bayesian_search_treinamento.csv` apenas os modelos presentes em
`resultados_validacao_final.csv`.
