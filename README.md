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
   - Calcula a 2ª derivada numérica sobre o resultado da 1ª derivada.
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
     - `plots/espectros_nir_preprocessados_SG_1D+2D+MeanCentering.png`

4. **Grid search (`scripts/run_grid_search.py`)**
   - Lê a recipe YAML informada na execução da pipeline.
   - Treina `RandomForestClassifier` em cada combinação de hiperparâmetros.
   - Repete o treinamento com os mesmos parâmetros para:
     - espectros brutos (`Raw`)
     - espectros pré-processados (`SG_1D+2D+MeanCentering`)
   - Registra métricas calculadas no conjunto de treino.
   - Salva:
     - modelos em `models/*.joblib`
     - métricas em `resultados_grid_search_treinamento.csv`

5. **Validação final (`scripts/run_validation.py`)**
   - Busca os modelos salvos em `models/*.joblib`.
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
  --recipe recipes/random_forest_grid_search.yaml
```
