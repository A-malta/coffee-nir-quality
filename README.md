# Classificação de Cafés Especiais com NIR e Random Forest

Pipeline desenvolvida para o TCC **Classificação de Cafés Especiais da Região de Huila, Colômbia, por Espectroscopia NIR Utilizando Algoritmo Random Forest**.

O projeto classifica amostras de café torrado e moído nas classes sensoriais `muito_bom` e `excelente` a partir de espectros FT-NIR.

## Execução rápida

O projeto usa Python 3.12 e [`uv`](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/A-malta/coffee-nir-quality.git
cd coffee-nir-quality
uv sync --locked
```

```bash
uv run --locked python main.py --spectra-file data/RawSpectra_RoastedCoffee.xlsx --quality-file data/SensoryQuality_RoastedCoffee.xlsx --recipe recipes/01.yaml
```

## Visão geral da pipeline

<div align="center">

```mermaid
---
config:
  theme: neutral
  look: classic
  flowchart:
    curve: basis
    nodeSpacing: 32
    rankSpacing: 36
---
flowchart TD
    A([Início da pipeline]) --> B[/Dados iniciais<br/>espectros NIR e qualidade<br/>sensorial/]
    B --> C[[Divisão treino/validação<br/>dos dados]]

    C --> D[/Conjunto de treinamento/]
    C --> E[/Conjunto de validação/]

    D --> F[[Savitzky-Golay]]
    E --> F
    F --> G[[1ª derivada]]
    G --> H[[Mean Centering]]

    H --> I[/Conjunto de treinamento<br/>tratado/]
    H --> J[/Conjunto de validação tratado/]

    I --> K[[Otimização de<br/>hiperparâmetros<br/>e treinamento dos modelos]]
    K --> L[/Modelos candidatos treinados/]

    L --> M[[Avaliação final do modelo]]
    J --> M
    M --> N[/Resultados finais<br/>métricas preditivas, matrizes<br/>de confusão<br/>e comparação entre modelos/]
    N --> O([Fim da pipeline])

    classDef terminador fill:#E8F6EF,stroke:#62B58F,color:#263238
    classDef dados fill:#EAF2FF,stroke:#6C91C2,color:#263238
    classDef processo fill:#FFF3D6,stroke:#D4A83F,color:#263238

    class A,O terminador
    class B,D,E,I,J,L,N dados
    class C,F,G,H,K,M processo

    linkStyle default stroke:#7AA695,stroke-width:2px
```

</div>

## Estrutura do repositório

```text
.
├── data/
│   ├── RawSpectra_RoastedCoffee.xlsx
│   └── SensoryQuality_RoastedCoffee.xlsx
├── docs/
│   ├── 00_pipeline_geral.md
│   ├── 01_divisao_dados.md
│   ├── 02_preprocessamento.md
│   ├── 03_otimizacao.md
│   ├── figura-06_espectros_nir_brutos.png
│   ├── figura-07_espectros_nir_brutos_por_classe.png
│   ├── figura-08_espectros_nir_preprocessados.png
│   ├── figura-09_espectros_nir_preprocessados_por_classe.png
│   └── figura-11_matriz_confusao_melhor_modelo.png
├── recipes/
│   └── 01.yaml
├── scripts/
│   ├── __init__.py
│   ├── filter_results.py
│   ├── plot.py
│   ├── run_bayesian_search.py
│   ├── run_preprocessing.py
│   └── run_validation.py
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py
│   │   ├── kennard_stone.py
│   │   └── splitter.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py
│   ├── modeling/
│   │   ├── __init__.py
│   │   └── feature_selection.py
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── spectra.py
│   ├── __init__.py
│   └── config.py
├── .gitignore
├── .python-version
├── README.md
├── main.py
├── pyproject.toml
└── uv.lock
```

## Dados versionados

| Arquivo | Conteúdo |
|---|---|
| `data/RawSpectra_RoastedCoffee.xlsx` | 192 espectros e 2.001 variáveis espectrais; aba `RawSpectra_RoastedCoffee` |
| `data/SensoryQuality_RoastedCoffee.xlsx` | Identificadores, notas e classes sensoriais; aba `Cup quality_RoastedCoffee` |

Os dados derivam do conjunto *Fourier Transform Near Infrared (FT-NIR) spectra and sensory scores in green and roasted specialty coffee for machine learning-based quality monitoring*, versão 3, de Gentil Andres Collazos-Escobar, Ever M. Morales-Angulo, Andrés Felipe Bahamón Monje e Nelson Gutierrez Guzman ([Mendeley Data, DOI 10.17632/nz2fr76trm.3](https://doi.org/10.17632/nz2fr76trm.3)). O conjunto é distribuído sob a licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

## Partes principais

### 1. Divisão em treinamento e validação

<div align="center">

```mermaid
---
config:
  theme: neutral
  look: classic
  flowchart:
    curve: basis
    nodeSpacing: 32
    rankSpacing: 36
---
flowchart TB
    A([Início]) --> B[/Espectros NIR brutos<br/>&amp; Qualidade Sensorial/]
    B --> C[Padronizar espectros]

    subgraph KS["Algoritmo de Kennard-Stone"]
        direction TB
        D[Calcular distâncias<br/>euclidianas]
        E[Selecionar o par inicial<br/>mais distante]
        F[Adicionar amostra com<br/>maior distância mínima]
        G{Atingiu 20%<br/>da classe?}

        D --> E
        E --> F
        F --> G
        G -- Não --> F
    end

    C --> D
    G -- "Não selecionadas" --> H[/"Treinamento da classe<br/>(80%)"/]
    G -- Selecionadas --> I[/"Validação da classe<br/>(20%)"/]
    H --> J([Fim])
    I --> J

    classDef terminador fill:#E8F6EF,stroke:#62B58F,color:#263238
    classDef dados fill:#EAF2FF,stroke:#6C91C2,color:#263238
    classDef processo fill:#FFF3D6,stroke:#D4A83F,color:#263238
    classDef decisao fill:#F4FAFF,stroke:#7BA8D8,color:#263238

    class A,J terminador
    class B,H,I dados
    class C,D,E,F processo
    class G decisao

    style KS fill:#F7F7F7,stroke:#D6D6D6,color:#263238
    linkStyle default stroke:#7AA695,stroke-width:2px
```

</div>

### 2. Pré-processamento e visualização

<div align="center">

```mermaid
---
config:
  theme: neutral
  look: classic
  flowchart:
    curve: basis
    nodeSpacing: 32
    rankSpacing: 36
---
flowchart TB
    subgraph PIPE["Pipeline de pré-processamento"]
        direction TB
        SG[Savitzky-Golay<br/>janela=15, ord=2]
        D1[1ª Derivada espectral]
        MC[Mean Centering]

        SG --> D1
        D1 --> MC
    end

    A([Início]) --> B[/Dados brutos - treinamento/]
    A --> C[/Dados brutos - validação/]

    B --> SG
    C --> SG

    B -. sem transformação .-> D[/Dados brutos - treinamento/]
    C -. sem transformação .-> E[/Dados brutos - validação/]

    MC --> F[/Dados de treinamento<br/>SG_1D+MeanCentering/]
    MC --> G[/Dados de validação<br/>SG_1D+MeanCentering/]

    D --> H([Fim])
    E --> H
    F --> H
    G --> H

    classDef terminador fill:#E8F6EF,stroke:#62B58F,color:#263238
    classDef dados fill:#EAF2FF,stroke:#6C91C2,color:#263238
    classDef processo fill:#FFF3D6,stroke:#D4A83F,color:#263238

    class A,H terminador
    class B,C,D,E,F,G dados
    class SG,D1,MC processo

    style PIPE fill:#F7F7F7,stroke:#D6D6D6,color:#263238
    linkStyle default stroke:#7AA695,stroke-width:2px
```

</div>

<table width="100%">
  <tr>
    <th width="50%">Espectros brutos por pontuação sensorial</th>
    <th width="50%">Espectros brutos por classe</th>
  </tr>
  <tr>
    <td align="center" valign="top"><img src="docs/figura-06_espectros_nir_brutos.png" width="100%" alt="Espectros NIR brutos coloridos pela pontuação sensorial"></td>
    <td align="center" valign="top"><img src="docs/figura-07_espectros_nir_brutos_por_classe.png" width="100%" alt="Espectros NIR brutos coloridos pela classe sensorial"></td>
  </tr>
  <tr>
    <th width="50%">Espectros pré-processados por pontuação sensorial</th>
    <th width="50%">Espectros pré-processados por classe</th>
  </tr>
  <tr>
    <td align="center" valign="top"><img src="docs/figura-08_espectros_nir_preprocessados.png" width="100%" alt="Espectros NIR pré-processados coloridos pela pontuação sensorial"></td>
    <td align="center" valign="top"><img src="docs/figura-09_espectros_nir_preprocessados_por_classe.png" width="100%" alt="Espectros NIR pré-processados coloridos pela classe sensorial"></td>
  </tr>
</table>

### 3. Seleção de variáveis e otimização

<div align="center">

```mermaid
---
config:
  theme: neutral
  look: classic
  flowchart:
    curve: basis
    nodeSpacing: 32
    rankSpacing: 36
---
flowchart TD
    A([Início da etapa]) --> B[/Dados de treinamento<br/>espectros e classes/]
    B --> D[[Otimização Bayesiana<br/>Algoritmo Optuna]]
    D --> E{Validação Cruzada<br/>Stratified K-Fold}

    E -- "Repetição por fold" --> F[/Subconjunto de Validação/]
    E -- "Repetição por fold" --> G[/Subconjunto de Treinamento/]

    G --> H[[Seleção de Variáveis<br/>Regressão Logística Lasso L1]]
    H --> I[[Treinamento do Classificador<br/>Random Forest]]

    F --> J[[Avaliação do Desempenho]]
    I --> J

    J --> K[/Métrica de Otimização:<br/>minimum class recall/]
    K --> L[[Cálculo do CV Score:<br/>Média entre os folds]]
    L --> M[/Seleção das melhores<br/>combinações de<br/>hiperparâmetros/]

    M --> N[[Seleção de Variáveis Final<br/>Ajuste com todo o conjunto<br/>de treinamento]]
    B --> N
    N --> O[[Artefatos finais:<br/>modelo + features]]
    O --> P[/Resultados de validação/]
    P --> Q([Fim da etapa])

    classDef terminador fill:#E8F6EF,stroke:#62B58F,color:#263238
    classDef dados fill:#EAF2FF,stroke:#6C91C2,color:#263238
    classDef processo fill:#FFF3D6,stroke:#D4A83F,color:#263238
    classDef decisao fill:#F4E5F7,stroke:#A363B3,color:#263238

    class A,Q terminador
    class B,F,G,K,M,P dados
    class D,H,I,J,L,N,O processo
    class E decisao

    linkStyle default stroke:#7AA695,stroke-width:2px
```

</div>

## Recipe

| Parâmetro | Configuração |
|---|---|
| Tentativas | 1.000 para cada versão espectral |
| Validação cruzada | 5 folds, estratificada e embaralhada |
| Função objetivo | maximizar `min_class_recall` |
| `n_estimators` | 350–450, passo 50 |
| `max_depth` | 14–15, passo 1 |
| `min_samples_split` | 10–19, passo 1 |
| `min_samples_leaf` | 1–2, passo 1 |
| `max_features` | 0,20–0,35, uniforme |
| `bootstrap` | `true` ou `false` |
| Modelos finais | 10 |
| Modelos na validação reservada | 4, selecionados por `cv_score` |

## Saídas de uma execução

| Caminho |
|---|
| `data/raw_split/` |
| `data/processed/` |
| `data/lasso_features_*.xlsx` |
| `plots/` |
| `models/` |
| `resultados_bayesian_search_treinamento.csv` |
| `resultados_validacao_final.csv` |
| `confusion_matrices/` |

## Resultado de referência

| Acurácia | Precisão | Recall | Especificidade | Balanced accuracy |
|---:|---:|---:|---:|---:|
| 0,769 | 0,772 | 0,769 | 0,766 | 0,766 |

<p align="center">
  <img src="docs/figura-11_matriz_confusao_melhor_modelo.png" width="50%" alt="Matriz de confusão normalizada do melhor modelo Random Forest">
</p>
