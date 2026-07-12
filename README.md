# Classificação de Cafés Especiais com NIR e Random Forest

Pipeline desenvolvida para o TCC **Classificação de Cafés Especiais da Região de Huila, Colômbia, por Espectroscopia NIR Utilizando Algoritmo Random Forest**.

O projeto classifica amostras de café torrado e moído nas classes sensoriais `muito_bom` e `excelente` a partir de espectros FT-NIR. A implementação reúne divisão representativa dos dados, pré-processamento espectral, seleção de variáveis, otimização de Random Forest e validação em um conjunto reservado.

## Visão geral da pipeline

```mermaid
---
config:
  theme: neutral
  look: classic
  flowchart:
    curve: basis
    nodeSpacing: 50
    rankSpacing: 58
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

A execução é coordenada por `main.py` e percorre cinco etapas: divisão dos dados, pré-processamento, visualização, busca bayesiana e validação final.

## Dados versionados

Os arquivos de entrada estão em `data/`:

| Arquivo | Conteúdo |
|---|---|
| `data/RawSpectra_RoastedCoffee.xlsx` | 192 espectros e 2.001 variáveis espectrais; aba `RawSpectra_RoastedCoffee` |
| `data/SensoryQuality_RoastedCoffee.xlsx` | Identificadores, notas e classes sensoriais; aba `Cup quality_RoastedCoffee` |

Os dados derivam dos dois arquivos de café torrado do conjunto *Fourier Transform Near Infrared (FT-NIR) spectra and sensory scores in green and roasted specialty coffee for machine learning-based quality monitoring*, versão 3, de Gentil Andres Collazos-Escobar, Ever M. Morales-Angulo, Andrés Felipe Bahamón Monje e Nelson Gutierrez Guzman ([Mendeley Data, DOI 10.17632/nz2fr76trm.3](https://doi.org/10.17632/nz2fr76trm.3)). O conjunto é distribuído sob a licença [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); consulte também o [artigo de dados associado](https://doi.org/10.1016/j.dib.2025.111609).

Para adequação à pipeline, os cabeçalhos espectrais foram consolidados, os identificadores de amostra e réplica foram combinados no formato `amostra_réplica`, os registros sensoriais foram reorganizados e a coluna `Class` foi derivada da nota: `excelente` para valores maiores ou iguais a 85 e `muito_bom` para os demais. Os valores de absorbância e as pontuações sensoriais foram preservados. Normalização de rótulos e conversão do eixo espectral ocorrem durante a execução, sem sobrescrever os XLSX versionados.

A pipeline utiliza a coluna `Class` como variável resposta. Quando o eixo espectral está em número de onda, ele é convertido para comprimento de onda, cobrindo aproximadamente 833 a 2.500 nm.

> **Nota de reprodutibilidade:** a planilha versionada contém 114 espectros `muito_bom` e 78 `excelente`. O texto e a Tabela 1 do TCC informam 111 e 81, respectivamente. A divisão e as métricas arquivadas correspondem aos dados versionados: 153 espectros de treinamento e 39 de validação, enquanto o texto do TCC informa 154 e 38.

## Partes principais

### 1. Divisão em treinamento e validação

```mermaid
---
config:
  theme: neutral
  look: classic
  flowchart:
    curve: basis
    nodeSpacing: 38
    rankSpacing: 42
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

O algoritmo Kennard–Stone seleciona aproximadamente 20% dos espectros de cada classe para validação. A seleção procura cobrir a variabilidade espectral da classe, enquanto as amostras restantes formam o conjunto de treinamento.

### 2. Pré-processamento e visualização

```mermaid
---
config:
  theme: neutral
  look: classic
  flowchart:
    curve: basis
    nodeSpacing: 50
    rankSpacing: 55
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

O tratamento `SG_1D+MeanCentering` aplica Savitzky-Golay com janela 15, polinômio de grau 2 e primeira derivada, seguido de centralização pela média. Treinamento e validação são processados separadamente, e as versões brutas são preservadas para comparação.

A visualização gera quatro gráficos: espectros brutos e pré-processados, coloridos por pontuação sensorial e por classe.

### 3. Seleção de variáveis e otimização

```mermaid
---
config:
  theme: neutral
  look: classic
  flowchart:
    curve: basis
    nodeSpacing: 42
    rankSpacing: 50
---
flowchart TD
    A([Início da etapa]) --> B[/Dados de treinamento<br/>espectros e classes/]
    B --> C[[Grid Search:<br/>Exploração do espaço de<br/>hiperparâmetros]]
    C --> D[[Otimização Bayesiana<br/>Algoritmo Optuna]]
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
    class C,D,H,I,J,L,N,O processo
    class E decisao

    linkStyle default stroke:#7AA695,stroke-width:2px
```

A regressão logística com penalização L1 seleciona os comprimentos de onda dentro de cada *fold*. Em seguida, o Random Forest é avaliado por validação cruzada estratificada, usando como objetivo o menor *recall* entre as classes. Uma execução materializa os folds uma única vez e os reutiliza em todas as tentativas, tanto nos espectros brutos quanto em `SG_1D+MeanCentering`, permitindo comparar os `cv_score` sob as mesmas partições.

O Grid Search mostrado no fluxograma foi uma etapa preliminar do TCC. Ele não é reexecutado por `main.py`, pois o texto informa apenas os limites e não os valores discretos da grade. A execução principal utiliza Optuna com o amostrador TPE nos intervalos refinados.

### 4. Validação final

As dez melhores combinações da busca conjunta são reajustadas com todo o conjunto de treinamento. Os quatro modelos com maior `cv_score` são aplicados ao conjunto reservado correspondente à sua versão espectral.

São calculadas acurácia, precisão, *recall*, especificidade e métricas balanceadas, globalmente e por classe. As matrizes de confusão são normalizadas pela classe real.

## Recipe do TCC

O repositório mantém uma única configuração executável: [`recipes/tcc.yaml`](recipes/tcc.yaml).

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

O TCC não informa numericamente a quantidade de tentativas, o número de folds, `C`, limiar, tolerância ou iterações do LASSO. Esses valores estão identificados no YAML como *defaults* operacionais históricos, e não como valores extraídos do texto.

## Instalação

Requer Python 3.12. Na raiz do repositório:

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No Windows PowerShell, ative o ambiente com `venv\Scripts\Activate.ps1`.

## Como executar

Execute a pipeline completa a partir da raiz:

```bash
python main.py \
  --spectra-file data/RawSpectra_RoastedCoffee.xlsx \
  --quality-file data/SensoryQuality_RoastedCoffee.xlsx \
  --recipe recipes/tcc.yaml
```

A recipe executa 2.000 tentativas no total — 1.000 para dados brutos e 1.000 para dados pré-processados — com cinco folds por tentativa. O tempo de execução depende do processador e pode ser elevado.

Uma chamada de `main.py` produz uma execução. O TCC utilizou cinco execuções sem *seed*; para repeti-las, execute o comando cinco vezes e preserve as saídas antes da chamada seguinte, pois os caminhos da raiz são reutilizados.

## Saídas de uma execução

| Caminho | Conteúdo |
|---|---|
| `data/raw_split/` | Espectros e classes de treinamento/validação |
| `data/processed/` | Espectros `SG_1D+MeanCentering` |
| `data/lasso_features_*.xlsx` | Máscaras de comprimentos de onda do LASSO reajustado em todo o treino para cada versão espectral |
| `plots/` | Gráficos dos espectros |
| `models/` | Dez pipelines treinadas em formato `.joblib` |
| `resultados_bayesian_search_treinamento.csv` | Ranking por validação cruzada e métricas de treinamento |
| `resultados_validacao_final.csv` | Métricas dos quatro modelos no conjunto reservado, preservando `cv_rank`, `cv_score` e ranking final |
| `confusion_matrices/` | Quatro matrizes de confusão normalizadas |

## Resultados apresentados no TCC

O recorte versionado está em [`resultados_tcc/`](resultados_tcc/):

- `rep_01/` a `rep_05/`: CSVs de treinamento e validação e planilhas de seleção LASSO;
- `figuras/`: Figuras 6–9 e 11 apresentadas no TCC;
- modelos `.joblib`, espectros intermediários e arquivos duplicados não são versionados.

O melhor modelo apresentado foi obtido na segunda repetição e alcançou:

| Acurácia | Precisão | Recall | Especificidade | Balanced accuracy |
|---:|---:|---:|---:|---:|
| 0,769 | 0,772 | 0,769 | 0,766 | 0,766 |

![Matriz de confusão normalizada do melhor modelo Random Forest](resultados_tcc/figuras/figura-11_matriz_confusao_melhor_modelo.png)

### Proveniência dos snapshots

Os arquivos em `resultados_tcc/` preservam as execuções históricas usadas nas tabelas e figuras. Nelas, foram armazenados 50 candidatos por repetição e a Tabela 5 foi formada pelos quatro primeiros do ranking no conjunto reservado. A recipe atualmente versionada segue o procedimento escrito nas Seções 4.5 e 4.6: dez modelos finais e escolha dos quatro por `cv_score`. Por isso, uma nova execução não reproduzirá esses snapshots byte a byte.

Os fontes Mermaid dos quatro fluxogramas também estão isolados em [`docs/fluxogramas/`](docs/fluxogramas/).
