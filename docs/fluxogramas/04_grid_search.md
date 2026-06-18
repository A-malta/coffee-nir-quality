# Fluxograma 04 - Otimização e treinamento dos modelos

Fluxograma metodológico da etapa de busca exaustiva de hiperparâmetros e treinamento supervisionado com Random Forest.

## Convenção visual

- Terminador: início ou fim do processo.
- Paralelogramo: entrada ou saída de dados/resultados.
- Retângulo: processo, transformação ou análise.
- Losando: decisão, repetição ou seleção.

```mermaid
---
config:
  theme: neutral
  look: classic
  fontFamily: '''Open Sans Variable'', sans-serif'
---
flowchart TB
    A(["Início da etapa"]) --> INP

    subgraph INP["Entradas"]
        direction LR
        TR[/"Treinamento bruto<br>& pré-processado"/]
        VAL[/"Validação (reservada,<br>não usada aqui)"/]
    end

    subgraph GS["Grid Search Exaustivo"]
        direction TB
        G["Todas as combinações de hiperparâmetros<br>n_estimators × max_depth × min_samples_split<br>× min_samples_leaf × max_features"]
        G --> H{"Dados brutos<br>ou pré-processados?"}
        H -- Bruto --> HB["Treinamento com<br>espectros brutos"]
        H -- Pré-proc --> HP["Treinamento com<br>SG_1D+MeanCentering"]
    end

    subgraph RF["Classificador Random Forest"]
        direction TB
        rf1["Construção de N árvores de decisão<br>n_estimators ∈ {300, 500, 700}"]
        rf2["Amostragem bootstrap dos dados<br>cada árvore vê um subconjunto aleatório"]
        rf3["Seleção aleatória de features<br>max_features ∈ {0.2, 0.25, 0.3, 0.35, 0.4}"]
        rf4["Crescimento controlado das árvores<br>max_depth ∈ {6, 8, 10, 12, 15}<br>min_samples_split ∈ {8, 10, 12, 15, 20}<br>min_samples_leaf ∈ {1, 2, 3, 4}"]
        rf5["Predição por votação majoritária<br>das N árvores"]
        rf1 --> rf2 --> rf3 --> rf4 --> rf5
    end

    subgraph MET["Avaliação no conjunto de treinamento"]
        direction LR
        m1["Acurácia"]
        m2["Precisão"]
        m3["Sensibilidade (Recall)"]
        m4["Especificidade"]
    end

    INP --> GS
    HB & HP --> RF
    RF --> MET
    MET -->|"registrar e repetir para<br>cada combinação"| H

    MET --> OUT[/"Modelos treinados (.joblib)<br>Métricas de ajuste (.csv)"/]
    VAL -.->|"permanece reservado"| FIM
    OUT --> FIM(["Fim da etapa"])
```

## Espaço de hiperparâmetros

| Hiperparâmetro       | Valores testados              | Papel no modelo                                  |
|----------------------|-------------------------------|--------------------------------------------------|
| `n_estimators`       | 300, 500, 700                 | Número de árvores do ensemble                    |
| `max_depth`          | 6, 8, 10, 12, 15              | Profundidade máxima de cada árvore               |
| `min_samples_split`  | 8, 10, 12, 15, 20             | Mínimo de amostras para dividir um nó            |
| `min_samples_leaf`   | 1, 2, 3, 4                    | Mínimo de amostras em uma folha                  |
| `max_features`       | 0.2, 0.25, 0.3, 0.35, 0.4    | Fração de features avaliadas em cada divisão     |
| `bootstrap`          | True                          | Usa amostragem com reposição                     |

**Total de combinações**: 3 × 5 × 5 × 4 × 5 = **1500 por tipo de dado** (bruto e pré-processado), totalizando **3000 modelos treinados**.

## Entradas

- Conjuntos de treinamento bruto e pré-processado com rótulos de classe.
- Grade de hiperparâmetros definida em `recipes/random_forest_grid_search.yaml`.
- Conjunto de validação mantido estritamente reservado.

## Saídas

- Modelos candidatos treinados e serializados em `.joblib`.
- Métricas de ajuste de treinamento registradas em `.csv`.

