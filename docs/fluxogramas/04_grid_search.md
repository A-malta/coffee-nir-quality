# Fluxograma 04 - Otimização e treinamento dos modelos

Fluxograma detalhado da etapa de seleção de variáveis, otimização bayeseana de hiperparâmetros e treinamento final do pipeline, utilizando terminologia teórica.

## Convenção visual

- Terminador: início ou fim do processo.
- Paralelogramo: entrada ou saída de dados/resultados.
- Retângulo: processo, transformação ou análise.
- Losango: decisão, repetição ou seleção.

```mermaid
---
config:
  theme: neutral
---
flowchart TB
    subgraph LassoSelection["Fase 1: Seleção de Variáveis (Lasso)"]
        direction TB
            L2["Padronização de Dados (Z-score)"]
            L1["Ajuste do Seletor (Lasso)"]
            L3["Regressão Logística com Penalização L1"]
            L4["Extrair coeficientes máximos por variável"]
            L5{"Magnitude do coeficiente <= Limiar?"}
            L6["Manter variável no subconjunto"]
            L7["Descartar variável"]
            L10["Gerar máscara de seleção final"]
    end
    subgraph ObjectiveFunction["Função Objetivo (Avaliação)"]
        direction TB
            CV2["Treinar Florestas Aleatórias"]
            CV1["Validação Cruzada Estratificada (5-fold)"]
            CV3["Predizer Partição de Teste"]
            CV4["Calcular Sensibilidade Mínima entre Classes"]
            CV5["Média do Desempenho entre Dobras"]
    end
    subgraph OptunaSearch["Fase 2: Otimização Bayesiana"]
        direction TB
            O2["Amostrador TPE (Tree-structured Parzen Estimator)"]
            O1["Definir Espaço de Busca"]
            ObjectiveFunction
            O3{"Número de tentativas atingido?"}
            O4["Retornar Melhor Configuração"]
    end
    subgraph FinalTraining["Fase 3: Treinamento Final e Avaliação"]
        direction TB
            F2["Reajustar Seletor Lasso"]
            F1["Ranquear Melhores Candidatos"]
            F3["Reduzir Dimensionalidade do Treino"]
            F4["Treinar Florestas Aleatórias"]
            F5["Montar Pipeline de Modelagem"]
            F6["Avaliação de Desempenho Multiclasse"]
    end
        START([Início da etapa]) --> LOAD[/"Dados de Treinamento<br>(Espectros + Classes)"/]
        START -. "reservado" .-> VAL[/"Conjunto de Validação<br>(estritamente isolado)"/]
        LOAD --> L1
        L1 --> L2
        L2 --> L3
        L3 --> L4
        L4 --> L5
        L5 -- "Sim" --> L7
        L5 -- "Não" --> L6
        L7 --> L10
        L6 --> L10
        L10 --> D_TYPE{"Próximo tipo<br>de dado?"}
        D_TYPE -- "Raw" --> DATA_RAW[/"Espectros Brutos<br>(variáveis reduzidas)"/]
        D_TYPE -- "SG_1D+2D+MeanCentering" --> DATA_PRE[/"Espectros Pré-processados<br>(variáveis reduzidas)"/]
        D_TYPE -- "Todos avaliados" --> RANKING["RANKING"]
        O1 --> O2
        CV1 --> CV2
        CV2 --> CV3
        CV3 --> CV4
        CV4 --> CV5
        O2 --> ObjectiveFunction
        CV5 --> O3
        O3 -- "Não" --> O2
        O3 -- "Sim" --> O4
        DATA_RAW --> O1
        DATA_PRE --> O1
        O4 --> D_TYPE
        F1 --> F2
        F2 --> F3
        F3 --> F4
        F4 --> F5
        F5 --> F6
        RANKING --> F1
        F6 --> EXPORT[/"Modelos Serializados<br>Resultados de Treinamento (CSV)"/]
        EXPORT --> END([Fim da etapa])
        VAL -.-> END

        style LassoSelection fill:#fcfcfc
        style OptunaSearch fill:#fcfcfc
```

## Espaço de busca (Otimização Bayesiana)

| Hiperparâmetro       | Espaço de busca (Range)    | Distribuição       | Papel no modelo                              |
|----------------------|----------------------------|--------------------|----------------------------------------------|
| `n_estimators`       | 300 - 700                  | Discreta (passo 50)| Número de árvores no ensemble                |
| `max_depth`          | 8 - 15                     | Discreta (passo 1) | Profundidade máxima de cada árvore           |
| `min_samples_split`  | 8 - 20                     | Discreta (passo 1) | Mínimo de amostras para dividir um nó        |
| `min_samples_leaf`   | 1 - 8                      | Discreta (passo 1) | Mínimo de amostras em uma folha              |
| `max_features`       | 0.1 - 0.6                  | Contínua Uniforme  | Fração de variáveis avaliadas em cada divisão|
| `bootstrap`          | [True, False]              | Categórica         | Amostragem com reposição entre árvores       |

## Detalhes Teóricos da Implementação

### Seleção de Variáveis (Lasso)
- **Padronização**: A aplicação do Z-score garante que todas as bandas espectrais estejam na mesma escala, permitindo que a penalização L1 identifique corretamente as variáveis mais informativas.
- **Penalização L1**: Utilizada para induzir esparsidade no modelo, forçando coeficientes de variáveis irrelevantes a zero.
- **Limiar de Seleção**: Um limiar de `1e-8` é aplicado para garantir a exclusão de variáveis com impacto estatisticamente insignificante.

### Métrica de Otimização: Sensibilidade Mínima entre Classes
Esta métrica prioriza a classe com o menor desempenho individual (Recall), garantindo que o modelo não negligencie categorias minoritárias ou de difícil detecção, promovendo a equidade no desempenho preditivo.

### Pipeline de Modelagem
O modelo final é estruturado como um fluxo sequencial que encapsula a seleção de variáveis e a classificação. Isso assegura que qualquer dado novo seja submetido à mesma transformação dimensional antes da inferência, mantendo a integridade estatística do processo.

## Entradas e Saídas
- **Entradas**: Configurações de busca e conjuntos de dados espectrais.
- **Saídas**: Melhores modelos candidatos e relatório de métricas de treinamento.


