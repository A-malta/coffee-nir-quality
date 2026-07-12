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
  look: classic
  fontFamily: '''Open Sans Variable'', sans-serif'
  themeVariables:
    fontSize: 28
    fontFamily: '''Open Sans Variable'', sans-serif'
    primaryColor: '#F1FAF6'
    primaryBorderColor: '#75B99A'
    primaryTextColor: '#263238'
    lineColor: '#7AA695'
    secondaryColor: '#FFF7E6'
    tertiaryColor: '#EEF6FF'
---
flowchart TD
    START([Início da etapa]) --> LOAD[/Dados de treinamento<br/>espectros e classes/]
    
    subgraph F1 [Fase 1: Otimização de Hiperparâmetros]
        direction LR
        GS[[Busca em Grade Prévia<br/>Exploração inicial]] --> OPTUNA[[Otimização Bayesiana<br/>Optuna TPE]]
    end
    
    LOAD --> GS
    OPTUNA --> CV
    
    subgraph F2 [Fase 2: Validação Cruzada Estratificada]
        direction TB
        CV{Validação Cruzada<br/>Stratified K-Fold}
        
        subgraph FOLD [Pipeline executada em cada Fold]
            direction LR
            TRAIN_FOLD[/Subconjunto de<br/>Treinamento/] --> LASSO[[Seleção de Variáveis<br/>Lasso L1 / saga]]
            LASSO --> RF[[Treinamento do Classificador<br/>Random Forest]]
            
            VAL_FOLD[/Subconjunto de<br/>Validação/] --> EVAL[[Avaliação do<br/>Desempenho]]
            
            RF --> EVAL
        end
        
        CV -->|Separação| TRAIN_FOLD
        CV -->|Separação| VAL_FOLD
        
        EVAL --> METRIC[/Métrica: minimum<br/>class recall/]
        METRIC --> SCORE[[Cálculo do CV Score<br/>Média dos folds]]
    end
    
    SCORE --> BEST10
    
    subgraph F3 [Fase 3: Treinamento dos Modelos Finais]
        direction LR
        BEST10[/10 melhores<br/>combinações/] --> FINAL_LASSO[[Seleção de Variáveis<br/>Todo o conjunto de treino]]
        FINAL_LASSO --> FINAL_RF[[Treinamento Final<br/>Random Forest]]
        FINAL_RF --> EXPORT[/Modelos finais<br/>e resultados/]
    end
    
    LOAD -.-> FINAL_LASSO
    EXPORT --> END([Fim da etapa])

    classDef terminador fill:#E8F6EF,stroke:#62B58F,color:#263238
    classDef dados fill:#EAF2FF,stroke:#6C91C2,color:#263238
    classDef processo fill:#FFF3D6,stroke:#D4A83F,color:#263238
    classDef decisao fill:#F4E5F7,stroke:#A363B3,color:#263238

    class START,END terminador
    class LOAD,TRAIN_FOLD,VAL_FOLD,METRIC,BEST10,EXPORT dados
    class GS,OPTUNA,LASSO,RF,EVAL,SCORE,FINAL_LASSO,FINAL_RF processo
    class CV decisao
    
    style F1 fill:#fafafa,stroke:#cccccc,stroke-dasharray: 5 5
    style F2 fill:#fafafa,stroke:#cccccc,stroke-dasharray: 5 5
    style F3 fill:#fafafa,stroke:#cccccc,stroke-dasharray: 5 5
    style FOLD fill:#ffffff,stroke:#bbbbbb
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
