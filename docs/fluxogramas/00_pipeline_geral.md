# Fluxograma 00 - Pipeline geral

Visão metodológica integrada da pipeline de classificação da qualidade sensorial de café torrado a partir de espectros NIR.

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
---
flowchart TD
    A([Início da pipeline]) --> B[/Dados iniciais<br/>espectros NIR brutos e qualidade sensorial/]

    B --> C[[Divisão treino/validação<br/>dos dados brutos]]

    C --> F[/Conjunto de treinamento bruto/]
    C --> G[/Conjunto de validação bruto/]

    F --> D[[Pré-processamento espectral]]
    G --> D

    D --> H[/Conjunto de treinamento tratado/]
    D --> I[/Conjunto de validação tratado/]

    F --> M[[Otimização de hiperparâmetros<br/>e treinamento dos modelos]]
    H --> M

    M --> N[/Modelos candidatos treinados/]

    N --> O[[Validação final]]
    G --> O
    I --> O

    O --> P[/Resultados finais<br/>métricas preditivas, matrizes de confusão<br/>e comparação entre modelos/]
    P --> Q([Fim da pipeline])
```

## Etapas detalhadas

- [01 - Divisão treino/validação](01_divisao_dados.md): Separação estratificada utilizando o algoritmo Kennard-Stone para garantir representatividade espectral.
- [02 - Pré-processamento espectral](02_preprocessamento.md): Tratamento matemático dos sinais (Savitzky-Golay, derivadas e centralização) aplicado de forma independente para evitar vazamento de dados.
- [03 - Visualização dos espectros](03_visualizacao_espectros.md): Análise exploratória e gráfica dos perfis espectrais antes e após o tratamento.
- [04 - Otimização e treinamento](04_grid_search.md): Busca exaustiva (Grid Search) pelos melhores hiperparâmetros do classificador Random Forest.
- [05 - Validação final](05_validacao_final.md): Avaliação do poder preditivo dos modelos em um conjunto de dados estritamente não visto durante o treinamento.
