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
  fontSize: 18
  flowchart:
    curve: basis
    nodeSpacing: 50
    rankSpacing: 58
  themeVariables:
    primaryColor: '#F1FAF6'
    primaryBorderColor: '#75B99A'
    primaryTextColor: '#263238'
    lineColor: '#7AA695'
    secondaryColor: '#FFF7E6'
    tertiaryColor: '#EEF6FF'
---
flowchart TD
    A([Início da pipeline]) --> B[/Dados iniciais<br/>espectros NIR e qualidade sensorial/]

    B --> C[[Divisão treino/validação<br/>dos dados]]

    C --> F[/Conjunto de treinamento/]
    C --> G[/Conjunto de validação/]

    F --> SG[[Savitzky-Golay]]
    G --> SG

    SG --> DER[[1ª derivada]]
    DER --> MC[[Mean Centering]]

    MC --> H[/Conjunto de treinamento tratado/]
    MC --> I[/Conjunto de validação tratado/]

    H --> M[[Otimização de hiperparâmetros<br/>e treinamento dos modelos]]

    M --> N[/Modelos candidatos treinados/]

    N --> O[[Avaliação final do modelo]]
    I --> O

    O --> P[/Resultados finais<br/>métricas preditivas, matrizes de confusão<br/>e comparação entre modelos/]
    P --> Q([Fim da pipeline])

    classDef terminador fill:#E8F6EF,stroke:#62B58F,color:#263238
    classDef dados fill:#EAF2FF,stroke:#6C91C2,color:#263238
    classDef processo fill:#FFF3D6,stroke:#D4A83F,color:#263238

    class A,Q terminador
    class B,F,G,H,I,N,P dados
    class C,SG,DER,MC,M,O processo
```

## Etapas detalhadas

- [01 - Divisão treino/validação](01_divisao_dados.md): Separação estratificada utilizando o algoritmo Kennard-Stone para garantir representatividade espectral.
- [02 - Pré-processamento espectral](02_preprocessamento.md): Pré-tratamento sequencial dos sinais por Savitzky-Golay, 1ª derivada e Mean Centering, aplicado de forma independente para evitar vazamento de dados.
- [03 - Visualização dos espectros](03_visualizacao_espectros.md): Análise exploratória e gráfica dos perfis espectrais antes e após o tratamento.
- [04 - Otimização e treinamento](04_grid_search.md): Busca exaustiva (Grid Search) pelos melhores hiperparâmetros do classificador Random Forest.
- [05 - Avaliação final do modelo](05_validacao_final.md): Avaliação do poder preditivo dos modelos em um conjunto de dados estritamente não visto durante o treinamento.
