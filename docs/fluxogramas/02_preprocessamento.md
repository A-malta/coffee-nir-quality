# Fluxograma 02 - Pré-processamento espectral

Fluxograma metodológico da etapa de tratamento dos espectros NIR antes da modelagem.

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
    fontFamily: '''Open Sans Variable'', sans-serif'
    primaryColor: '#F1FAF6'
    primaryBorderColor: '#75B99A'
    primaryTextColor: '#263238'
    lineColor: '#7AA695'
    secondaryColor: '#FFF7E6'
    tertiaryColor: '#EEF6FF'
---
flowchart TB
 subgraph PIPE["Pipeline de pré-processamento"]
    direction TB
        mc["Mean Centering"]
        d1["1ª Derivada espectral"]
        sg["Savitzky-Golay<br>janela=15, ord=2"]
  end
    A(["Início"]) --> F[/"Dados brutos - treinamento"/] & G[/"Dados brutos - validação"/]
    sg --> d1
    d1 --> mc
    F --> PIPE
    G --> PIPE
    F -. sem transformação .-> OUT_F[/"Dados brutos - treinamento"/]
    G -. sem transformação .-> OUT_G[/Dados brutos - validação/]
    PIPE --> OUT_O[/Dados de treinamento<br>SG_1D+MeanCentering/] & OUT_P[/"Dados de validação SG_1D+MeanCentering"/]
    OUT_F --> Q(["Fim"])
    OUT_G --> Q
    OUT_O --> Q
    OUT_P --> Q

    classDef terminador fill:#E8F6EF,stroke:#62B58F,color:#263238
    classDef dados fill:#EAF2FF,stroke:#6C91C2,color:#263238
    classDef processo fill:#FFF3D6,stroke:#D4A83F,color:#263238

    class A,Q terminador
    class F,G,OUT_F,OUT_G,OUT_O,OUT_P dados
    class sg,d1,mc processo

    style PIPE fill:#F7F7F7,stroke:#D6D6D6,color:#263238
    linkStyle default stroke:#7AA695,stroke-width:2px
```

## Entradas

- Espectros NIR do conjunto de treinamento.
- Espectros NIR do conjunto de validação.

## Saídas

- Matriz espectral de treinamento pré-processada.
- Matriz espectral de validação pré-processada, mantida fora do ajuste dos modelos.
- Sinais preparados para modelagem multivariada.
