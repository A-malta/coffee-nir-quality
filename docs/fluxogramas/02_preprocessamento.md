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

    style PIPE fill:#f9f9f9

## Entradas

- Espectros NIR do conjunto de treinamento.
- Espectros NIR do conjunto de validação.

## Saídas

- Matriz espectral de treinamento pré-processada.
- Matriz espectral de validação pré-processada, mantida fora do ajuste dos modelos.
- Sinais preparados para modelagem multivariada.
