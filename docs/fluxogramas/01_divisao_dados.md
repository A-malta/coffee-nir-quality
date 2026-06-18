# Fluxograma 01 - Divisão treino/validação

Fluxograma metodológico da etapa de separação dos dados experimentais em conjuntos de treinamento e validação.

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
 subgraph KS["Algoritmo Kennard-Stone"]
        L1["Padronização espectral"]
        L2["Cálculo de distâncias<br>euclidianas"]
        L3["Selecionar as 2 amostras<br>mais distantes"]
        L4{"Atingiu 20%?"}
        L5["Adicionar amostra com<br>maior distância mínima"]
        L6["Adicionar subconjunto<br>à validação"]
  end
    A(["Início"]) --> A1["Classificação Binária<br>Excelente: 85-88 pts (n=78)<br>Muito Bom: 80-84.69 pts (n=114)"]
    A1 --> B[/"Espectros NIR brutos<br>&amp; Qualidade Sensorial"/]
    B --> I{"Todas as classes<br>avaliadas?"}
    I -- Não --> J["Selecionar classe"]
    J --> K["Calcular 20% das<br>amostras para validação"]
    K --> L["Seleção representativa<br>(intra-classe)"]
    L --> L1
    L1 --> L2
    L2 --> L3
    L3 --> L4
    L4 -- Não --> L5
    L5 --> L4
    L4 -- Sim --> L6
    L6 --> I
    I -- Sim --> N["Treinamento: amostras<br>restantes (80%)"]
    N --> O["Verificar estratificação"]
    O --> P[/"Treinamento (80%)"/] & Q[/"Validação (20%)"/]
    P --> R(["Fim"])
    Q --> R

    style KS fill:#f9f9f9
```

## Entradas

- Espectros NIR brutos.
- Tabela de qualidade sensorial das amostras.

## Saídas

- Conjunto de treinamento com espectros e classes.
- Conjunto de validação com espectros e classes, criado nesta etapa e mantido fora do treinamento.
- Separação estratificada por classe e representativa da variabilidade espectral.
