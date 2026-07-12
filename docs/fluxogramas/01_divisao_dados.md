# Fluxograma 01 - Divisão treino/validação

Fluxograma metodológico da etapa de separação dos dados experimentais em conjuntos de treinamento e validação. O procedimento representado corresponde à seleção dentro de uma classe e foi aplicado separadamente às duas classes sensoriais.

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
    nodeSpacing: 38
    rankSpacing: 42
  themeVariables:
    primaryColor: '#F1FAF6'
    primaryBorderColor: '#75B99A'
    primaryTextColor: '#263238'
    lineColor: '#7AA695'
    secondaryColor: '#FFF7E6'
    tertiaryColor: '#EEF6FF'
---
flowchart TB
 subgraph KS["Algoritmo de Kennard-Stone"]
        direction TB
        L2A["Calcular distâncias<br>euclidianas"]
        L2["Selecionar o par inicial<br>mais distante"]
        L3["Adicionar amostra com<br>maior distância mínima"]
        L4{"Atingiu 20%<br>da classe?"}

        L2A --> L2
  end
    A(["Início"]) --> B[/"Espectros NIR brutos<br>&amp; Qualidade Sensorial"/]
    B --> L1["Padronizar espectros"]
    L1 --> L2A
    L2 --> L3
    L3 --> L4
    L4 -- Não --> L3
    L4 -- "Selecionadas" --> Q[/"Validação da classe<br>(20%)"/]
    L4 -- "Não selecionadas" --> P[/"Treinamento da classe<br>(80%)"/]
    P --> R(["Fim"])
    Q --> R

    classDef terminador fill:#E8F6EF,stroke:#62B58F,color:#263238
    classDef dados fill:#EAF2FF,stroke:#6C91C2,color:#263238
    classDef processo fill:#FFF3D6,stroke:#D4A83F,color:#263238
    classDef decisao fill:#F4FAFF,stroke:#7BA8D8,color:#263238

    class A,R terminador
    class B,P,Q dados
    class L1,L2A,L2,L3 processo
    class L4 decisao

    style KS fill:#F7F7F7,stroke:#D6D6D6,color:#263238
    linkStyle default stroke:#7AA695,stroke-width:2px
```

## Entradas

- Espectros NIR brutos.
- Tabela de qualidade sensorial das amostras.

## Saídas

- Conjunto de treinamento com espectros e classes.
- Conjunto de validação com espectros e classes, criado nesta etapa e mantido fora do treinamento.
- Separação representativa da variabilidade espectral dentro de cada classe, aplicada separadamente às duas classes sensoriais.
