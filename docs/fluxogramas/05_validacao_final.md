# Fluxograma 05 - Validação final

Fluxograma metodológico da etapa de avaliação independente dos modelos candidatos.

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
    A([Início da etapa]) --> B[/Modelos candidatos treinados (.joblib)<br/>gerados com o conjunto de treinamento/]
    A --> C[/Conjunto independente de validação<br/>criado no início e mantido reservado/]
    B --> D[Selecionar um modelo candidato<br/>para avaliação externa]
    D --> E[Identificar o tipo de entrada esperado<br/>pelo modelo selecionado (extraído do nome do arquivo)]
    C --> G[/Versões da validação<br/>bruta e pré-processada/]
    E --> F{Modelo treinado com<br/>espectros brutos?}
    G --> F
    F -- Sim --> H[Selecionar matriz de validação bruta]
    F -- Não --> I[Selecionar matriz de validação pré-processada<br/>(SG_1D+2D+MeanCentering)]
    H --> J[/Amostras de validação<br/>compatíveis com o modelo/]
    I --> J

    J --> K[Aplicar o modelo às amostras<br/>não usadas no treinamento]
    K --> L[Comparar classes previstas<br/>com classes observadas da validação]
    L --> M[Calcular métricas preditivas<br/>Acurácia, Precisão, Recall (Sensibilidade), Especificidade]
    M --> N[Construir matriz de confusão normalizada<br/>por classe real (normalize='true')]
    N --> O[Gerar heatmap da matriz com Seaborn<br/>colormap 'Blues', anotações em %]
    O --> P[Registrar desempenho do modelo<br/>em validação independente]

    P --> Q{Todos os modelos candidatos<br/>foram avaliados?}
    Q -- Não --> D
    Q -- Sim --> R[Ordenar resultados finais<br/>por acurácia de validação (val_accuracy)]
    R --> S[Salvar resultados em CSV<br/>e matrizes em PNG (300 dpi)]
    S --> T[/Métricas finais (.csv), matrizes de confusão (.png)<br/>e ranking dos modelos/]
    T --> U([Fim da etapa])
```

## Entradas

- Modelos candidatos treinados.
- Conjunto independente de validação com classes conhecidas.
- Versões bruta e pré-processada dos espectros de validação.

## Saídas

- Métricas finais de validação.
- Matrizes de confusão normalizadas.
- Comparação final dos modelos e seleção dos melhores desempenhos.
