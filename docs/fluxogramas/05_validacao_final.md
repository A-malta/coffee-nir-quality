# Etapa 05 - Validação final

O TCC não apresenta um fluxograma independente para esta etapa. A Seção 4.6 descreve o procedimento em texto e por meio das equações reproduzidas abaixo.

## Procedimento descrito no TCC

1. Os quatro artefatos com maior *cross-validation score* foram carregados.
2. Modelos treinados com espectros brutos foram aplicados ao conjunto de validação bruto.
3. Modelos treinados com `SG_1D+MeanCentering` foram aplicados ao conjunto de validação submetido ao mesmo pré-processamento.
4. As classes preditas foram comparadas às classes reais.
5. As métricas globais e por classe foram calculadas.
6. Matrizes de confusão normalizadas em porcentagem foram geradas para representar os acertos e erros entre as classes sensoriais.

## Métricas

$$
\text{Acurácia} = \frac{TP + TN}{TP + TN + FP + FN}
$$

$$
\text{Precisão} = \frac{TP}{TP + FP}
$$

$$
\text{Recall} = \frac{TP}{TP + FN}
$$

$$
\text{Especificidade} = \frac{TN}{TN + FP}
$$

$$
\text{Balanced accuracy} = \frac{\text{Recall} + \text{Especificidade}}{2}
$$

`TP`, `TN`, `FP` e `FN` representam, respectivamente, verdadeiros positivos, verdadeiros negativos, falsos positivos e falsos negativos. A especificidade foi calculada pela abordagem um-contra-todos a partir da matriz de confusão.

## Saídas descritas

- métricas globais de acurácia, precisão, *recall*, especificidade e *balanced accuracy*;
- as mesmas medidas calculadas individualmente para cada classe;
- matrizes de confusão normalizadas em porcentagem.
