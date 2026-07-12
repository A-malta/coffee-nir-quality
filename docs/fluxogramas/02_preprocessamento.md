# Figura 5 - Pré-processamento espectral

Reprodução da Figura 5 do TCC (página 30), preservando o conteúdo visual original.

![Fluxograma de execução dos pré-processamentos espectrais.](assets/figura-05-preprocessamento-espectral.png)

**Figura 5 – Fluxograma de execução dos pré-processamentos espectrais.**

Fonte: Elaborado pela autora.

## Procedimento descrito no TCC

O pré-processamento foi aplicado sequencialmente:

1. filtro de Savitzky-Golay, com janela 15 e polinômio de grau 2;
2. primeira derivada espectral;
3. *mean centering*, subtraindo de cada comprimento de onda a média do respectivo conjunto.

Treinamento e validação foram processados separadamente para preservar a independência do conjunto de validação. Os espectros sem transformação também foram mantidos para comparação com a versão `SG_1D+MeanCentering`.

## Conjuntos resultantes

- dados brutos de treinamento;
- dados brutos de validação;
- dados de treinamento `SG_1D+MeanCentering`;
- dados de validação `SG_1D+MeanCentering`.
