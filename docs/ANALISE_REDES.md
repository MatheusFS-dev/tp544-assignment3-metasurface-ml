# Análise das redes

- Contagem pelas dimensões das camadas, incluindo pesos e biases.
- Dense: parâmetros = (entradas + 1) × saídas. Dropout e ativações não adicionam parâmetros.

| Rede | Dimensões | Parâmetros | MACs por amostra, forward | Pesos FP32, MB decimais |
|---|---|---:|---:|---:|
| Artigo, irrestrita | 24 → 24 → 300 → 300 → 300 → 300 → 1024 | 587,224 | 584,976 | 2.3489 |
| Artigo, restrita | 24 → 24 → 500 → 500 → 500 → 500 → 48 | 788,648 | 786,576 | 3.1546 |
| Reduzida | 8 → 32 → 64 → 16 | 3,440 | 3,328 | 0.0138 |

## Contagem por camada

| Rede | Camada | Entradas → saídas | Parâmetros | MACs |
|---|---:|---|---:|---:|
| Artigo, irrestrita | 1 | 24 → 24 | 600 | 576 |
| Artigo, irrestrita | 2 | 24 → 300 | 7,500 | 7,200 |
| Artigo, irrestrita | 3 | 300 → 300 | 90,300 | 90,000 |
| Artigo, irrestrita | 4 | 300 → 300 | 90,300 | 90,000 |
| Artigo, irrestrita | 5 | 300 → 300 | 90,300 | 90,000 |
| Artigo, irrestrita | 6 | 300 → 1024 | 308,224 | 307,200 |
| Artigo, restrita | 1 | 24 → 24 | 600 | 576 |
| Artigo, restrita | 2 | 24 → 500 | 12,500 | 12,000 |
| Artigo, restrita | 3 | 500 → 500 | 250,500 | 250,000 |
| Artigo, restrita | 4 | 500 → 500 | 250,500 | 250,000 |
| Artigo, restrita | 5 | 500 → 500 | 250,500 | 250,000 |
| Artigo, restrita | 6 | 500 → 48 | 24,048 | 24,000 |
| Reduzida | 1 | 8 → 32 | 288 | 256 |
| Reduzida | 2 | 32 → 64 | 2,112 | 2,048 |
| Reduzida | 3 | 64 → 16 | 1,040 | 1,024 |

## Interpretação

- As redes originais têm aproximadamente 171 e 229 vezes mais parâmetros que a reduzida.
- São maiores para este exercício, mas menos de um milhão de parâmetros não é uma rede excepcionalmente grande para GPUs atuais.
- A Tabela 1 informa 90.300 parâmetros em 24 → 300. O valor coerente é 7.500.
  A soma publicada é 670.024, a contagem pelas dimensões é 587.224.
- MACs medem o trabalho das camadas densas em uma passagem. Backward, Adam,
  checkpoints, transferências e uso da GPU afetam o tempo total.
- O artigo relata 84/81 minutos de treinamento e curvas de 10.000 épocas.
  Não informa o tempo total de geração das 2.000 simulações.
- Não é possível declarar o treinamento inviável em uma semana apenas por essas redes.
  O risco do prazo está na geração física dos dados, convergência e validação.

## Referência

1. Ghorbani et al., [artigo original](https://doi.org/10.1038/s41598-021-86588-2),
   Tabelas 1, 2 e 4 e Figuras 4 e 5. PDF preservado em `paper/`.
