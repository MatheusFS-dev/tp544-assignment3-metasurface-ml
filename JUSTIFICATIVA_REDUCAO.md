# Por que reduzimos o experimento

## O que o artigo fez

- Oito padrões de anéis, 16 posições independentes e células de 32 × 32 pixels.
- Espaço restrito: `8^16 = 281.474.976.710.656` layouts, cerca de 281 trilhões.
- Saída irrestrita: `2^1024` máscaras binárias possíveis. Isso não significa que
  os autores simularam máscaras arbitrárias ou percorreram esse espaço.
- Dataset utilizado: 2.000 geometrias amostradas da biblioteca de oito padrões,
  simuladas no CST. Apenas 70% dos exemplos foram usados para treinamento.
- Duas redes, curvas de 10.000 épocas e treinamento de 84/81 minutos.
- O PDF não fornece link para baixar o dataset, código ou checkpoints. Não temos
  acesso aos dados originais e precisamos gerar nossos próprios espectros.

## Por que uma reprodução completa é arriscada em uma semana

- Explorar exaustivamente os trilhões de layouts é inviável, mas isso não foi
  necessário no artigo. O problema prático é recriar as 2.000 simulações.
- Precisamos configurar o Lumerical, verificar materiais, malha e convergência,
  gerar os dados e reservar tempo para validar as previsões.
- O artigo não informa tempo por simulação ou tempo total de geração dos dados.
  A duração deve ser medida no piloto, antes de comprometer o prazo.
- Exemplo hipotético: a 10 minutos por layout, 2.000 simulações sequenciais
  levariam cerca de 333 horas. As 256 reduzidas levariam cerca de 43 horas.
  Isso exclui preflight, falhas, escrita de arquivos e validação final.
- As redes originais são muito maiores que a reduzida, mas não justificam afirmar
  que o treinamento seria impossível em uma semana. Os tempos publicados são
  próximos de 1,4 hora, depois de gerar os dados.
- Hardware: o artigo menciona Tesla K80 no Colab. A placa tem 12 GB por GPU e
  24 GB no total. O valor 13 MB do texto é incorreto. Sem medir a nossa máquina,
  não podemos afirmar que a K80 seja superior nem extrapolar seu tempo.

## Redução dos dados e da geometria

| Item | Artigo | Exercício reduzido |
|---|---|---|
| Padrões | 8 anéis | Vazio e 3 anéis quadrados definidos |
| Escolhas independentes | 16 | 4, espelhadas para preencher 16 posições |
| Combinações | 281.474.976.710.656 | 256 |
| Geometrias simuladas | 2.000 aleatórias | Até 256, enumeradas |
| Dataset utilizável | 2.000 pares reportados | Subconjunto com 1–2 notches válidos, mínimo operacional de 24 |
| Notches / entradas | Até 8 / 24 descritores | Até 2 / 8 descritores, incluindo presença |
| Simulador | CST | Lumerical FDTD via `lumapi` |
| Separação | 70% treino, 30% teste | 70% treino, 15% validação, restante teste |
| Dimensões / banda | 6,4 mm, FR4 de 1,5 mm, 4–45 GHz | Mantidas nominalmente |

## Redução da rede

| Item | Artigo, irrestrita | Artigo, restrita | Exercício |
|---|---|---|---|
| Dimensões | 24 → 24 → 300 → 300 → 300 → 300 → 1024 | 24 → 24 → 500 → 500 → 500 → 500 → 48 | 8 → 32 → 64 → 16 |
| Parâmetros pelas dimensões | 587.224 | 788.648 | 3.440 |
| Saída | Pixels, sigmoid | Bits de padrões, sigmoid | 4 decisões categóricas de 4 classes |
| Perda | MSE | MSE | Entropia cruzada |
| Épocas | 10.000 nas figuras | 10.000 nas figuras | Até 500, parada após 30 épocas sem melhora |
| Batch / otimizador | 30 / Adam | 30 / Adam | 16 / Adam |

- A rede reduzida tem aproximadamente 171 e 229 vezes menos parâmetros.
- A Tabela 1 tem uma inconsistência na camada 24 → 300. Usamos a contagem
  pelas dimensões. A soma dos valores publicados seria 670.024.
- O objetivo é testar o método completo e validar uma previsão física,
  sem prometer a precisão ou cobertura de frequências dos resultados originais.
- Reduzir o escopo melhora o controle do prazo, mas não garante que todas as
  geometrias produzam notches úteis. Essa viabilidade depende das simulações.

## Referências

1. Ghorbani et al., [artigo original](https://doi.org/10.1038/s41598-021-86588-2),
   Tabelas 1, 2 e 4 e Figuras 4 e 5. PDF preservado em `paper/`.
2. NVIDIA, [especificação da Tesla K80](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-product-literature/nvidia-tesla-k80-overview.pdf).
