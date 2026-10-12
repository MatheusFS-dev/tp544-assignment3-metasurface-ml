# Justificativa técnica para a não conclusão da reprodução integral

## Síntese

O prazo de uma semana não é compatível com uma reprodução integral de Ghorbani et al. (2021). O principal limite não é o treinamento da rede neural, mas a construção e a validação do conjunto de dados eletromagnéticos. Cada amostra exige uma simulação FDTD licenciada, e os resultados ainda precisam passar por verificações de malha, tempo de simulação, material, condições de contorno e convergência.

O trabalho realizado deve, portanto, ser apresentado como **implementação independente e reprodução reduzida do método**, e não como réplica exata dos resultados publicados.

## Escala do problema

| Item | Artigo original | Reprodução reduzida | Consequência prática |
|---|---:|---:|---|
| Padrões disponíveis por posição | 8 | 4 | A redução simplifica a família geométrica |
| Posições independentes | 16 | 4, com espelhamento | Reduz drasticamente o número de combinações |
| Espaço de busca restrito | \(8^{16} = 281.474.976.710.656\) | \(4^4 = 256\) | Nem o espaço original pode ser enumerado, nem 2.000 amostras o cobrem exaustivamente |
| Geometrias simuladas | 2.000 amostras aleatórias | Até 256 candidatas | Cada candidata ainda exige uma simulação física |
| Faixa de frequência | 4 a 45 GHz | 4 a 45 GHz | A banda larga aumenta o custo numérico |
| Pontos de frequência | Não informado com precisão suficiente para réplica | 1.641 | A resolução espectral precisa ser mantida para localizar e medir os notches |
| Simulador | CST Microwave Studio | Lumerical FDTD | A troca de solver impede equivalência numérica direta |

Pontos importantes:

- As 2.000 geometrias do artigo são uma amostra de aproximadamente **281 trilhões** de layouts possíveis na família restrita.
- O espaço original é cerca de **1,10 trilhão de vezes maior** que o conjunto reduzido de 256 layouts.
- Reduzir para 256 casos torna o experimento menor, mas não elimina o custo de resolver as equações eletromagnéticas para cada geometria.
- Nem toda simulação produz uma resposta utilizável. Casos sem notch, com excesso de notches ou com notches truncados podem ser preservados, mas não necessariamente servem para treinamento.

## Custo de simulação

O pré-teste executado no Lumerical produziu uma estimativa de aproximadamente **10 horas** apenas para a etapa preliminar de verificação das geometrias e da configuração numérica. Esse valor é uma estimativa obtida no ambiente deste projeto, não a duração medida da campanha completa nem um tempo reportado pelo artigo.

Mesmo desconsiderando preparação, falhas e repetições, o custo cresce linearmente com o número de geometrias:

| Tempo médio hipotético por geometria | 256 geometrias | 2.000 geometrias |
|---:|---:|---:|
| 10 min | 42,7 h, ou 1,8 dia contínuo | 333,3 h, ou 13,9 dias contínuos |
| 30 min | 128 h, ou 5,3 dias contínuos | 1.000 h, ou 41,7 dias contínuos |
| 60 min | 256 h, ou 10,7 dias contínuos | 2.000 h, ou 83,3 dias contínuos |

Essas linhas são **cenários ilustrativos**, não medições deste conjunto completo. O tempo real também inclui:

- preflight do solver e validação da referência vazia e do espelho PEC.
- ajuste e verificação do material FR4.
- refinamento de malha, tempo de simulação e critério de desligamento.
- repetição de pilotos para demonstrar convergência numérica.
- tratamento de erros da API, licenças, reinicializações e retomada de checkpoints.
- nova simulação da geometria prevista pela rede.
- análise dos espectros e extração consistente de frequência, profundidade e largura dos notches.

Uma execução contínua também não equivale a uma semana de trabalho disponível. Desenvolvimento, inspeção dos resultados e correção de configurações ocorrem entre as rodadas e podem exigir que parte das simulações seja repetida.

## Informações insuficientes para uma réplica exata

| Informação necessária | Situação no artigo | Impacto na reprodução |
|---|---|---|
| Dataset com as 2.000 geometrias e espectros | Não foi localizado um link público no artigo | Todo o conjunto precisa ser gerado novamente |
| Código e checkpoints treinados | Não disponibilizados no material consultado | A implementação e o treinamento precisam ser refeitos |
| Máscaras binárias exatas dos oito padrões de 8 × 8 pixels | As figuras mostram a família, mas não definem todas as matrizes | Qualquer reconstrução contém aproximações geométricas |
| Critério exato da largura do notch | Não especificado suficientemente | Diferentes regras produzem descritores de entrada diferentes |
| Probabilidade de dropout | Não informada | A arquitetura não pode ser repetida exatamente |
| Configuração completa do projeto CST | Não disponibilizada | Malha, portas, limites, materiais e critérios do solver precisam ser reinterpretados |
| Semente e divisão exata dos dados | Não estabelecidas de forma reprodutível | Acurácia e curvas de treinamento não podem ser esperadas como idênticas |
| Definição operacional completa da acurácia | Insuficiente para comparação inequívoca | Não é válido prometer reproduzir os valores de aproximadamente 91% e 92% |

Há ainda uma inconsistência interna que precisa ser registrada: a Tabela 1 atribui 90.300 parâmetros à camada densa de 24 para 300 neurônios, mas suas dimensões implicam 7.500 parâmetros. O total calculado pelas dimensões é 587.224, enquanto a soma dos valores publicados é 670.024. Isso reforça que números do artigo não podem ser transcritos automaticamente como especificação executável.

## Por que uma semana é insuficiente

Em sete dias seria necessário, ao mesmo tempo:

- interpretar detalhes geométricos e numéricos ausentes.
- implementar e depurar a automação pela API do Lumerical.
- validar a física antes de coletar dados.
- executar centenas ou milhares de simulações.
- verificar convergência e repetir casos problemáticos.
- preparar, filtrar e dividir o dataset sem vazamento de dados.
- treinar, ajustar e avaliar a rede.
- revalidar fisicamente ao menos uma previsão.
- documentar diferenças, limitações e resultados.

O prazo pode comportar a implementação do fluxo, testes sintéticos claramente identificados e alguns pilotos físicos. Ele não comporta, com rigor, a geração e validação de todo o conjunto eletromagnético nem a reprodução dos resultados quantitativos do artigo.

## Conclusão e escopo defensável

A conclusão tecnicamente defensável é:

- **não foi possível concluir a reprodução integral no prazo de uma semana**.
- o gargalo dominante é a geração e a validação dos dados por FDTD, não somente o código da rede neural.
- a ausência de artefatos e parâmetros essenciais impede uma réplica exata mesmo com mais capacidade computacional.
- o fluxo reduzido de 256 candidatas é uma mudança metodológica explícita e não deve ser confundido com o experimento original.
- uma reprodução física completa e auditável exigiria, de forma realista, **várias semanas e potencialmente alguns meses**, dependendo do tempo por simulação, da disponibilidade da licença e da quantidade de refinamentos e repetições.
- resultados sintéticos podem validar o software, mas não podem ser apresentados como resultados eletromagnéticos reproduzidos.

Assim, o produto adequado para o prazo é uma prova de implementação com escopo reduzido, acompanhada dos testes e pilotos efetivamente concluídos e desta declaração de limitações. A obtenção de métricas físicas comparáveis às do artigo deve permanecer como trabalho futuro condicionado à conclusão das simulações.

## Fontes consultadas no repositório

- Ghorbani et al., *Deep neural network-based automatic metasurface design with a wide frequency range*, Scientific Reports 11, 7102 (2021), DOI: <https://doi.org/10.1038/s41598-021-86588-2>.
- `paper/s41598-021-86588-2.pdf` e `paper/paper.txt`.
- `.agents/prompts/assignment_context.md`.
- `JUSTIFICATIVA_REDUCAO.md`.
- `docs/REAL_PAPER.md`.

