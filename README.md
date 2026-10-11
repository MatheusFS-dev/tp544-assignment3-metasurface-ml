# tp544-assignment3-metasurface-ml

Adaptação reduzida do método de Ghorbani et al.,
[Scientific Reports, 2021](https://doi.org/10.1038/s41598-021-86588-2).
Aprende a prever uma geometria a partir de notches de reflexão e exporta uma
previsão para verificação no Lumerical.

[JUSTIFICATIVA_REDUCAO.md](JUSTIFICATIVA_REDUCAO.md) contém a justificativa curta
com tabelas. [AGENTS.md](AGENTS.md) detalha o cenário. O PDF e texto pesquisável
estão em `paper/`. O exercício está em `.agents/prompts/assignment_context.md`.

## Fluxo

| Etapa | Comando | Entrada → saída | Lumerical? |
|---|---|---|---|
| Geometrias | `python 01_generate_geometries.py` | Configuração → máscaras e códigos | Não |
| Simulação | `python 02_simulate_geometries.py` | Geometrias salvas → espectros brutos | Sim, via lumapi |
| Preparação e treino | `python 03_train.py` | Espectros → dataset preparado e modelo | Não |
| Teste e exportação | `python 04_test_and_export.py` | Modelo/dataset → métricas e um desenho | Apenas na validação física opcional |

A primeira etapa apenas desenha o cobre. A segunda calcula a resposta física.
O treino usa arquivos salvos, sem consultar o solver. Uma licença remota pode
exigir rede durante as simulações, mas o treino pode funcionar offline.

## 1. Instalar Python e conferir a GPU

Use Python 3.12. Na pasta do projeto:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

No Windows, ative com `.venv\Scripts\Activate.ps1` no PowerShell.
Instale PyTorch com CUDA pelo comando apropriado no
[instalador oficial](https://pytorch.org/get-started/locally/). Confira:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
python -m pip freeze > environment.lock.txt
```

Mantenha esse ambiente para retomar checkpoints. Mudanças de versão do PyTorch
invalidam o treino existente. Testes de software CPU usam `requirements-cpu.txt`.
Não há fallback automático de CUDA para CPU. GPU no PyTorch não habilita GPU FDTD.

## 2. Configurar

Edite `config.py`. Os scripts não recebem argumentos:

- `LUMAPI_DIR`: pasta que contém `lumapi.py` na instalação do Lumerical.
  Exemplos: `/opt/lumerical/v252/api/python` ou
  `Path(r"C:\Program Files\Lumerical\v252\api\python")`.
  Versões recentes podem usar `C:\Program Files\Ansys Inc\v261\Lumerical\api\python`.
  Confira sua instalação antes de usar um exemplo.
- `BACKEND = "lumerical"` para resultados físicos.
- `GENERATION_LIMIT = 4` para um piloto de quatro simulações adicionais.
  Depois use `None` para retomar todas as restantes. Esse limite não limita máscaras.
- `TRAINING["device"] = "cuda"` para treino na GPU NVIDIA.
- `RUN_EXPORTED_DESIGN = False` para exportar sem iniciar o solver.
- `GEOMETRY_DIR`, `DATA_DIR`, `TRAIN_DIR`, `EXPORT_DIR`: diretórios de saída.

O cenário tem 256 combinações espelhadas, pixel de 0,2 mm, célula de 6,4 mm,
FR4 de 1,5 mm, metal de 18 micrômetros, banda de 4–45 GHz, incidência normal
com polarização x. Os quatro padrões em `geometry.py` são uma adaptação.
Não são os oito padrões exatos do artigo.

## 3. Gerar e salvar geometrias

```bash
python 01_generate_geometries.py
```

Em `runs/geometries/`, `geometry_checkpoint.npz` salva todos os códigos A/B/C/D,
as máscaras 32 × 32, escala, checksum e flags. `geometry_codes.csv` lista os IDs.
Cada máscara é confirmada antes da próxima. Esta etapa não importa Torch nem
Lumerical. Repetir o comando retoma trabalho pendente e mantém o manifesto completo.

## 4. Simular geometrias salvas

Com Lumerical FDTD instalado e licença disponível:

```bash
python 02_simulate_geometries.py
```

O adaptador carrega `lumapi` da instalação e constrói materiais, substrato,
plano de terra, cobre, fonte, monitores e fronteiras para cada layout salvo.
As máscaras do manifesto são verificadas contra a construção por códigos.
Um manifesto ausente, incompleto ou incompatível é recusado.

Antes da coleta, verifica referência vazia, espelho PEC, cobertura da fonte,
ajuste do FR4, desligamento automático e refinamento de malha/tempo em um piloto.
Falhas interrompem a coleta. O cobre usa condutividade de 5,8e7 S/m.
Uma malha inicial pode não resolver adequadamente perdas por efeito pelicular.

`runs/data/simulation_checkpoint.npz` salva frequências, códigos, potências
co/cross/total, flags, tempos e proveniência após cada simulação.
Nenhum notch é extraído nesta etapa. `seconds` mede a chamada por layout,
incluindo construção e validação dos arrays, mas exclui preflight e escrita.

Inspecione o progresso sem abrir o solver:

```bash
python -c "import config; from dataset import load_simulations; d=load_simulations(config.DATA_DIR/'simulation_checkpoint.npz'); print('completed:',d['completed'].sum()); print('simulation seconds:',d['seconds'].sum())"
```

Meça o piloto antes de decidir coletar todas as 256 geometrias. Repetir o comando
pula resultados salvos. Uma falha repete apenas o layout não confirmado.

## 5. Preparar dataset e treinar offline

O Lumerical pode ser fechado:

```bash
python 03_train.py
```

O script extrai frequência, profundidade, largura e presença para até dois
notches e salva `runs/data/training_dataset.npz`. Os espectros brutos permanecem
intactos. A largura usa cruzamentos interpolados de -10 dB. Espectros sem notch,
com mais de dois, truncados ou não resolvidos ficam preservados com o motivo.
O mínimo de 24 exemplos elegíveis é operacional, não garantia de generalização.

A rede CUDA tem `8 → 32 → 64 → 16` e 3.440 parâmetros. As saídas formam quatro
decisões de quatro classes. Usa entropia cruzada, Adam, taxa 0,001, batch 16,
até 500 épocas e parada após 30 sem melhora. Divisão: 70% treino, 15% validação,
restante teste. O scaler usa somente treino. O melhor modelo usa validação.

`runs/training/last.pt` salva pesos atuais/melhores, Adam, divisão, scaler,
histórico, ordem/cursor dos batches e estados aleatórios CPU/CUDA após cada batch.
Reexecutar retoma o último batch confirmado. Igualdade exata na retomada foi
verificada em CPU. Não é prometida entre GPUs ou versões distintas.

A análise automática salva `docs/ANALISE_REDES.md` e `network_analysis.json`.
Para contar as redes sem Torch, CUDA ou solver:

```bash
python 00_analyze_networks.py
```

## 6. Avaliar e exportar um desenho

```bash
python 04_test_and_export.py
```

O alvo padrão é o primeiro exemplo de teste, sem escolher o melhor resultado.
Um alvo customizado usa `TARGET_NOTCHES = [(30.0, -20.0, 0.5)]`, em GHz, dB e GHz.
Aceita até dois notches. Alvos fora da faixa de treino são sinalizados.

| Arquivo em `runs/design/` | Uso |
|---|---|
| `report.json` | Acurácia por posição/layout, baseline, alvo, previsão e status físico |
| `design_pixels.csv` | Exatamente uma máscara 32 × 32, linhas em y crescente |
| `design_rectangles_m.csv` | Retângulos de cobre em metros e limites z |
| `design_spec.json` | Materiais, dimensões, camadas, fonte, malha e proveniência |
| `build_design.lsf` | Construção completa e salvamento do projeto Lumerical |
| `design_preview.png` | Desenho previsto |
| `training_loss.png` | Curvas de treino/validação |
| `spectrum_comparison.png` | Espectro em cache, quando disponível |

Uma comparação em cache não é uma simulação nova. Acurácia de classes não
comprova correspondência física. Arquivos físicos antigos são invalidados na exportação.

## 7. Verificar no Lumerical

1. Abra um projeto FDTD vazio. Configure o diretório de trabalho como a pasta
   absoluta `runs/design/` e rode `build_design.lsf` no Script File Editor.
   O script limpa objetos desse projeto e salva `design.fsp`.
2. Compare com a imagem. Confira período de 6,4 mm, camadas em `design_spec.json`,
   fonte propagando para -z, polarização x, fronteiras periódicas x/y e PML em z.
   A célula representa uma superfície periódica infinita, não um painel finito.
3. Rode a simulação. Confira desligamento automático e ajustes dos materiais.
4. Refine malha e tempo desse desenho. Convergência do piloto não valida todos
   os layouts. O cobre pode precisar de malha muito mais fina para resolver perdas.

Para comparação quantitativa consistente, configure `RUN_EXPORTED_DESIGN = True`
e repita `python 04_test_and_export.py`. Ele simula o mesmo único desenho,
salva `design.fsp` e `design_validation_spectrum.csv`, e adiciona erros de
frequência, profundidade e largura quando a quantidade de notches coincide.
Caso contrário, registra a divergência de quantidade.

No monitor `reflection`, `transmission("reflection")` dá o fluxo refletido total
positivo, pois a reflexão propaga para +z. Campos no monitor `incident` do
dispositivo contêm onda incidente e refletida. A normalização usa a referência
vazia, não a potência desse monitor no dispositivo. O notch co-polar usa
`10*log10(R_co)`. Absorção exige reflexão total, incluindo polarização cruzada,
e transmissão desprezível pelo plano de terra.

## 8. Recuperação e testes

- Após falha, repita o comando da etapa interrompida.
- Mantenha o arquivo principal e `.bak`. Escrita atômica e checksum protegem
  os snapshots. Use somente um escritor por diretório.
- Preparação é determinística e atômica. Uma falha repete a extração, mantendo
  os espectros brutos. Uma reexecução sem novos dados preserva a identidade.
- Novas simulações mudam o dataset. Use nova `TRAIN_DIR` para retreinar.
  Mudanças físicas exigem nova `DATA_DIR`. Mudança de escala exige novo manifesto.
- Não misture cobre, PEC e dados sintéticos. O helper legado `dataset.generate`
  permanece para compatibilidade dos testes anteriores, não é usado pelos novos scripts.
- Checkpoints do fluxo anterior não migram automaticamente. Preserve-os e use
  novas pastas para este fluxo.

```bash
python -m pytest -q
```

Os testes usam dados sintéticos e CPU explicitamente selecionados. Não comprovam
execução CUDA nem resultados físicos do Lumerical. Evidência desta revisão está
em `evidence/validation.md`. `evidence/software-proof/` contém um exemplo artificial.

## Referências

1. [Artigo original](https://doi.org/10.1038/s41598-021-86588-2).
2. Ansys, [instalação e uso de lumapi](https://optics.ansys.com/hc/en-us/articles/39744901602707-Installation-and-Getting-Started-Python-API).
3. Ansys, [formato de resultados Python](https://optics.ansys.com/hc/en-us/articles/39744236202771-Accessing-Simulation-Results-Python-API).
4. Ansys, [metodologia de simulação RF](https://optics.ansys.com/hc/en-us/articles/360042054154-Radio-frequency-simulation-methodology).
5. [Instalação oficial de PyTorch](https://pytorch.org/get-started/locally/).
