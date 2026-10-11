# Verificação desta revisão

Data: 2026-10-11. Ambiente em `runtime.json`.

| Verificação | Resultado | Evidência |
|---|---|---|
| Baseline anterior, sem testes novos | 40 passaram | `baseline-current.log` |
| Testes de separação antes da implementação | 9 falhas por funcionalidades ausentes | `separated-red.log` |
| Testes novos depois da implementação | 9 passaram | `separated-green.log` |
| Suite completa desta revisão | 49 passaram | `pytest-separated-full.log` |
| Quatro entrypoints com fixtures sintéticos CPU | Passaram | `entrypoints-separated.log` |
| Retomada de geometrias e simulações | Preserva resultados confirmados | Testes de separação |
| Treino interrompido após 3 batches | Mesmos pesos e histórico da execução contínua CPU | Script de prova |
| Preparação repetida, sem solver | Bytes/id estáveis, espectros brutos intactos | Script de prova |
| Exportação | Exatamente um desenho, marcado como sintético | `software-proof/` |
| Rede reduzida | 3.440 parâmetros, verificados no PyTorch | Teste de análise |
| PDF original | SHA-256 idêntico ao anexo fornecido | `runtime.json` |
| Suite após extração do ZIP | 49 passaram em 120,94 s | `zip-tests.log` |
| Revisão independente | Sem achados críticos/importantes, 49 testes passaram | `review-separated.md` |
| Dependências | `pip check`: sem requisitos quebrados | Ambiente de prova |

O teste de contagem inicialmente tinha uma soma esperada incorreta. A contagem
independente com camadas `torch.nn.Linear` confirmou 587.224 para a arquitetura
irrestrita e 670.024 para a soma dos valores publicados. O teste e a prosa
foram corrigidos. A discrepância do artigo em 24 → 300 permanece documentada.

Para repetir a prova dos entrypoints:

```bash
python evidence/prove_separated_pipeline.py
```

Esse script modifica configurações apenas na memória, usa pastas temporárias e
escreve um exemplo artificial em `software-proof/`. Não modifica `config.py`,
não executa lumapi e não comprova o funcionamento de CUDA ou a física FDTD.

Logs anteriores (`entrypoints.log`, `pytest-final.log`, `review-*.log`,
`export-green.log`) são históricos da versão de três scripts. A tabela acima
é a evidência do novo fluxo separado.

Não executados: CUDA real, Lumerical licenciado, convergência da previsão final,
reprodução da precisão do artigo, fabricação ou medições. O padrão entregue
continua solicitando CUDA e dados reais, sem substituir silenciosamente esses
recursos por CPU ou dados artificiais.
