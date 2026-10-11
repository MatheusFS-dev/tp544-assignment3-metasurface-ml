"""Architecture-derived costs, with explicit correction of paper Table 1."""

import json
from pathlib import Path

from storage import atomic_write


def dense_cost(widths):
    """Count dense weights, biases and forward multiply-accumulate operations.

    Args:
        widths (list[int]): Input width followed by each dense output width.
    Returns:
        dict: Per-layer costs, parameter total and forward MACs per sample.
    Raises:
        ValueError: If widths contain fewer than two positive integers.
    Notes:
        MACs omit activations, loss, backward propagation and optimizer work.
        FP32 weight size excludes activations, gradients and Adam states.
    Examples:
        dense_cost([8, 32, 64, 16])['parameters'] == 3440
    """
    if len(widths) < 2 or any(not isinstance(n, int) or isinstance(n, bool) or n < 1 for n in widths):
        raise ValueError("Dense widths must be positive integers.")
    layers = [{"input": a, "output": b, "parameters": (a + 1) * b,
               "forward_macs_per_sample": a * b} for a, b in zip(widths[:-1], widths[1:])]
    total = sum(row["parameters"] for row in layers)
    return {"widths": widths, "layers": layers, "parameters": total,
            "forward_macs_per_sample": sum(row["forward_macs_per_sample"] for row in layers),
            "fp32_weights_bytes": 4 * total}


def analyze_networks():
    """Compare both original networks and the reduced implementation exactly.

    Args:
        None.
    Returns:
        dict: Analytical counts, paper table sum and a correction explanation.
    Raises:
        None.
    Notes:
        Dropout/ReLU/sigmoid add no trainable parameters. Paper Table 1 reports
        90,300 for 24 -> 300, inconsistent with the stated layer dimensions.
        Runtime cannot be inferred from these counts alone.
    """
    result = {"paper_unrestricted": dense_cost([24, 24, 300, 300, 300, 300, 1024]),
              "paper_restricted": dense_cost([24, 24, 500, 500, 500, 500, 48]),
              "reduced": dense_cost([8, 32, 64, 16])}
    result["paper_unrestricted"].update(
        published_parameter_sum=600 + 4 * 90300 + 308224,
        table_correction="Table 1 reports 90,300 parameters for 24 -> 300. Dense dimensions imply 7,500.")
    for name in ("paper_unrestricted", "paper_restricted"):
        result[name]["parameter_ratio_to_reduced"] = result[name]["parameters"] / result["reduced"]["parameters"]
    result["assumptions"] = [
        "Counts follow the published dense widths, including biases.",
        "MAC counts describe one forward pass, not full training cost.",
        "Actual training speed requires a benchmark on the execution workstation.",
        "The paper trained on 2,000 sampled layouts, not every possible combination."]
    return result


def save_analysis(directory):
    """Save a machine-readable report and a concise Portuguese layer analysis.

    Args:
        directory (Path-like): Report destination directory.
    Returns:
        dict: Counts also written to network_analysis.json.
    Raises:
        OSError: If report files cannot be written.
    """
    result = analyze_networks()
    rows = []
    labels = {"paper_unrestricted": "Artigo, irrestrita", "paper_restricted": "Artigo, restrita", "reduced": "Reduzida"}
    for name, label in labels.items():
        r = result[name]
        rows.append(f"| {label} | {' → '.join(map(str, r['widths']))} | {r['parameters']:,} | {r['forward_macs_per_sample']:,} | {r['fp32_weights_bytes'] / 1e6:.4f} |")
    text = """# Análise das redes

- Contagem pelas dimensões das camadas, incluindo pesos e biases.
- Dense: parâmetros = (entradas + 1) × saídas. Dropout e ativações não adicionam parâmetros.

| Rede | Dimensões | Parâmetros | MACs por amostra, forward | Pesos FP32, MB decimais |
|---|---|---:|---:|---:|
""" + "\n".join(rows) + """

## Contagem por camada

| Rede | Camada | Entradas → saídas | Parâmetros | MACs |
|---|---:|---|---:|---:|
"""
    for name, label in labels.items():
        for i, row in enumerate(result[name]["layers"], 1):
            text += f"| {label} | {i} | {row['input']} → {row['output']} | {row['parameters']:,} | {row['forward_macs_per_sample']:,} |\n"
    text += """
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
"""
    directory = Path(directory)
    for filename, payload in [("network_analysis.json", json.dumps(result, indent=2)), ("ANALISE_REDES.md", text)]:
        encoded = payload.encode("utf-8")
        atomic_write(directory / filename, lambda stream, content=encoded: stream.write(content))
    print("Network parameters: " + ", ".join(f"{name}={r['parameters']}" for name, r in result.items() if isinstance(r, dict)))
    return result
