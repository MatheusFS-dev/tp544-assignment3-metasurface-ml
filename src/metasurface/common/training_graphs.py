"""Plot committed training history without evaluating the model again."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metasurface.common.storage import atomic_write

plt.rcParams["font.family"] = "DejaVu Serif"


def _save_plot(destination, epochs, series, ylabel):
    """Draw available scalar series against completed epoch numbers.

    Args:
        destination (Path): PNG destination.
        epochs (list[dict]): Saved epoch records.
        series (list[tuple[str, str]]): Metric keys and legend labels.
        ylabel (str): Vertical axis label.

    Returns:
        bool: Whether at least one complete series was drawn.
    """
    fig, axis = plt.subplots(figsize=(7, 4))
    fig.patch.set_facecolor("white")
    axis.set_facecolor("white")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_linewidth(0.8)
    axis.spines["bottom"].set_linewidth(0.8)
    axis.grid(axis="y", color="0.86", linestyle="--", linewidth=0.6)
    axis.set_axisbelow(True)
    drawn = False
    colors = ("#355C7D", "#A36A3B", "#5B8C70", "#9A5A69", "#756A8B", "#687C80")
    for index, (key, label) in enumerate(series):
        points = [(row["epoch"], row[key]) for row in epochs if key in row and row[key] is not None]
        if points:
            axis.plot([point[0] for point in points], [point[1] for point in points],
                      linewidth=1.6, color=colors[index % len(colors)], label=label)
            drawn = True
    if drawn:
        axis.set(xlabel="Epoch", ylabel=ylabel)
        if len(axis.lines) > 1:
            axis.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(destination, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return drawn


def generate(metrics_path, output_dir):
    """Save loss, accuracy, and compute-time graphs from recorded metrics.

    Args:
        metrics_path (Path-like): Stage 03 training_metrics.json.
        output_dir (Path-like): Stage 05 graph directory.

    Returns:
        list[Path]: Saved graph and data files.

    Raises:
        ValueError: If no completed epochs or malformed records are present.
    """
    metrics = json.loads(Path(metrics_path).read_text())
    epochs = metrics["epochs"]
    if not epochs:
        raise ValueError("No completed epoch is available for plotting.")
    if any("epoch" not in row or "train_loss" not in row for row in epochs):
        raise ValueError("Training history lacks epoch or loss values.")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    saved = []
    plots = [
        ("training_loss.png", [("train_loss", "Train minibatch mean"),
                               ("train_eval_loss", "Train evaluation"),
                               ("test_loss", "Held-out synthetic test"),
                               ("loss", "Validation")], "Loss"),
        ("training_accuracy.png", [("train_position_accuracy", "Train position"),
                                   ("position_accuracy", "Validation position"),
                                   ("train_exact_accuracy", "Train exact layout"),
                                   ("exact_accuracy", "Validation exact layout"),
                                   ("train_bit_accuracy", "Train bit"),
                                   ("train_tile_accuracy", "Train tile"),
                                   ("test_bit_accuracy", "Held-out bit"),
                                   ("test_tile_accuracy", "Held-out tile"),
                                   ("test_exact_accuracy", "Held-out exact layout")], "Accuracy (fraction)"),
    ]
    for filename, series, ylabel in plots:
        path = destination / filename
        if _save_plot(path, epochs, series, ylabel):
            saved.append(path)
    seconds = {}
    for batch in metrics.get("minibatches", []):
        epoch = batch["epoch"]
        seconds[epoch] = seconds.get(epoch, 0.0) + batch["compute_seconds"]
    timing = [{"epoch": epoch, "seconds": value} for epoch, value in sorted(seconds.items())]
    if timing:
        path = destination / "training_time.png"
        _save_plot(path, timing, [("seconds", "Minibatch compute")], "Compute time (s)")
        saved.append(path)
    summary = destination / "graph_data.json"
    atomic_write(summary, lambda stream: stream.write(json.dumps(
        {"source": str(metrics_path), "epochs": epochs, "compute_seconds_by_epoch": timing},
        indent=2).encode("utf-8")))
    saved.append(summary)
    for path in saved:
        print(f"Training graph: saved {path}", flush=True)
    return saved
