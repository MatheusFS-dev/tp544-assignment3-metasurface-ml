"""Training graph generation from saved epoch records."""

import json

import pytest

from metasurface.common.training_graphs import generate


@pytest.mark.parametrize("history,expected", [
    ([{"epoch": 1, "train_loss": 1.1, "loss": 1.2,
       "train_position_accuracy": 0.3, "position_accuracy": 0.2,
       "train_exact_accuracy": 0.1, "exact_accuracy": 0.0}],
     {"training_loss.png", "training_accuracy.png", "training_time.png"}),
    ([{"epoch": 1, "train_loss": 0.2, "train_bit_accuracy": 0.6,
       "train_tile_accuracy": 0.3, "train_exact_accuracy": 0.1}],
     {"training_loss.png", "training_accuracy.png", "training_time.png"}),
])
def test_generate_training_graphs(tmp_path, history, expected):
    """Plot only recorded metrics and save a summary of the plotted data.

    Args:
        tmp_path (Path): Temporary output directory.
        history (list[dict]): Synthetic epoch metrics.
        expected (set[str]): Required figure filenames.
    """
    metrics = tmp_path / "training_metrics.json"
    metrics.write_text(json.dumps({"epochs": history, "minibatches": [
        {"epoch": 1, "compute_seconds": 0.5}]}))
    destination = tmp_path / "graphs"
    generate(metrics, destination)
    assert expected <= {path.name for path in destination.iterdir()}
    assert all((destination / name).stat().st_size > 0 for name in expected)
    summary = json.loads((destination / "graph_data.json").read_text())
    assert summary["epochs"] == history
    assert summary["compute_seconds_by_epoch"] == [{"epoch": 1, "seconds": 0.5}]


def test_generate_requires_completed_epochs(tmp_path):
    """Reject an unfinished training record with no epoch data.

    Args:
        tmp_path (Path): Temporary output directory.
    """
    metrics = tmp_path / "training_metrics.json"
    metrics.write_text(json.dumps({"epochs": [], "minibatches": []}))
    with pytest.raises(ValueError, match="completed epoch"):
        generate(metrics, tmp_path / "graphs")
