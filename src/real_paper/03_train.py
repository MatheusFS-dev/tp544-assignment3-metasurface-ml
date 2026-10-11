"""Prepare paper-scale spectra and train the restricted model."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Execute this stage and save its results under the scenario run directory."""
    from metasurface.common.run_records import write_dataset_metrics, write_training_metrics
    from metasurface.paper.dataset import prepare
    from metasurface.paper.learning import train

    data_dir = config.PAPER_ROOT / "data"
    with timed_part("dataset_preparation"):
        dataset_path = prepare(data_dir / "simulation_checkpoint.npz",
                               data_dir / "training_dataset.npz")
    with timed_part("dataset_metrics_export"):
        write_dataset_metrics(dataset_path)
    with timed_part("training"):
        checkpoint = train(dataset_path, config.PAPER_ROOT / "training", config.PAPER_TRAINING)
    with timed_part("training_metrics_export"):
        write_training_metrics(checkpoint)


if __name__ == "__main__":
    run_stage(config.PAPER_ROOT, "03_train", _run)
