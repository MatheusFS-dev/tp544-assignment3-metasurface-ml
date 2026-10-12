"""Prepare and train the restricted network on synthetic surrogate data."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Prepare the dataset and run resumable CUDA training."""
    from metasurface.common.run_records import write_dataset_metrics, write_training_metrics
    from metasurface.paper.learning import train
    from metasurface.synthetic_paper.dataset import prepare
    print("SYNTHETIC SURROGATE TRAINING: metrics are not physical results.", flush=True)
    data_dir = config.SYNTHETIC_PAPER_ROOT / "data"
    with timed_part("dataset_preparation"):
        dataset = prepare(data_dir / "simulation_checkpoint.npz",
                          data_dir / "training_dataset.npz")
    with timed_part("dataset_metrics_export"):
        write_dataset_metrics(dataset)
    with timed_part("training"):
        checkpoint = train(dataset, config.SYNTHETIC_PAPER_ROOT / "training",
                           config.SYNTHETIC_PAPER_TRAINING)
    with timed_part("training_metrics_export"):
        write_training_metrics(checkpoint)


if __name__ == "__main__":
    run_stage(config.SYNTHETIC_PAPER_ROOT, "03_train", _run)

