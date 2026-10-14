"""Prepare paper-scale narrowband spectra and train offline."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Prepare the narrowband dataset and train the restricted network."""
    from metasurface.common.run_records import write_dataset_metrics, write_training_metrics
    from metasurface.paper.dataset import prepare
    from metasurface.paper.learning import train
    data_dir = config.PAPER_NARROWBAND_ROOT / "data"
    with timed_part("dataset_preparation"):
        dataset = prepare(data_dir / "simulation_checkpoint.npz",
                          data_dir / "training_dataset.npz")
    with timed_part("dataset_metrics_export"):
        write_dataset_metrics(dataset)
    with timed_part("training"):
        checkpoint = train(dataset, config.PAPER_NARROWBAND_ROOT / "training",
                           config.PAPER_NARROWBAND_TRAINING)
    with timed_part("training_metrics_export"):
        write_training_metrics(checkpoint)


if __name__ == "__main__":
    run_stage(config.PAPER_NARROWBAND_ROOT, "03_train", _run)
