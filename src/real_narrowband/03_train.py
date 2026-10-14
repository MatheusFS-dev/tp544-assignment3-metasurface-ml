"""Prepare reduced narrowband spectra and train offline."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Prepare the narrowband dataset and train the reduced network."""
    from metasurface.common.network_analysis import save_analysis
    from metasurface.common.run_records import write_dataset_metrics, write_training_metrics
    from metasurface.reduced.dataset import prepare_dataset
    from metasurface.reduced.learning import train
    with timed_part("network_analysis"):
        save_analysis(config.NARROWBAND_ROOT / "analysis")
    with timed_part("dataset_preparation"):
        dataset = prepare_dataset(
            config.NARROWBAND_DATA_DIR / "simulation_checkpoint.npz",
            config.NARROWBAND_DATA_DIR / "training_dataset.npz")
    with timed_part("dataset_metrics_export"):
        write_dataset_metrics(dataset)
    with timed_part("training"):
        checkpoint = train(dataset, config.NARROWBAND_TRAIN_DIR,
                           config.NARROWBAND_TRAINING)
    with timed_part("training_metrics_export"):
        write_training_metrics(checkpoint)


if __name__ == "__main__":
    run_stage(config.NARROWBAND_ROOT, "03_train", _run)
