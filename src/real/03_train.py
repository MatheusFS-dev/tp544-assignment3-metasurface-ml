"""Prepare physical spectra and train the reduced model."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Execute this stage and save its results under the scenario run directory."""
    from metasurface.common.network_analysis import save_analysis
    from metasurface.common.run_records import write_dataset_metrics, write_training_metrics
    from metasurface.reduced.dataset import prepare_dataset
    from metasurface.reduced.learning import train

    with timed_part("network_analysis"):
        save_analysis(config.ROOT / "runs" / "real" / "analysis")
    with timed_part("dataset_preparation"):
        dataset_path = prepare_dataset(config.DATA_DIR / "simulation_checkpoint.npz",
                                       config.DATA_DIR / "training_dataset.npz")
    with timed_part("dataset_metrics_export"):
        write_dataset_metrics(dataset_path)
    with timed_part("training"):
        checkpoint = train(dataset_path, config.TRAIN_DIR, config.TRAINING)
    with timed_part("training_metrics_export"):
        write_training_metrics(checkpoint)


if __name__ == "__main__":
    run_stage(config.ROOT / "runs" / "real", "03_train", _run)
