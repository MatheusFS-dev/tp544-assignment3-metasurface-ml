"""Train the reduced model from synthetic spectra."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Execute this stage and save its results under the scenario run directory."""
    import torch
    from metasurface.common.network_analysis import save_analysis
    from metasurface.common.run_records import write_dataset_metrics, write_training_metrics
    from metasurface.reduced.dataset import prepare_dataset
    from metasurface.reduced.learning import train

    data_dir = config.DUMMY_ROOT / "data"
    with timed_part("dataset_preparation"):
        dataset_path = prepare_dataset(data_dir / "simulation_checkpoint.npz",
                                       data_dir / "training_dataset.npz")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"SYNTHETIC SOFTWARE DEMONSTRATION: training on {device}.", flush=True)
    with timed_part("network_analysis"):
        save_analysis(config.DUMMY_ROOT / "analysis")
    options = {**config.TRAINING, "device": device, "allow_synthetic": True}
    with timed_part("dataset_metrics_export"):
        write_dataset_metrics(dataset_path)
    with timed_part("training"):
        checkpoint = train(dataset_path, config.DUMMY_ROOT / "training", options)
    with timed_part("training_metrics_export"):
        write_training_metrics(checkpoint)


if __name__ == "__main__":
    run_stage(config.DUMMY_ROOT, "03.1_dummy_train", _run)
