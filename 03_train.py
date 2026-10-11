"""Prepare descriptors from saved spectra, then train or resume PyTorch offline."""

import config
from dataset import prepare_dataset
from learning import train
from network_analysis import save_analysis

if __name__ == "__main__":
    save_analysis(config.ROOT / "docs")
    dataset_path = prepare_dataset(config.DATA_DIR / "simulation_checkpoint.npz",
                                   config.DATA_DIR / "training_dataset.npz")
    train(dataset_path, config.TRAIN_DIR, config.TRAINING)
