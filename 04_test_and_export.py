"""Evaluate held-out samples and export exactly one predicted Lumerical design."""

import config
from evaluation import evaluate

if __name__ == "__main__":
    evaluate(config.TRAIN_DIR / "last.pt", config.DATA_DIR / "training_dataset.npz", config.EXPORT_DIR,
             config.TARGET_NOTCHES, config.RUN_EXPORTED_DESIGN)
