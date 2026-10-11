"""Evaluate and export one reduced physical design."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Execute this stage and save its results under the scenario run directory."""
    from metasurface.reduced.evaluation import evaluate
    with timed_part("evaluation_export"):
        evaluate(config.TRAIN_DIR / "last.pt", config.DATA_DIR / "training_dataset.npz",
                 config.EXPORT_DIR, config.TARGET_NOTCHES, config.RUN_EXPORTED_DESIGN)


if __name__ == "__main__":
    run_stage(config.ROOT / "runs" / "real", "04_test_and_export", _run)
