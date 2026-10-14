"""Evaluate and export one reduced narrowband design."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Evaluate held-out data and optionally validate the exported design."""
    from metasurface.reduced.evaluation import evaluate
    with timed_part("evaluation_export"):
        evaluate(config.NARROWBAND_TRAIN_DIR / "last.pt",
                 config.NARROWBAND_DATA_DIR / "training_dataset.npz",
                 config.NARROWBAND_EXPORT_DIR, config.NARROWBAND_TARGET_NOTCHES,
                 config.NARROWBAND_RUN_EXPORTED_DESIGN,
                 preflight_mode=config.NARROWBAND_PREFLIGHT_MODE)


if __name__ == "__main__":
    run_stage(config.NARROWBAND_ROOT, "04_test_and_export", _run)
