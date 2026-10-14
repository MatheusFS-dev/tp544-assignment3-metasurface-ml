"""Evaluate and export one paper-scale narrowband design."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Evaluate held-out data and optionally validate the exported design."""
    from metasurface.paper.evaluation import evaluate
    with timed_part("evaluation_export"):
        evaluate(config.PAPER_NARROWBAND_ROOT / "training" / "last.pt",
                 config.PAPER_NARROWBAND_ROOT / "data" / "training_dataset.npz",
                 config.PAPER_NARROWBAND_ROOT / "design",
                 config.PAPER_NARROWBAND_RUN_EXPORTED_DESIGN,
                 preflight_mode=config.PAPER_NARROWBAND_PREFLIGHT_MODE)


if __name__ == "__main__":
    run_stage(config.PAPER_NARROWBAND_ROOT, "04_test_and_export", _run)
