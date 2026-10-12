"""Evaluate synthetic held-out data and export one predicted design."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Write synthetic metrics and solver-ready geometry artifacts."""
    from metasurface.synthetic_paper.evaluation import evaluate
    with timed_part("synthetic_evaluation_and_export"):
        evaluate(config.SYNTHETIC_PAPER_ROOT / "training" / "last.pt",
                 config.SYNTHETIC_PAPER_ROOT / "data" / "training_dataset.npz",
                 config.SYNTHETIC_PAPER_ROOT / "design")


if __name__ == "__main__":
    run_stage(config.SYNTHETIC_PAPER_ROOT, "04_test_and_export", _run)

