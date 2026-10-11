"""Evaluate and export one paper-scale design."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Execute this stage and save its results under the scenario run directory."""
    from metasurface.paper.evaluation import evaluate
    with timed_part("evaluation_export"):
        evaluate(config.PAPER_ROOT / "training" / "last.pt",
                 config.PAPER_ROOT / "data" / "training_dataset.npz",
                 config.PAPER_ROOT / "design", config.PAPER_RUN_EXPORTED_DESIGN)


if __name__ == "__main__":
    run_stage(config.PAPER_ROOT, "04_test_and_export", _run)
