"""Evaluate and export one synthetic design."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Execute this stage and save its results under the scenario run directory."""
    from metasurface.reduced.evaluation import evaluate
    print("SYNTHETIC SOFTWARE DEMONSTRATION: exported design is not physically validated.", flush=True)
    with timed_part("evaluation_export"):
        evaluate(config.DUMMY_ROOT / "training" / "last.pt",
                 config.DUMMY_ROOT / "data" / "training_dataset.npz",
                 config.DUMMY_ROOT / "design", config.TARGET_NOTCHES, run_lumerical=False)


if __name__ == "__main__":
    run_stage(config.DUMMY_ROOT, "04.1_dummy_test_and_export", _run)
