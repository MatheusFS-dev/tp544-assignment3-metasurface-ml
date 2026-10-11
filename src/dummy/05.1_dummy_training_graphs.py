"""Draw dummy training history from saved metrics."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Write synthetic training graphs inside the dummy run directory."""
    from metasurface.common.training_graphs import generate

    with timed_part("training_graphs"):
        generate(config.DUMMY_ROOT / "training" / "training_metrics.json",
                 config.DUMMY_ROOT / "graphs")


if __name__ == "__main__":
    run_stage(config.DUMMY_ROOT, "05.1_dummy_training_graphs", _run)
