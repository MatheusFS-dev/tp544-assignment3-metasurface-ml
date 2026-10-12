"""Draw explicitly synthetic paper-scale training history."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Write graphs from saved synthetic training metrics."""
    from metasurface.common.training_graphs import generate
    with timed_part("training_graphs"):
        generate(config.SYNTHETIC_PAPER_ROOT / "training" / "training_metrics.json",
                 config.SYNTHETIC_PAPER_ROOT / "graphs")


if __name__ == "__main__":
    run_stage(config.SYNTHETIC_PAPER_ROOT, "05_training_graphs", _run)

