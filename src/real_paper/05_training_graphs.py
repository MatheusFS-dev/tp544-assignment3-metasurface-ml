"""Draw paper-scale training history from saved metrics."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Write training graphs inside the paper-scale run directory."""
    from metasurface.common.training_graphs import generate

    with timed_part("training_graphs"):
        generate(config.PAPER_ROOT / "training" / "training_metrics.json",
                 config.PAPER_ROOT / "graphs")


if __name__ == "__main__":
    run_stage(config.PAPER_ROOT, "05_training_graphs", _run)
