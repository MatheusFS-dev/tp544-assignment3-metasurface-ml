"""Draw paper-scale narrowband training history."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Write training graphs in the paper narrowband run directory."""
    from metasurface.common.training_graphs import generate
    with timed_part("training_graphs"):
        generate(config.PAPER_NARROWBAND_ROOT / "training" / "training_metrics.json",
                 config.PAPER_NARROWBAND_ROOT / "graphs")


if __name__ == "__main__":
    run_stage(config.PAPER_NARROWBAND_ROOT, "05_training_graphs", _run)
