"""Draw reduced physical training history from saved metrics."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Write training graphs inside the real run directory."""
    from metasurface.common.training_graphs import generate

    with timed_part("training_graphs"):
        generate(config.TRAIN_DIR / "training_metrics.json",
                 config.ROOT / "runs" / "real" / "graphs")


if __name__ == "__main__":
    run_stage(config.ROOT / "runs" / "real", "05_training_graphs", _run)
