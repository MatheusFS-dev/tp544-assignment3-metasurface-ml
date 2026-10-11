"""Count network parameters and save analysis."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Execute this stage and save its results under the scenario run directory."""
    from metasurface.common.network_analysis import save_analysis
    with timed_part("network_analysis"):
        save_analysis(config.ROOT / "runs" / "real" / "analysis")


if __name__ == "__main__":
    run_stage(config.ROOT / "runs" / "real", "00_analyze_networks", _run)
