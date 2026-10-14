"""Count network parameters for the reduced narrowband workflow."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Save network analysis in the narrowband run directory."""
    from metasurface.common.network_analysis import save_analysis
    with timed_part("network_analysis"):
        save_analysis(config.NARROWBAND_ROOT / "analysis")


if __name__ == "__main__":
    run_stage(config.NARROWBAND_ROOT, "00_analyze_networks", _run)
