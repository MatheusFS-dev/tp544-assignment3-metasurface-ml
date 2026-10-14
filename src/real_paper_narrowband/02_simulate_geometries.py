"""Simulate paper-scale narrowband layouts with Lumerical."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Collect 22--28 GHz spectra using reference-only preflight."""
    from metasurface.common.run_records import write_simulation_metrics
    from metasurface.paper.dataset import simulate
    with timed_part("simulation"):
        checkpoint = simulate(
            config.PAPER_NARROWBAND_ROOT / "geometries" / "geometry_checkpoint.npz",
            config.PAPER_NARROWBAND_ROOT / "data", config.NARROWBAND_PHYSICS,
            config.PAPER_NARROWBAND_GENERATION_LIMIT,
            preflight_mode=config.PAPER_NARROWBAND_PREFLIGHT_MODE)
    with timed_part("simulation_metrics_export"):
        write_simulation_metrics(checkpoint)


if __name__ == "__main__":
    run_stage(config.PAPER_NARROWBAND_ROOT, "02_simulate_geometries", _run)
