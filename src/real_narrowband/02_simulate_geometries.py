"""Simulate reduced narrowband layouts with Lumerical."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Collect 22--28 GHz spectra using reference-only preflight."""
    from metasurface.common.run_records import write_simulation_metrics
    from metasurface.reduced.dataset import simulate_geometries
    with timed_part("simulation"):
        checkpoint = simulate_geometries(
            config.NARROWBAND_GEOMETRY_DIR / "geometry_checkpoint.npz",
            config.NARROWBAND_DATA_DIR, config.NARROWBAND_PHYSICS, "lumerical",
            config.NARROWBAND_GENERATION_LIMIT,
            preflight_mode=config.NARROWBAND_PREFLIGHT_MODE)
    with timed_part("simulation_metrics_export"):
        write_simulation_metrics(checkpoint)


if __name__ == "__main__":
    run_stage(config.NARROWBAND_ROOT, "02_simulate_geometries", _run)
