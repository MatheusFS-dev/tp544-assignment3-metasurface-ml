"""Simulate reduced layouts with Lumerical."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Execute this stage and save its results under the scenario run directory."""
    from metasurface.reduced.dataset import simulate_geometries
    from metasurface.common.run_records import write_simulation_metrics
    with timed_part("simulation"):
        checkpoint = simulate_geometries(config.GEOMETRY_DIR / "geometry_checkpoint.npz", config.DATA_DIR,
                            config.PHYSICS, "lumerical", config.GENERATION_LIMIT,
                            preflight_mode=config.PREFLIGHT_MODE)

    with timed_part("simulation_metrics_export"):
        write_simulation_metrics(checkpoint)

if __name__ == "__main__":
    run_stage(config.ROOT / "runs" / "real", "02_simulate_geometries", _run)
