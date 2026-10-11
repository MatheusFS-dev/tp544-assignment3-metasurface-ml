"""Simulate paper-scale layouts with Lumerical."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Execute this stage and save its results under the scenario run directory."""
    from metasurface.paper.dataset import simulate
    from metasurface.common.run_records import write_simulation_metrics
    with timed_part("simulation"):
        checkpoint = simulate(config.PAPER_ROOT / "geometries" / "geometry_checkpoint.npz",
                 config.PAPER_ROOT / "data", config.PHYSICS, config.PAPER_GENERATION_LIMIT)

    with timed_part("simulation_metrics_export"):
        write_simulation_metrics(checkpoint)

if __name__ == "__main__":
    run_stage(config.PAPER_ROOT, "02_simulate_geometries", _run)
