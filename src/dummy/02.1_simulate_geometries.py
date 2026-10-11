"""Save deterministic synthetic spectra."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Execute this stage and save its results under the scenario run directory."""
    from metasurface.reduced.dataset import simulate_geometries
    print("SYNTHETIC SOFTWARE DEMONSTRATION: no electromagnetic solver is used.", flush=True)
    from metasurface.common.run_records import write_simulation_metrics
    with timed_part("simulation"):
        checkpoint = simulate_geometries(config.DUMMY_ROOT / "geometries" / "geometry_checkpoint.npz",
                            config.DUMMY_ROOT / "data", config.PHYSICS, backend="synthetic",
                            limit=config.GENERATION_LIMIT)

    with timed_part("simulation_metrics_export"):
        write_simulation_metrics(checkpoint)

if __name__ == "__main__":
    run_stage(config.DUMMY_ROOT, "02.1_simulate_geometries", _run)
