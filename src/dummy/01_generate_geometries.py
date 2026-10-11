"""Generate the independent dummy geometry manifest."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Execute this stage and save its results under the scenario run directory."""
    from metasurface.reduced.geometry_manifest import generate_geometries
    from metasurface.common.run_records import write_geometry_metrics
    with timed_part("geometry_generation"):
        checkpoint = generate_geometries(config.DUMMY_ROOT / "geometries", config.PHYSICS["pixel_m"])

    with timed_part("geometry_metrics_export"):
        write_geometry_metrics(checkpoint)

if __name__ == "__main__":
    run_stage(config.DUMMY_ROOT, "01_generate_geometries", _run)
