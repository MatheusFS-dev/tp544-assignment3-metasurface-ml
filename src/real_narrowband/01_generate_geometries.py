"""Generate the reduced narrowband geometry manifest."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Generate and summarize all 256 narrowband geometry candidates."""
    from metasurface.common.run_records import write_geometry_metrics
    from metasurface.reduced.geometry_manifest import generate_geometries
    with timed_part("geometry_generation"):
        checkpoint = generate_geometries(config.NARROWBAND_GEOMETRY_DIR,
                                         config.NARROWBAND_PHYSICS["pixel_m"])
    with timed_part("geometry_metrics_export"):
        write_geometry_metrics(checkpoint)


if __name__ == "__main__":
    run_stage(config.NARROWBAND_ROOT, "01_generate_geometries", _run)
