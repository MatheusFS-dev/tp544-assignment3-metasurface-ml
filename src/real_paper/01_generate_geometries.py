"""Generate 2,000 paper-scale sampled layouts."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Execute this stage and save its results under the scenario run directory."""
    from metasurface.paper.manifest import generate
    from metasurface.common.run_records import write_geometry_metrics
    with timed_part("geometry_generation"):
        checkpoint = generate(config.PAPER_ROOT / "geometries", config.PAPER_COUNT,
                 config.PAPER_GENERATION_LIMIT)

    with timed_part("geometry_metrics_export"):
        write_geometry_metrics(checkpoint)

if __name__ == "__main__":
    run_stage(config.PAPER_ROOT, "01_generate_geometries", _run)
