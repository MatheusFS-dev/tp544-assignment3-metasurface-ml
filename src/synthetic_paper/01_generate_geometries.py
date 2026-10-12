"""Generate the independent 2,000-layout synthetic-paper manifest."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Generate or resume the synthetic-paper geometry manifest."""
    from metasurface.paper.manifest import generate
    from metasurface.common.run_records import write_geometry_metrics
    with timed_part("geometry_generation"):
        checkpoint = generate(config.SYNTHETIC_PAPER_ROOT / "geometries",
                              config.SYNTHETIC_PAPER_COUNT,
                              config.SYNTHETIC_PAPER_GENERATION_LIMIT)
    with timed_part("geometry_metrics_export"):
        write_geometry_metrics(checkpoint)


if __name__ == "__main__":
    run_stage(config.SYNTHETIC_PAPER_ROOT, "01_generate_geometries", _run)

