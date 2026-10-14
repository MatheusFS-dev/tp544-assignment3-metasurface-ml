"""Generate 2,000 paper-scale narrowband sampled layouts."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Generate and summarize the paper-scale narrowband manifest."""
    from metasurface.common.run_records import write_geometry_metrics
    from metasurface.paper.manifest import generate
    with timed_part("geometry_generation"):
        checkpoint = generate(config.PAPER_NARROWBAND_ROOT / "geometries",
                              config.PAPER_NARROWBAND_COUNT,
                              config.PAPER_NARROWBAND_GENERATION_LIMIT)
    with timed_part("geometry_metrics_export"):
        write_geometry_metrics(checkpoint)


if __name__ == "__main__":
    run_stage(config.PAPER_NARROWBAND_ROOT, "01_generate_geometries", _run)
