"""Generate deterministic nonphysical paper-scale spectra."""

from metasurface import config
from metasurface.common.run_records import run_stage, timed_part


def _run():
    """Generate spectra, prepare completed data, and write diagnostics."""
    from metasurface.common.run_records import write_simulation_metrics
    from metasurface.synthetic_paper.dataset import generate_spectra, load_raw, prepare
    from metasurface.synthetic_paper.diagnostics import write_diagnostics
    from metasurface.synthetic_paper.surrogate import SurrogateConfig
    print("SYNTHETIC SURROGATE: no electromagnetic solver is used.", flush=True)
    data_dir = config.SYNTHETIC_PAPER_ROOT / "data"
    with timed_part("synthetic_spectrum_generation"):
        checkpoint = generate_spectra(
            config.SYNTHETIC_PAPER_ROOT / "geometries" / "geometry_checkpoint.npz",
            data_dir, config.PHYSICS, SurrogateConfig(**config.SYNTHETIC_PAPER_SURROGATE),
            config.SYNTHETIC_PAPER_GENERATION_LIMIT)
    with timed_part("simulation_metrics_export"):
        write_simulation_metrics(checkpoint)
    if load_raw(checkpoint)["completed"].all():
        with timed_part("synthetic_diagnostics"):
            dataset = prepare(checkpoint, data_dir / "training_dataset.npz")
            write_diagnostics(dataset, config.SYNTHETIC_PAPER_ROOT / "analysis")


if __name__ == "__main__":
    run_stage(config.SYNTHETIC_PAPER_ROOT, "02_generate_synthetic_spectra", _run)

