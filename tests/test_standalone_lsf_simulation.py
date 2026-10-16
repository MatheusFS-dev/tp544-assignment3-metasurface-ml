"""Static contracts for the self-contained FDTD GUI simulation scripts."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_physical_workflows_have_standalone_gpu_lsf_jobs():
    """Each physical folder owns one direct geometry-and-simulation LSF job."""
    expectations = {
        "real": ("1641", "256", "reduced"),
        "real_narrowband": ("301", "256", "reduced"),
        "real_paper": ("1641", "2000", "paper"),
        "real_paper_narrowband": ("301", "2000", "paper"),
    }
    for folder, (points, count, family) in expectations.items():
        path = ROOT / "src" / folder / "02_simulate_in_fdtd_gui.lsf"
        text = path.read_text()
        assert f"frequency_points = {points};" in text
        assert f"sample_count = {count};" in text
        assert f'geometry_family = "{family}";' in text
        assert 'runsystemcheck("FDTD","GPU")' in text
        assert 'run("FDTD","GPU")' in text
        assert "pixel_mask = matrix(32,32);" in text
        assert "sample_idx = 1:sample_count" in text
        assert "matlabsavelegacy" in text
        assert "readdata(" not in text
        assert "lumapi" not in text
