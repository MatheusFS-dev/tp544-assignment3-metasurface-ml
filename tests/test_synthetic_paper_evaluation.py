"""Evaluation, export, and optional validation contracts."""

import json

import numpy as np
import pytest

from metasurface import config
from metasurface.paper.learning import train
from metasurface.paper.manifest import generate
from metasurface.synthetic_paper.dataset import generate_spectra, prepare
from metasurface.synthetic_paper.evaluation import evaluate
from metasurface.synthetic_paper.surrogate import response
from metasurface.synthetic_paper.validation import validate_export


def _trained_fixture(tmp_path):
    manifest = generate(tmp_path / "geometries", count=32)
    raw = generate_spectra(manifest, tmp_path / "data", config.PHYSICS)
    dataset = prepare(raw, tmp_path / "data/training_dataset.npz")
    options = {**config.PAPER_TRAINING, "device": "cpu", "max_epochs": 1,
               "allow_synthetic_surrogate": True}
    checkpoint = train(dataset, tmp_path / "training", options)
    return checkpoint, dataset


def test_evaluation_exports_synthetic_and_lumerical_artifacts(tmp_path):
    """Export metrics, spectra, geometry, checksums, and construction script."""
    checkpoint, dataset = _trained_fixture(tmp_path)
    output = tmp_path / "design"
    report = evaluate(checkpoint, dataset, output)

    assert report["scenario"] == "synthetic_surrogate"
    assert report["physical_validation"] == "not_run"
    assert np.asarray(report["confusion_matrix"]).shape == (8, 8)
    for name in ("report.json", "design_spec.json", "design_pixels.csv",
                 "design_rectangles_m.csv", "design_preview.png", "build_design.lsf",
                 "target_synthetic_spectrum.csv", "predicted_synthetic_spectrum.csv"):
        assert (output / name).exists()
    spec = json.loads((output / "design_spec.json").read_text())
    assert spec["backend"] == "synthetic_surrogate"
    assert len(spec["pattern_codes"]) == 16
    assert len(spec["predicted_features"]) == 24
    assert any(spec["predicted_features"])
    assert set(spec["artifact_checksums"]) == {"design_pixels.csv", "design_rectangles_m.csv"}


def test_validation_uses_exact_export_and_fake_solver(tmp_path):
    """Validate an unchanged export without loading a real solver."""
    checkpoint, dataset = _trained_fixture(tmp_path)
    design = tmp_path / "design"
    evaluate(checkpoint, dataset, design)

    class FakeSolver:
        def __init__(self, physics):
            self.physics = physics

        def preflight(self, directory):
            return None

        def _run(self, codes, reference="device", project_path=None):
            project_path.write_text("fake project")
            return {"codes": np.asarray(codes)}

        def _powers(self, raw):
            frequencies = np.linspace(self.physics["frequency_min_hz"],
                                      self.physics["frequency_max_hz"],
                                      self.physics["frequency_points"])
            return response(raw["codes"], frequencies)

    result = validate_export(design, solver_factory=FakeSolver)
    assert result["physical_validation"] == "fresh_lumerical_run"
    assert (design / "design.fsp").exists()
    assert (design / "design_validation_spectrum.csv").exists()
    assert (design / "validation_report.json").exists()
    assert result["physical_errors"]["target_count"] == sum(
        1 for row in np.asarray(json.loads((design / "design_spec.json").read_text())
                                ["predicted_features"]).reshape(8, 3) if row[0] > 0)

    with (design / "design_pixels.csv").open("a") as stream:
        stream.write("0\n")
    with pytest.raises(ValueError, match="checksum"):
        validate_export(design, solver_factory=FakeSolver)
