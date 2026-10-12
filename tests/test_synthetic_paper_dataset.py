"""Synthetic paper dataset generation and provenance contracts."""

import json

import numpy as np
import pytest

from metasurface import config
from metasurface.common.storage import load_npz
from metasurface.paper.manifest import generate
from metasurface.synthetic_paper.dataset import generate_spectra, prepare
from metasurface.synthetic_paper.diagnostics import write_diagnostics


def test_small_generation_is_resumable_and_solver_free(tmp_path, monkeypatch):
    """Save every row and preserve committed spectra on resume."""
    manifest = generate(tmp_path / "geometries", count=8)
    path = generate_spectra(manifest, tmp_path / "data", config.PHYSICS, limit=2)
    partial = load_npz(path)
    saved = partial["co"][:2].copy()
    assert partial["completed"].sum() == 2
    assert json.loads(partial["metadata"].item())["backend"] == "synthetic_surrogate"

    import metasurface.synthetic_paper.dataset as module
    original = module.response
    calls = []

    def counted(codes, frequencies, surrogate_config=None):
        calls.append(tuple(codes))
        return original(codes, frequencies, surrogate_config)

    monkeypatch.setattr(module, "response", counted)
    generate_spectra(manifest, tmp_path / "data", config.PHYSICS)
    complete = load_npz(path)
    assert len(calls) == 6
    assert complete["completed"].all()
    assert np.array_equal(saved, complete["co"][:2])
    assert complete["co"].shape == (8, 1641)
    assert complete["seconds"].shape == (8,)


def test_changed_identity_is_rejected(tmp_path):
    """Refuse to mix changed physics with a saved synthetic dataset."""
    manifest = generate(tmp_path / "geometries", count=2)
    generate_spectra(manifest, tmp_path / "data", config.PHYSICS, limit=1)
    changed = {**config.PHYSICS, "frequency_points": 101}
    with pytest.raises(ValueError, match="identity"):
        generate_spectra(manifest, tmp_path / "data", changed)


def test_preparation_and_diagnostics_preserve_synthetic_provenance(tmp_path):
    """Prepare features and write truthful diagnostic summaries."""
    manifest = generate(tmp_path / "geometries", count=32)
    raw = generate_spectra(manifest, tmp_path / "data", config.PHYSICS)
    prepared = prepare(raw, tmp_path / "data/training_dataset.npz")
    report_path = write_diagnostics(prepared, tmp_path / "analysis")
    state = load_npz(prepared)
    report = json.loads(report_path.read_text())

    assert state["features"].shape == (32, 24)
    assert json.loads(state["metadata"].item())["backend"] == "synthetic_surrogate"
    assert report["scenario"] == "synthetic_surrogate"
    assert report["completed"] == 32
    assert report["eligible"] >= 24
    assert report["passivity_violations"] == 0
    assert (tmp_path / "analysis/synthetic_dataset_summary.csv").exists()


def test_training_requires_explicit_synthetic_permission(tmp_path):
    """Require an affirmative option before training on surrogate data."""
    from metasurface.paper.learning import train

    manifest = generate(tmp_path / "geometries", count=32)
    raw = generate_spectra(manifest, tmp_path / "data", config.PHYSICS)
    prepared = prepare(raw, tmp_path / "data/training_dataset.npz")
    options = {**config.PAPER_TRAINING, "device": "cpu", "max_epochs": 1,
               "allow_test_fixture": False, "allow_synthetic_surrogate": False}
    with pytest.raises(ValueError, match="synthetic surrogate"):
        train(prepared, tmp_path / "blocked", options, max_steps=1)

    options["allow_synthetic_surrogate"] = True
    checkpoint = train(prepared, tmp_path / "allowed", options)
    from metasurface.reduced.learning import load_checkpoint
    record = load_checkpoint(checkpoint)["history"][0]
    assert {"test_loss", "test_bit_accuracy", "test_tile_accuracy",
            "test_exact_accuracy"} <= record.keys()
