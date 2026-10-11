"""Training recovery, leakage, provenance, and exactly-one export tests."""

import json

import numpy as np
import pytest
import torch

from tests.test_data import module


def fixture_data(tmp_path):
    """Generate artificial data solely for exercising the software pipeline.

    Args:
        tmp_path (Path): Test directory.
    Returns:
        Path: Complete synthetic checkpoint.
    Raises:
        Exception: If generation fails.
    """
    return module("metasurface.reduced.dataset").generate(tmp_path / "data", module("metasurface.config").PHYSICS, "synthetic")


def options():
    """Provide short deterministic training settings for regression tests.

    Args:
        None.
    Returns:
        dict: CPU configuration with explicit synthetic permission.
    Raises:
        None.
    """
    return {**module("metasurface.config").TRAINING, "device": "cpu", "max_epochs": 4, "patience": 30, "allow_synthetic": True}


def test_cpu_resume_is_bitwise_equal_to_uninterrupted(tmp_path):
    """Cpu resume is bitwise equal to uninterrupted.

    Args:
        tmp_path (object): Pytest fixture or parametrized input.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    data = fixture_data(tmp_path)
    m = module("metasurface.reduced.learning")
    settings = options()
    full = m.train(data, tmp_path / "full", settings)
    m.train(data, tmp_path / "resumed", settings, max_steps=3)
    resumed = m.train(data, tmp_path / "resumed", settings)
    a, b = m.load_checkpoint(full), m.load_checkpoint(resumed)
    assert a["history"] == b["history"]
    assert a["step"] == b["step"]
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key])
    assert torch.equal(a["rng"], b["rng"])


def test_epoch_reports_train_and_validation_metrics(tmp_path, capsys):
    """Save and print both split accuracies after a completed epoch.

    Args:
        tmp_path (Path): Isolated test directory.
        capsys (CaptureFixture): Captured console output.
    """
    data = fixture_data(tmp_path)
    m = module("metasurface.reduced.learning")
    checkpoint = m.train(data, tmp_path / "train", {**options(), "max_epochs": 1})
    record = m.load_checkpoint(checkpoint)["history"][0]
    assert {"train_loss", "train_position_accuracy", "train_exact_accuracy",
            "loss", "position_accuracy", "exact_accuracy"} <= record.keys()
    output = capsys.readouterr().out
    assert "train_loss=" in output
    assert "val_loss=" in output
    assert "train_position_accuracy=" in output
    assert "val_position_accuracy=" in output


def test_synthetic_training_requires_explicit_permission(tmp_path):
    """Synthetic training requires explicit permission.

    Args:
        tmp_path (object): Pytest fixture or parametrized input.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    data = fixture_data(tmp_path)
    with pytest.raises(ValueError, match="synthetic"):
        module("metasurface.reduced.learning").train(data, tmp_path / "train", {**options(), "allow_synthetic": False})


def test_changed_training_options_rejected(tmp_path):
    """Changed training options rejected.

    Args:
        tmp_path (object): Pytest fixture or parametrized input.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    data = fixture_data(tmp_path)
    m = module("metasurface.reduced.learning")
    m.train(data, tmp_path / "train", options(), max_steps=2)
    with pytest.raises(ValueError, match="configuration"):
        m.train(data, tmp_path / "train", {**options(), "learning_rate": 0.01})


def test_split_disjoint_and_scaler_fitted_only_to_training(tmp_path):
    """Split disjoint and scaler fitted only to training.

    Args:
        tmp_path (object): Pytest fixture or parametrized input.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    data = fixture_data(tmp_path)
    m = module("metasurface.reduced.learning")
    path = m.train(data, tmp_path / "train", options(), max_steps=1)
    state = m.load_checkpoint(path)
    train, val, test = [set(state["splits"][k]) for k in ("train", "val", "test")]
    assert not train & val and not train & test and not val & test
    rows = module("metasurface.reduced.dataset").load_dataset(data)["features"][state["splits"]["train"]]
    assert np.allclose(state["mean"].numpy(), rows.mean(axis=0))


def test_corrupt_training_checkpoint_recovers_and_finishes(tmp_path):
    """Corrupt training checkpoint recovers and finishes.

    Args:
        tmp_path (object): Pytest fixture or parametrized input.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    data = fixture_data(tmp_path)
    m = module("metasurface.reduced.learning")
    path = m.train(data, tmp_path / "train", options(), max_steps=3)
    path.write_bytes(b"crash")
    result = m.train(data, tmp_path / "train", options())
    assert m.load_checkpoint(result)["finished"]


def test_evaluation_exports_exactly_one_nn_geometry(tmp_path):
    """Evaluation exports exactly one nn geometry.

    Args:
        tmp_path (object): Pytest fixture or parametrized input.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    data = fixture_data(tmp_path)
    m = module("metasurface.reduced.learning")
    checkpoint = m.train(data, tmp_path / "train", options())
    (tmp_path / "export").mkdir()
    for name in ("design.fsp", "design_validation_spectrum.csv", "report.json"):
        (tmp_path / "export" / name).write_text("stale")
    report = module("metasurface.reduced.evaluation").evaluate(checkpoint, data, tmp_path / "export")
    assert len(list((tmp_path / "export").glob("*_pixels.csv"))) == 1
    assert report["backend"] == "synthetic"
    assert report["physical_validation"] == "not_run"
    layout = np.loadtxt(tmp_path / "export" / "design_pixels.csv", delimiter=",")
    assert layout.shape == (32, 32)
    assert np.array_equal(layout, module("metasurface.reduced.geometry").pixel_layout(report["predicted_codes"]))
    project = (tmp_path / "export" / "build_design.lsf").read_text()
    assert 'save("design.fsp")' in project
    info = json.loads((tmp_path / "export" / "design_spec.json").read_text())
    assert info["physics"] == module("metasurface.config").PHYSICS
    assert info["pattern_codes"] == report["predicted_codes"]
    assert 0 <= report["test"]["position_accuracy"] <= 1
    assert "nearest_neighbor" in report
    assert not (tmp_path / "export" / "design.fsp").exists()
    assert not (tmp_path / "export" / "design_validation_spectrum.csv").exists()


def test_invalid_custom_target_rejected(tmp_path):
    """Invalid custom target rejected.

    Args:
        tmp_path (object): Pytest fixture or parametrized input.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    data = fixture_data(tmp_path)
    checkpoint = module("metasurface.reduced.learning").train(data, tmp_path / "train", options())
    with pytest.raises(ValueError):
        module("metasurface.reduced.evaluation").evaluate(checkpoint, data, tmp_path / "export", target=[(50, -20, 0.5)])


def test_dataset_change_invalidates_training_resume(tmp_path):
    """Dataset change invalidates training resume.

    Args:
        tmp_path (object): Pytest fixture or parametrized input.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    d = module("metasurface.reduced.dataset")
    m = module("metasurface.reduced.learning")
    physics = module("metasurface.config").PHYSICS
    data = d.generate(tmp_path / "data", physics, "synthetic", limit=100)
    m.train(data, tmp_path / "train", options(), max_steps=1)
    d.generate(tmp_path / "data", physics, "synthetic", limit=1)
    with pytest.raises(ValueError, match="dataset"):
        m.train(data, tmp_path / "train", options())
