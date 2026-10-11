"""Contracts for the separate paper-scale restricted workflow."""

import numpy as np
import pytest
import torch

from metasurface.paper.geometry import masks, pixel_layout, sample_codes
from metasurface.paper.learning import RestrictedNet


def test_paper_tiles_and_sample_plan():
    tiles = masks()
    assert tiles.shape == (8, 8, 8)
    assert len({tile.tobytes() for tile in tiles}) == 8
    codes = sample_codes(2000, seed=2026)
    assert codes.shape == (2000, 16)
    assert len({tuple(row) for row in codes}) == 2000
    assert pixel_layout(codes[0]).shape == (32, 32)


def test_restricted_network_shapes_and_parameters():
    model = RestrictedNet()
    assert model(torch.zeros(2, 24)).shape == (2, 48)
    assert sum(p.numel() for p in model.parameters()) == 788648


def test_paper_fixture_pipeline_is_separate_and_labeled(tmp_path, capsys):
    """Exercise paper-scale checkpoints without claiming a physical result.

    Args:
        tmp_path (Path): Isolated test directory.
    """
    import json

    from metasurface import config
    from metasurface.paper.dataset import prepare, simulate
    from metasurface.paper.evaluation import evaluate
    from metasurface.paper.features import extract_features
    from metasurface.paper.learning import train
    from metasurface.paper.manifest import generate
    from metasurface.reduced.learning import load_checkpoint

    manifest = generate(tmp_path / "geometries", count=32)

    def artificial_response(codes, frequencies):
        """Return a labeled, resolved artificial notch for a fixture.

        Args:
            codes (ndarray): Tile choices.
            frequencies (ndarray): Frequency samples in Hz.

        Returns:
            dict: Invented reflected powers.
        """
        center = 18 + int(codes[0])
        ghz = frequencies / 1e9
        db = -20 * np.maximum(1 - abs(ghz - center) / 1.5, 0)
        power = 10 ** (db / 10)
        return {"co": power, "cross": np.zeros_like(power), "total": power.copy()}

    raw = simulate(manifest, tmp_path / "data", config.PHYSICS, simulator=artificial_response)
    prepared = prepare(raw, tmp_path / "data/training_dataset.npz")
    features, reason = extract_features(np.linspace(4e9, 45e9, 1641),
                                        artificial_response(np.zeros(16), np.linspace(4e9, 45e9, 1641))["co"])
    assert reason == "ok" and features.shape == (24,)
    options = {**config.PAPER_TRAINING, "device": "cpu", "max_epochs": 1,
               "allow_test_fixture": True}
    checkpoint = train(prepared, tmp_path / "training", options, max_steps=1)
    assert load_checkpoint(checkpoint)["step"] == 1
    assert len(load_checkpoint(checkpoint)["batch_history"]) == 1
    train(prepared, tmp_path / "training", options)
    full = load_checkpoint(train(prepared, tmp_path / "full", options))
    resumed = load_checkpoint(checkpoint)
    assert {"train_loss", "train_bit_accuracy", "train_tile_accuracy",
            "train_exact_accuracy"} <= resumed["history"][0].keys()
    output = capsys.readouterr().out
    assert "train_loss=" in output
    assert "train_bit_accuracy=" in output
    assert resumed["history"] == full["history"]
    assert all(torch.equal(resumed["model"][name], full["model"][name])
               for name in full["model"])
    report = evaluate(checkpoint, prepared, tmp_path / "design")
    with pytest.raises(ValueError, match="physically validated"):
        evaluate(checkpoint, prepared, tmp_path / "blocked", run_lumerical=True)
    assert report["backend"] == "test_fixture"
    assert report["physical_validation"] == "not_run"
    assert report["scenario"] == "paper_scale_approximation"
    assert json.loads((tmp_path / "design/design_spec.json").read_text())["backend"] == "test_fixture"
