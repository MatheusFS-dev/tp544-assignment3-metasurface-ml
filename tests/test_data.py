"""Regression tests for geometry, descriptors, and durable generation."""

import importlib
import importlib.util
import json

import numpy as np
import pytest


def module(name):
    """Load a component and fail explicitly when its implementation is absent.

    Args:
        name (str): Top-level module name.
    Returns:
        module: Imported implementation.
    Raises:
        AssertionError: If the component has not been implemented.
    """
    assert importlib.util.find_spec(name) is not None, f"Missing implementation: {name}"
    return importlib.import_module(name)


def test_all_256_distinct_symmetric_layouts():
    """All 256 distinct symmetric layouts.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    g = module("metasurface.reduced.geometry")
    codes = g.all_codes()
    assert codes.shape == (256, 4)
    assert len({g.pixel_layout(c).tobytes() for c in codes}) == 256
    for c in codes:
        p = g.pixel_layout(c)
        assert np.array_equal(p, p[::-1])
        assert np.array_equal(p, p[:, ::-1])


@pytest.mark.parametrize("codes", [[-1, 0, 0, 0], [4, 0, 0, 0], [0, 0, 0], [1.5, 0, 0, 0]])
def test_invalid_codes(codes):
    """Invalid codes.

    Args:
        codes (object): Pytest fixture or parametrized input.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    with pytest.raises(ValueError):
        module("metasurface.reduced.geometry").pixel_layout(codes)


def test_ring_areas_and_rectangle_area():
    """Ring areas and rectangle area.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    g = module("metasurface.reduced.geometry")
    expected = [0, 12, 20, 32]
    for i, area in enumerate(expected):
        p = g.pixel_layout([i] * 4)
        assert p.sum() == 16 * area
        rects = g.rectangles([i] * 4, 0.2e-3)
        assert sum(r[2] * r[3] for r in rects) == pytest.approx(p.sum() * (0.2e-3)**2)


def test_notch_width_uses_threshold_crossings():
    """Notch width uses threshold crossings.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    s = module("metasurface.reduced.spectra")
    f = np.linspace(4e9, 45e9, 4101)
    db = -20 * np.maximum(1 - abs(f / 1e9 - 20), 0)
    feat, reason = s.extract_features(f, 10**(db / 10))
    assert reason == "ok"
    assert feat[:4] == pytest.approx([20, -20, 1, 1])
    assert np.array_equal(feat[4:], np.zeros(4))


@pytest.mark.parametrize("case,reason", [("flat", "no_notches"), ("boundary", "truncated_notch"), ("three", "too_many_notches")])
def test_ineligible_spectra(case, reason):
    """Ineligible spectra.

    Args:
        case (object): Pytest fixture or parametrized input.
        reason (object): Pytest fixture or parametrized input.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    s = module("metasurface.reduced.spectra")
    f = np.linspace(4e9, 45e9, 4101)
    db = np.zeros_like(f)
    centers = {"flat": [], "boundary": [4], "three": [10, 20, 30]}[case]
    for c in centers:
        db -= 20 * np.maximum(1 - abs(f / 1e9 - c), 0)
    _, actual = s.extract_features(f, 10**(db / 10))
    assert actual == reason


def test_nonfinite_spectrum_rejected():
    """Nonfinite spectrum rejected.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    s = module("metasurface.reduced.spectra")
    with pytest.raises(ValueError):
        s.extract_features(np.linspace(4e9, 45e9, 100), np.full(100, np.nan))


def test_generation_resume_skips_completed(tmp_path):
    """Generation resume skips completed.

    Args:
        tmp_path (object): Pytest fixture or parametrized input.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    d = module("metasurface.reduced.dataset")
    physics = module("metasurface.config").PHYSICS.copy()
    p = d.generate(tmp_path, physics, "synthetic", limit=3)
    state = d.load_dataset(p)
    assert state["completed"].sum() == 3
    d.generate(tmp_path, physics, "synthetic", limit=2)
    state = d.load_dataset(p)
    assert state["completed"].sum() == 5
    assert json.loads(state["metadata"].item())["backend"] == "synthetic"


def test_generation_crash_restarts_only_uncommitted(tmp_path):
    """Generation crash restarts only uncommitted.

    Args:
        tmp_path (object): Pytest fixture or parametrized input.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    d = module("metasurface.reduced.dataset")
    physics = module("metasurface.config").PHYSICS.copy()
    seen = []

    def interrupted(codes, frequencies):
        """Emulate a solver failure after two successful results.

        Args:
            codes (ndarray): Pattern choices.
            frequencies (ndarray): Requested frequency vector.
        Returns:
            dict: Synthetic power spectra.
        Raises:
            RuntimeError: At the injected crash.
        """
        seen.append(tuple(codes))
        if len(seen) == 3:
            raise RuntimeError("Injected solver crash")
        return module("metasurface.reduced.synthetic").response(codes, frequencies)

    with pytest.raises(RuntimeError, match="Injected"):
        d.generate(tmp_path, physics, "synthetic", simulator=interrupted)
    p = tmp_path / "dataset_checkpoint.npz"
    assert d.load_dataset(p)["completed"].sum() == 2
    d.generate(tmp_path, physics, "synthetic", limit=1)
    assert d.load_dataset(p)["completed"].sum() == 3


def test_changed_generation_config_rejected(tmp_path):
    """Changed generation config rejected.

    Args:
        tmp_path (object): Pytest fixture or parametrized input.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    d = module("metasurface.reduced.dataset")
    physics = module("metasurface.config").PHYSICS.copy()
    d.generate(tmp_path, physics, "synthetic", limit=1)
    physics["substrate_thickness_m"] *= 2
    with pytest.raises(ValueError, match="configuration"):
        d.generate(tmp_path, physics, "synthetic", limit=1)


def test_tampered_dataset_rejected(tmp_path):
    """Tampered dataset rejected.

    Args:
        tmp_path (object): Pytest fixture or parametrized input.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    d = module("metasurface.reduced.dataset")
    p = d.generate(tmp_path, module("metasurface.config").PHYSICS.copy(), "synthetic", limit=3)
    data = d.load_dataset(p)
    data["codes"][0] = [3, 3, 3, 3]
    np.savez_compressed(p, **data)
    with pytest.raises(ValueError):
        d.load_dataset(p)


def test_corrupt_checkpoint_recovers_previous_snapshot(tmp_path):
    """Corrupt checkpoint recovers previous snapshot.

    Args:
        tmp_path (object): Pytest fixture or parametrized input.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    d = module("metasurface.reduced.dataset")
    p = d.generate(tmp_path, module("metasurface.config").PHYSICS.copy(), "synthetic", limit=3)
    p.write_bytes(b"interrupted file")
    assert d.load_dataset(p)["completed"].sum() == 2
