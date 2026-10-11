"""Behaviour checks for the explicitly separated offline and solver stages."""

import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest

from tests.test_data import module


def manifest(tmp_path):
    """Create the approved geometry manifest for stage tests.

    Args:
        tmp_path (Path): Isolated test directory.
    Returns:
        Path: Complete geometry manifest.
    Raises:
        AssertionError: If implementation is absent.
    """
    return module("metasurface.reduced.geometry_manifest").generate_geometries(tmp_path / "geometries", 0.2e-3)


def test_geometry_stage_saves_exact_masks_and_stable_identity(tmp_path):
    """Require saved masks, all codes and idempotent repeated generation.

    Args:
        tmp_path (Path): Isolated directory.
    Returns:
        None.
    Raises:
        AssertionError: If the persisted layout contract differs.
    """
    m = module("metasurface.reduced.geometry_manifest")
    path = manifest(tmp_path)
    state = m.load_geometries(path)
    original = path.read_bytes()
    assert state["completed"].all()
    assert state["pixels"].shape == (256, 32, 32)
    for i in (0, 85, 170, 255):
        assert np.array_equal(state["pixels"][i], module("metasurface.reduced.geometry").pixel_layout(state["codes"][i]))
    assert (path.parent / "geometry_codes.csv").exists()
    m.generate_geometries(path.parent, 0.2e-3)
    assert path.read_bytes() == original


def test_geometry_crash_resumes_and_preserves_saved_rows(tmp_path, monkeypatch):
    """Require only uncommitted geometry work to repeat after interruption.

    Args:
        tmp_path (Path): Isolated directory.
        monkeypatch (object): Pytest patch manager.
    Returns:
        None.
    Raises:
        AssertionError: If resume changes committed masks or skips work.
    """
    m = module("metasurface.reduced.geometry_manifest")
    original = m.save_npz
    calls = []

    def interrupted(path, arrays):
        """Inject a crash before the third row can be committed.

        Args:
            path (Path): Checkpoint destination.
            arrays (dict): Snapshot arrays.
        Returns:
            None.
        Raises:
            OSError: On the injected commit failure.
        """
        calls.append(int(arrays["completed"].sum()))
        if len(calls) == 4:
            raise OSError("injected geometry crash")
        original(path, arrays)

    monkeypatch.setattr(m, "save_npz", interrupted)
    with pytest.raises(OSError, match="injected"):
        manifest(tmp_path)
    path = tmp_path / "geometries" / "geometry_checkpoint.npz"
    before = m.load_geometries(path, require_complete=False)
    assert before["completed"].sum() == 2
    monkeypatch.setattr(m, "save_npz", original)
    m.generate_geometries(path.parent, 0.2e-3)
    after = m.load_geometries(path)
    assert np.array_equal(after["pixels"][:2], before["pixels"][:2])
    assert after["completed"].all()


def test_incomplete_and_wrong_scale_manifests_are_rejected(tmp_path):
    """Require simulation consumers to receive complete, matching layouts.

    Args:
        tmp_path (Path): Isolated directory.
    Returns:
        None.
    Raises:
        AssertionError: If an incomplete or changed-scale manifest is accepted.
    """
    m = module("metasurface.reduced.geometry_manifest")
    path = m.generate_geometries(tmp_path / "geometries", 0.2e-3, limit=2)
    with pytest.raises(ValueError, match="complete"):
        m.load_geometries(path)
    with pytest.raises(ValueError, match="configuration"):
        m.generate_geometries(path.parent, 0.3e-3)


def test_simulation_requires_preexisting_geometry_file(tmp_path):
    """Require stage two to fail before creating or simulating missing designs.

    Args:
        tmp_path (Path): Isolated directory.
    Returns:
        None.
    Raises:
        AssertionError: If the missing manifest is silently regenerated.
    """
    d = module("metasurface.reduced.dataset")
    assert hasattr(d, "simulate_geometries"), "Missing separated simulation stage"
    with pytest.raises(FileNotFoundError):
        d.simulate_geometries(tmp_path / "missing.npz", tmp_path / "raw", module("metasurface.config").PHYSICS, "synthetic")
    assert not (tmp_path / "raw").exists()


def test_simulation_is_raw_only_and_resume_uses_saved_codes(tmp_path, monkeypatch):
    """Require raw spectra without notch extraction and skip committed layouts.

    Args:
        tmp_path (Path): Isolated directory.
        monkeypatch (object): Pytest patch manager.
    Returns:
        None.
    Raises:
        AssertionError: If the stages are coupled or committed work repeats.
    """
    d = module("metasurface.reduced.dataset")
    assert hasattr(d, "simulate_geometries"), "Missing separated simulation stage"
    path = manifest(tmp_path)
    seen = []

    def forbidden(*args, **kwargs):
        """Fail if raw collection attempts feature extraction.

        Args:
            *args (tuple): Unused positional arguments.
            **kwargs (dict): Unused keyword arguments.
        Returns:
            None.
        Raises:
            AssertionError: Whenever invoked.
        """
        raise AssertionError("notch extraction belongs to stage three")

    def response(codes, frequencies):
        """Record layouts submitted to a software-only response generator.

        Args:
            codes (ndarray): Four pattern IDs.
            frequencies (ndarray): Frequency axis in Hz.
        Returns:
            dict: Artificial reflected powers.
        Raises:
            ValueError: If codes are invalid.
        """
        seen.append(tuple(codes))
        return module("metasurface.reduced.synthetic").response(codes, frequencies)

    monkeypatch.setattr(d, "extract_features", forbidden)
    raw = d.simulate_geometries(path, tmp_path / "raw", module("metasurface.config").PHYSICS, "synthetic", 2, response)
    d.simulate_geometries(path, tmp_path / "raw", module("metasurface.config").PHYSICS, "synthetic", 1, response)
    state = d.load_simulations(raw)
    assert seen == [tuple(c) for c in module("metasurface.reduced.geometry").all_codes()[:3]]
    assert state["completed"].sum() == 3
    assert "features" not in state and "reason" not in state
    assert json.loads(state["metadata"].item())["geometry_id"] == module("metasurface.reduced.geometry_manifest").load_geometries(path)["checksum"].item()


def test_raw_simulation_crash_resume_and_physics_identity(tmp_path):
    """Require simulation recovery and rejection of incompatible settings.

    Args:
        tmp_path (Path): Isolated directory.
    Returns:
        None.
    Raises:
        AssertionError: If interrupted simulation loses committed records.
    """
    d = module("metasurface.reduced.dataset")
    assert hasattr(d, "simulate_geometries"), "Missing separated simulation stage"
    path = manifest(tmp_path)
    seen = []

    def interrupted(codes, frequencies):
        """Raise a solver-like failure on the third synthetic invocation.

        Args:
            codes (ndarray): Tile IDs.
            frequencies (ndarray): Frequency axis in Hz.
        Returns:
            dict: Artificial spectra.
        Raises:
            RuntimeError: On the third invocation.
        """
        seen.append(tuple(codes))
        if len(seen) == 3:
            raise RuntimeError("injected simulation crash")
        return module("metasurface.reduced.synthetic").response(codes, frequencies)

    physics = module("metasurface.config").PHYSICS
    with pytest.raises(RuntimeError, match="injected"):
        d.simulate_geometries(path, tmp_path / "raw", physics, "synthetic", simulator=interrupted)
    raw = d.simulate_geometries(path, tmp_path / "raw", physics, "synthetic", limit=1)
    assert d.load_simulations(raw)["completed"].sum() == 3
    with pytest.raises(ValueError, match="configuration"):
        d.simulate_geometries(path, tmp_path / "raw", {**physics, "epsilon_real": 4.3}, "synthetic", limit=1)


def test_offline_preparation_preserves_source_and_is_idempotent(tmp_path, monkeypatch):
    """Require derived features to be reproducible without the solver API.

    Args:
        tmp_path (Path): Isolated directory.
        monkeypatch (object): Pytest patch manager.
    Returns:
        None.
    Raises:
        AssertionError: If preparation changes source data or loads lumapi.
    """
    d = module("metasurface.reduced.dataset")
    assert hasattr(d, "prepare_dataset"), "Missing offline preparation stage"
    raw = d.simulate_geometries(manifest(tmp_path), tmp_path / "raw", module("metasurface.config").PHYSICS, "synthetic", limit=32)
    original = raw.read_bytes()
    monkeypatch.setitem(sys.modules, "lumapi", None)
    prepared = d.prepare_dataset(raw, tmp_path / "prepared" / "training_dataset.npz")
    state = d.load_dataset(prepared)
    assert raw.read_bytes() == original
    assert state["completed"].sum() == 32
    assert (state["reason"] == "ok").sum() == 32
    before = prepared.read_bytes()
    d.prepare_dataset(raw, prepared)
    assert prepared.read_bytes() == before
    assert json.loads(state["metadata"].item())["raw_simulation_id"] == d.load_simulations(raw)["checksum"].item()


def test_geometry_and_data_offline_imports_do_not_load_solver_or_torch():
    """Require the first stage and dataset preparation to import offline.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: If proprietary API or Torch is loaded unnecessarily.
    """
    assert importlib.util.find_spec("metasurface.reduced.geometry_manifest") is not None, "Missing geometry stage"
    code = "import sys; import metasurface.reduced.geometry_manifest, metasurface.reduced.dataset; assert 'torch' not in sys.modules; assert 'lumapi' not in sys.modules; assert 'lumerical' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_network_parameter_counts_match_shapes_and_torch():
    """Require analytical parameter counts to agree with the implemented model.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: If analytical or published-table counts are confused.
    """
    a = module("metasurface.common.network_analysis").analyze_networks()
    assert a["paper_unrestricted"]["parameters"] == 587224
    assert a["paper_unrestricted"]["published_parameter_sum"] == 670024
    assert a["paper_restricted"]["parameters"] == 788648
    assert a["reduced"]["parameters"] == 3440
    assert sum(p.numel() for p in module("metasurface.reduced.learning").InverseNet().parameters()) == a["reduced"]["parameters"]
    assert a["reduced"]["forward_macs_per_sample"] == 3328
