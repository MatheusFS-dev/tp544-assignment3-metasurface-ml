"""Solver boundary tests, without claiming a real electromagnetic run."""

import numpy as np
import pytest

from tests.test_data import module


def test_project_has_physical_geometry_and_boundaries():
    """Project has physical geometry and boundaries.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    script = module("metasurface.reduced.lumerical").project_script([1, 2, 3, 0], module("metasurface.config").PHYSICS)
    for text in ['"Periodic"', '"PML"', '"Backward"', '"Conductivity"', '"Sampled data"', '"reflection"', '"incident"', '"simulation time"']:
        assert text in script
    assert 'set("polarization angle",0)' in script
    assert 'set("z min",-0.0015)' in script
    assert 'set("z max",1.8e-05)' in script


def test_project_keeps_lumerical_default_fdtd_region_name():
    """Avoid renaming the solver region, which Lumerical 2025 R2 rejects."""
    script = module("metasurface.reduced.lumerical").project_script(
        [0, 0, 0, 0], module("metasurface.config").PHYSICS, "empty")
    assert 'set("name","FDTD")' not in script


def test_fr4_requests_fit_settings_that_meet_material_tolerance():
    """Use the fit controls validated against Lumerical 2025 R2."""
    script = module("metasurface.reduced.lumerical").project_script(
        [0, 0, 0, 0], module("metasurface.config").PHYSICS, "empty")
    assert 'setmaterial("FR4_RF","max coefficients",8)' in script
    assert 'setmaterial("FR4_RF","tolerance",0.001)' in script


def test_pec_option_and_reference_do_not_include_copper_pixels():
    """Pec option and reference do not include copper pixels.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    m = module("metasurface.reduced.lumerical")
    p = module("metasurface.config").PHYSICS.copy()
    p["metal_model"] = "pec"
    pec_script = m.project_script([1, 2, 3, 0], p)
    assert 'set("material","PEC (Perfect Electrical Conductor)")' in pec_script
    assert 'set("material","PEC")' not in pec_script
    assert "pixel_" not in m.project_script([1, 2, 3, 0], p, "empty")


@pytest.mark.parametrize("status", [0, 1, 3])
def test_unconverged_solver_rejected(status):
    """Unconverged solver rejected.

    Args:
        status (object): Pytest fixture or parametrized input.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    with pytest.raises(RuntimeError, match="autoshutoff"):
        module("metasurface.reduced.lumerical").check_status(status)


def test_source_pulse_rejects_weak_frequency_edges():
    """Source pulse rejects weak frequency edges.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    m = module("metasurface.reduced.lumerical")
    with pytest.raises(RuntimeError, match="excitation"):
        m.check_pulse(np.array([1e-10, 1, 1]), 1e-3)


def test_source_pulse_uses_actual_spectrum():
    """Source pulse uses actual spectrum.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    m = module("metasurface.reduced.lumerical")
    assert m.check_pulse(np.array([0.1, 1, 0.1]), 1e-3).min() == pytest.approx(0.01)


def test_complex_field_averaging_preserves_frequency_axis():
    """Complex field averaging preserves frequency axis.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    m = module("metasurface.reduced.lumerical")
    fields = np.ones((3, 4, 1, 20), dtype=complex) * (1 + 2j)
    assert np.array_equal(m.mean_field(fields, 20), np.full(20, 1 + 2j))


def test_bad_field_shape_and_nonfinite_fields_rejected():
    """Bad field shape and nonfinite fields rejected.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    m = module("metasurface.reduced.lumerical")
    with pytest.raises(ValueError):
        m.mean_field(np.ones((20, 3)), 20)
    with pytest.raises(ValueError):
        m.mean_field(np.full((1, 1, 1, 20), np.nan), 20)


def test_copper_mesh_resolves_18_micrometre_layer():
    """Copper mesh resolves 18 micrometre layer.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    m = module("metasurface.reduced.lumerical")
    script = m.project_script([2, 2, 2, 2], module("metasurface.config").PHYSICS)
    assert '"copper_mesh_top"' in script
    assert 'set("dz",6e-06)' in script


def test_invalid_physical_band_and_material_rejected():
    """Invalid physical band and material rejected.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    m = module("metasurface.reduced.lumerical")
    p = module("metasurface.config").PHYSICS.copy()
    p["frequency_max_hz"] = 60e9
    with pytest.raises(ValueError, match="diffraction"):
        m.project_script([1, 2, 3, 0], p)
    p["frequency_max_hz"] = 45e9
    p["metal_model"] = "unverified_library_copper"
    with pytest.raises(ValueError):
        m.project_script([1, 2, 3, 0], p)


def test_reference_only_preflight_skips_physical_validation(tmp_path, monkeypatch):
    """Cache only the normalization reference when validation is disabled."""
    m = module("metasurface.reduced.lumerical")
    physics = module("metasurface.config").PHYSICS
    solver = m.LumericalSolver(physics)
    frequencies = np.linspace(
        physics["frequency_min_hz"], physics["frequency_max_hz"],
        physics["frequency_points"])
    raw = {
        "frequencies": frequencies,
        "incident": np.ones(len(frequencies), dtype=complex),
        "ex": np.zeros(len(frequencies), dtype=complex),
        "ey": np.zeros(len(frequencies), dtype=complex),
        "total": np.zeros(len(frequencies)),
    }
    calls = []

    def reference_run(codes, reference="device", project_path=None):
        calls.append((list(codes), reference, project_path))
        return raw

    monkeypatch.setattr(solver, "_run", reference_run)
    solver.preflight(tmp_path, mode="reference_only")
    assert len(calls) == 1
    assert calls[0][1] == "empty"
    assert (tmp_path / "physical_reference.npz").exists()

    cached = m.LumericalSolver(physics)
    monkeypatch.setattr(cached, "_run", lambda *args, **kwargs: pytest.fail("cache not reused"))
    cached.preflight(tmp_path, mode="reference_only")
    assert np.array_equal(cached.reference["incident"], raw["incident"])


def test_lumerical_run_explicitly_requires_gpu():
    """Require every physical solve to select FDTD GPU explicitly."""
    m = module("metasurface.reduced.lumerical")
    calls = []

    class FakeSession:
        """Record the hardware check and solver launch arguments."""

        def runsystemcheck(self, *args):
            calls.append(("check", args))
            return {"memory": "fixture"}

        def run(self, *args):
            calls.append(("run", args))

    m.run_on_gpu(FakeSession())
    assert calls == [
        ("check", ("FDTD", "GPU")),
        ("run", ("FDTD", "GPU")),
    ]


@pytest.mark.parametrize("failure_method", ["runsystemcheck", "run"])
def test_lumerical_gpu_failure_has_no_cpu_fallback(failure_method):
    """Raise a clear error instead of retrying an unavailable GPU solve on CPU."""
    m = module("metasurface.reduced.lumerical")
    calls = []

    class FailingSession:
        """Fail one GPU operation while recording all requested devices."""

        def runsystemcheck(self, *args):
            calls.append(("check", args))
            if failure_method == "runsystemcheck":
                raise RuntimeError("GPU unavailable")
            return {}

        def run(self, *args):
            calls.append(("run", args))
            if failure_method == "run":
                raise RuntimeError("GPU unavailable")

    with pytest.raises(RuntimeError, match="GPU-required Lumerical FDTD"):
        m.run_on_gpu(FailingSession())
    assert all("CPU" not in args for _, args in calls)
