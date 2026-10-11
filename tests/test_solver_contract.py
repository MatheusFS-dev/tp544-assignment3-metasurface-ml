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
    script = module("lumerical").project_script([1, 2, 3, 0], module("config").PHYSICS)
    for text in ['"Periodic"', '"PML"', '"Backward"', '"Conductivity"', '"Sampled data"', '"reflection"', '"incident"', '"simulation time"']:
        assert text in script
    assert 'set("polarization angle",0)' in script
    assert 'set("z min",-0.0015)' in script
    assert 'set("z max",1.8e-05)' in script


def test_pec_option_and_reference_do_not_include_copper_pixels():
    """Pec option and reference do not include copper pixels.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    m = module("lumerical")
    p = module("config").PHYSICS.copy()
    p["metal_model"] = "pec"
    assert 'set("material","PEC")' in m.project_script([1, 2, 3, 0], p)
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
        module("lumerical").check_status(status)


def test_source_pulse_rejects_weak_frequency_edges():
    """Source pulse rejects weak frequency edges.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: If the expected contract is violated.
    """
    m = module("lumerical")
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
    m = module("lumerical")
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
    m = module("lumerical")
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
    m = module("lumerical")
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
    m = module("lumerical")
    script = m.project_script([2, 2, 2, 2], module("config").PHYSICS)
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
    m = module("lumerical")
    p = module("config").PHYSICS.copy()
    p["frequency_max_hz"] = 60e9
    with pytest.raises(ValueError, match="diffraction"):
        m.project_script([1, 2, 3, 0], p)
    p["frequency_max_hz"] = 45e9
    p["metal_model"] = "unverified_library_copper"
    with pytest.raises(ValueError):
        m.project_script([1, 2, 3, 0], p)
