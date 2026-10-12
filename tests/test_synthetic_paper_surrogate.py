"""Contracts for the paper-scale deterministic spectral surrogate."""

import numpy as np
import pytest

from metasurface.synthetic_paper.surrogate import (
    geometry_descriptors,
    response,
    surrogate_identity,
)


FREQUENCIES = np.linspace(4e9, 45e9, 1641)


def test_surrogate_is_deterministic_and_geometry_dependent():
    """Require exact repetition and a nontrivial geometry dependency."""
    first_codes = np.zeros(16, dtype=np.uint8)
    second_codes = np.ones(16, dtype=np.uint8)
    first = response(first_codes, FREQUENCIES)
    repeated = response(first_codes, FREQUENCIES)
    second = response(second_codes, FREQUENCIES)

    assert set(first) == {"co", "cross", "total"}
    assert all(values.shape == (1641,) for values in first.values())
    assert all(np.array_equal(first[key], repeated[key]) for key in first)
    assert not np.array_equal(first["co"], second["co"])
    assert surrogate_identity() == surrogate_identity()


def test_surrogate_is_passive_and_power_consistent():
    """Require finite passive reflected powers on representative layouts."""
    for codes in (np.zeros(16, dtype=np.uint8), np.arange(16) % 8,
                  np.full(16, 7, dtype=np.uint8)):
        result = response(codes, FREQUENCIES)
        assert all(np.isfinite(values).all() for values in result.values())
        assert np.all((0 <= result["co"]) & (result["co"] <= 1))
        assert np.array_equal(result["cross"], np.zeros(1641))
        assert np.allclose(result["total"], result["co"] + result["cross"])
        assert np.all(result["total"] <= 1)


def test_surrogate_rejects_invalid_inputs():
    """Reject malformed layouts and invalid frequency axes."""
    with pytest.raises(ValueError):
        response(np.zeros(15), FREQUENCIES)
    with pytest.raises(ValueError):
        response(np.zeros(16), FREQUENCIES[::-1])
    with pytest.raises(ValueError):
        response(np.zeros(16), np.array([4e9, np.nan, 5e9]))


def test_spatial_arrangement_changes_coupling():
    """Distinguish equal tile histograms with different neighbourhoods."""
    grouped = np.array([1] * 8 + [5] * 8, dtype=np.uint8)
    checkerboard = np.array([1, 5] * 8, dtype=np.uint8)
    assert sorted(grouped.tolist()) == sorted(checkerboard.tolist())
    assert not np.array_equal(response(grouped, FREQUENCIES)["co"],
                              response(checkerboard, FREQUENCIES)["co"])


def test_larger_current_path_has_lower_fundamental():
    """Preserve the expected inverse relation between path and frequency."""
    small = geometry_descriptors(np.full(16, 1, dtype=np.uint8))
    large = geometry_descriptors(np.full(16, 5, dtype=np.uint8))
    assert large["mean_path_pixels"] > small["mean_path_pixels"]
    assert large["fundamental_ghz"] < small["fundamental_ghz"]
    assert 4 < small["fundamental_ghz"] < 45
    assert 4 < large["fundamental_ghz"] < 45
