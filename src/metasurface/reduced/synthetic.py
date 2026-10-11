"""Explicit synthetic fixture backend, never an electromagnetic solver."""

import numpy as np


def response(codes, frequencies):
    """Produce deterministic artificial notches for software verification.

    Args:
        codes (array-like): Four pattern IDs.
        frequencies (array-like): Frequency vector in Hz.
    Returns:
        dict: Co-polarized, cross-polarized, and total reflected power vectors.
    Raises:
        ValueError: If geometry codes are invalid.
    Notes:
        These invented responses do not represent the physical geometry.
    """
    from metasurface.reduced.geometry import pixel_layout

    pixel_layout(codes)
    a, b, c, d = np.asarray(codes, dtype=int)
    f = np.asarray(frequencies) / 1e9
    center = 8 + 0.08 * (64 * a + 16 * b + 4 * c + d)
    db = -(18 + b) * np.maximum(1 - abs(f - center) / (0.4 + 0.05 * c), 0)
    if d == 3:
        db -= (16 + c) * np.maximum(1 - abs(f - 40) / 0.5, 0)
    power = 10**(db / 10)
    return {"co": power, "cross": np.zeros_like(power), "total": power.copy()}
