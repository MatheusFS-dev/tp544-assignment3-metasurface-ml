"""Resolved -10 dB notch descriptors, with explicit exclusions."""

import numpy as np


def validate_spectrum(frequencies, reflectance):
    """Validate an ascending RF spectrum before descriptor extraction.

    Args:
        frequencies (array-like): Frequencies in Hz, increasing strictly.
        reflectance (array-like): Reflected power divided by incident power.
    Returns:
        tuple[ndarray, ndarray]: Validated frequency and power vectors.
    Raises:
        ValueError: If shapes, finiteness, ordering, or physical bounds fail.
    """
    f = np.asarray(frequencies, dtype=float)
    r = np.asarray(reflectance, dtype=float)
    if f.ndim != 1 or len(f) < 10 or r.shape != f.shape:
        raise ValueError("Expected matching one-dimensional spectra with at least 10 points.")
    if not np.isfinite(f).all() or not np.isfinite(r).all():
        raise ValueError("Nonfinite spectral data.")
    if np.any(f <= 0) or np.any(np.diff(f) <= 0) or np.any(r < 0) or np.any(r > 1.05):
        raise ValueError("Invalid frequencies or nonphysical reflection power.")
    return f, r


def extract_features(frequencies, reflectance):
    """Encode up to two resolved reflection notches in frequency order.

    Args:
        frequencies (array-like): Strictly increasing frequencies in Hz.
        reflectance (array-like): Co-polarized reflected power fraction.
    Returns:
        tuple[ndarray, str]: Eight features and an eligibility reason. Each slot
        stores GHz, dB, GHz, and presence. Unused slots contain zeros.
    Raises:
        ValueError: If the input spectrum is invalid.
    Notes:
        Width is the contiguous interval below -10 dB, with interpolated
        crossings. At least three subthreshold points and a -10.5 dB minimum
        are required. This explicit definition differs from the paper's
        insufficiently documented width convention.
    """
    f, r = validate_spectrum(frequencies, reflectance)
    db = 10 * np.log10(np.maximum(r, 1e-12))
    below = db < -10
    changes = np.diff(np.r_[False, below, False].astype(int))
    spans = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
    features = np.zeros(8, dtype=float)
    if not spans:
        return features, "no_notches"
    if any(start == 0 or end == len(f) for start, end in spans):
        return features, "truncated_notch"
    if len(spans) > 2:
        return features, "too_many_notches"
    for slot, (start, end) in enumerate(spans):
        peak = start + np.argmin(db[start:end])
        if end - start < 3 or db[peak] > -10.5:
            return np.zeros(8), "unresolved_notch"
        left = np.interp(-10, [db[start], db[start - 1]], [f[start], f[start - 1]])
        right = np.interp(-10, [db[end - 1], db[end]], [f[end - 1], f[end]])
        features[slot * 4:slot * 4 + 4] = [f[peak] / 1e9, db[peak], (right - left) / 1e9, 1]
    return features, "ok"
