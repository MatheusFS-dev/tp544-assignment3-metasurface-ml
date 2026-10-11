"""Explicit paper-scale approximation to unspecified notch preprocessing."""

import numpy as np

from metasurface.reduced.spectra import validate_spectrum


def extract_features(frequencies, reflectance):
    """Encode up to eight resolved -10 dB notches as 24 descriptors.

    Args:
        frequencies (array-like): Increasing frequencies in Hz.
        reflectance (array-like): Co-polarized reflected power.

    Returns:
        tuple[ndarray, str]: Zero-padded descriptors and eligibility reason.
    """
    f, power = validate_spectrum(frequencies, reflectance)
    db = 10 * np.log10(np.maximum(power, 1e-12))
    changes = np.diff(np.r_[False, db < -10, False].astype(int))
    spans = list(zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)))
    result = np.zeros(24)
    if not spans:
        return result, "no_notches"
    if any(left == 0 or right == len(f) for left, right in spans):
        return result, "truncated_notch"
    if len(spans) > 8:
        return result, "too_many_notches"
    for slot, (left, right) in enumerate(spans):
        center = left + np.argmin(db[left:right])
        if right - left < 3 or db[center] > -10.5:
            return np.zeros(24), "unresolved_notch"
        start = np.interp(-10, [db[left], db[left - 1]], [f[left], f[left - 1]])
        end = np.interp(-10, [db[right - 1], db[right]], [f[right - 1], f[right]])
        result[3 * slot:3 * slot + 3] = (f[center] / 1e9, db[center], (end - start) / 1e9)
    return result, "ok"
