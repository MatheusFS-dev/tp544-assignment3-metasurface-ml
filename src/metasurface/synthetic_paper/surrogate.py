"""Deterministic passive surrogate for paper-scale reflection spectra."""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from metasurface.paper.geometry import masks, pixel_layout


@dataclass(frozen=True)
class SurrogateConfig:
    """Configuration for the qualitative resonant spectral surrogate."""

    schema: int = 1
    effective_permittivity: float = 2.6
    frequency_scale_ghz_pixel: float = 285.0
    internal_loss_ghz: float = 0.16
    external_loss_ghz: float = 0.22
    neighbour_shift_ghz: float = 0.13
    position_shift_ghz: float = 0.08
    merge_distance_ghz: float = 0.30
    maximum_modes: int = 8


def _validate_frequencies(frequencies):
    values = np.asarray(frequencies, dtype=float)
    if (values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all()
            or np.any(np.diff(values) <= 0) or values[0] <= 0):
        raise ValueError("Frequencies must be finite, positive, and strictly increasing.")
    return values


def _tile_properties():
    properties = []
    for index, tile in enumerate(masks()):
        occupied = np.argwhere(tile)
        height = int(np.ptp(occupied[:, 0]) + 1)
        width = int(np.ptp(occupied[:, 1]) + 1)
        path = 2.0 * (height + width) - 4.0
        edges = 0
        padded = np.pad(tile, 1)
        for axis in (0, 1):
            edges += int(np.abs(np.diff(padded, axis=axis)).sum())
        properties.append({"area": int(tile.sum()), "perimeter": edges,
                           "path": path, "family": "diamond" if index >= 6 else "square"})
    return properties


def geometry_descriptors(codes):
    """Derive exact mask summaries and qualitative modal properties.

    Args:
        codes (array-like): Sixteen paper tile IDs from zero through seven.

    Returns:
        dict: Geometry summaries used by the surrogate.

    Raises:
        ValueError: If the tile codes are invalid.
    """
    pixel_layout(codes)
    values = np.asarray(codes, dtype=np.uint8).reshape(4, 4)
    properties = _tile_properties()
    paths = np.array([properties[int(code)]["path"] for code in values.flat])
    areas = np.array([properties[int(code)]["area"] for code in values.flat])
    perimeters = np.array([properties[int(code)]["perimeter"] for code in values.flat])
    agreements = []
    for row in range(4):
        for col in range(4):
            if col < 3:
                agreements.append(values[row, col] == values[row, col + 1])
            if row < 3:
                agreements.append(values[row, col] == values[row + 1, col])
    mean_path = float(paths.mean())
    config = SurrogateConfig()
    fundamental = config.frequency_scale_ghz_pixel / (
        mean_path * np.sqrt(config.effective_permittivity))
    return {"mean_path_pixels": mean_path, "mean_area_pixels": float(areas.mean()),
            "mean_perimeter_pixels": float(perimeters.mean()),
            "same_neighbour_fraction": float(np.mean(agreements)),
            "tile_counts": np.bincount(values.ravel(), minlength=8),
            "fundamental_ghz": float(fundamental)}


def surrogate_identity(config=None):
    """Return a stable identity for all surrogate-defining constants.

    Args:
        config (SurrogateConfig or None): Optional configuration override.

    Returns:
        str: SHA-256 hexadecimal identity.
    """
    selected = config or SurrogateConfig()
    payload = {"config": asdict(selected), "masks": masks().tolist()}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _candidate_modes(codes, config):
    values = np.asarray(codes, dtype=np.uint8).reshape(4, 4)
    properties = _tile_properties()
    modes = []
    for row in range(4):
        for col in range(4):
            code = int(values[row, col])
            item = properties[code]
            neighbours = []
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr, cc = row + dr, col + dc
                if 0 <= rr < 4 and 0 <= cc < 4:
                    neighbours.append(int(values[rr, cc]))
            agreement = sum(other == code for other in neighbours) / max(len(neighbours), 1)
            base = config.frequency_scale_ghz_pixel / (
                item["path"] * np.sqrt(config.effective_permittivity))
            spatial = config.position_shift_ghz * ((row - 1.5) + 0.7 * (col - 1.5))
            coupling = config.neighbour_shift_ghz * (agreement - 0.5)
            family_shift = 0.35 if item["family"] == "diamond" else 0.0
            fill = item["area"] / 64.0
            for harmonic, strength in ((1.0, 1.0), (1.86, 0.62), (2.63, 0.36)):
                center = harmonic * base + spatial + coupling + family_shift
                if 3.0 <= center <= 46.0:
                    internal = config.internal_loss_ghz * (1.15 - 0.5 * fill)
                    external = config.external_loss_ghz * strength * (0.75 + 0.5 * agreement)
                    modes.append([center, internal, external, strength * (0.5 + fill)])
    modes.sort(key=lambda mode: mode[0])
    merged = []
    for mode in modes:
        if merged and abs(mode[0] - merged[-1][0]) <= config.merge_distance_ghz:
            total_weight = merged[-1][3] + mode[3]
            merged[-1][0] = (merged[-1][0] * merged[-1][3] + mode[0] * mode[3]) / total_weight
            merged[-1][1] = min(0.8, merged[-1][1] + 0.15 * mode[1])
            merged[-1][2] = min(1.2, merged[-1][2] + 0.18 * mode[2])
            merged[-1][3] = total_weight
        else:
            merged.append(mode.copy())
    ranked = sorted(merged, key=lambda mode: mode[3], reverse=True)[:config.maximum_modes]
    return sorted(ranked, key=lambda mode: mode[0])


def response(codes, frequencies, config=None):
    """Generate deterministic qualitative reflection powers.

    Args:
        codes (array-like): Sixteen paper tile IDs.
        frequencies (array-like): Increasing positive frequencies in Hz.
        config (SurrogateConfig or None): Optional surrogate configuration.

    Returns:
        dict: Co-polar, cross-polar, and total reflected power arrays.

    Raises:
        ValueError: If geometry, frequencies, or configuration are invalid.

    Notes:
        The response is a passive software surrogate, not an electromagnetic
        simulation or a calibrated prediction.
    """
    pixel_layout(codes)
    f = _validate_frequencies(frequencies)
    selected = config or SurrogateConfig()
    if (selected.effective_permittivity <= 0 or selected.maximum_modes < 1
            or selected.internal_loss_ghz <= 0 or selected.external_loss_ghz <= 0):
        raise ValueError("Invalid surrogate configuration.")
    ghz = f / 1e9
    reflection = -np.ones_like(ghz, dtype=np.complex128)
    for center, internal, external, _ in _candidate_modes(codes, selected):
        detuning = ghz - center
        factor = (1j * detuning + internal - external) / (
            1j * detuning + internal + external)
        reflection *= factor
    co = np.clip(np.abs(reflection) ** 2, 0.0, 1.0)
    cross = np.zeros_like(co)
    return {"co": co, "cross": cross, "total": co.copy()}
