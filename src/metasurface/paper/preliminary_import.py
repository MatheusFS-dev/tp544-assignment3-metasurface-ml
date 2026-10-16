"""Import traceable external S11 records for paper-scale offline training."""

import csv
import json
from pathlib import Path

import numpy as np

from metasurface.common.storage import canonical_json, save_npz
from metasurface.config import PHYSICS


def _sample_codes(path):
    """Read and validate tile-code labels from an external sample table."""
    fields = [f"p{row}{column}" for row in range(1, 5) for column in range(1, 5)]
    result = {}
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "id" not in reader.fieldnames or any(field not in reader.fieldnames for field in fields):
            raise ValueError("Sample table must contain id and p11 through p44 columns.")
        for row in reader:
            try:
                sample_id = int(row["id"])
                values = np.asarray([int(row[field]) for field in fields], dtype=np.uint8)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("Invalid sample ID or tile code.") from error
            if sample_id < 1 or sample_id in result or np.any((values < 1) | (values > 8)):
                raise ValueError("Sample IDs must be unique and tile codes must be from 1 through 8.")
            expected_bits = "".join(f"{value - 1:03b}" for value in values)
            if "bits48" in row and row["bits48"] != expected_bits:
                raise ValueError(f"Sample {sample_id} has inconsistent bits48 encoding.")
            result[sample_id] = values - 1
    if not result:
        raise ValueError("Sample table is empty.")
    return result


def _frequencies_hz(path):
    """Read an increasing GHz frequency vector and convert it to Hz."""
    values = np.atleast_1d(np.loadtxt(path, dtype=float))
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all() or np.any(np.diff(values) <= 0):
        raise ValueError("Frequency file must contain finite, strictly increasing GHz values.")
    frequencies = values * 1e9
    canonical = np.linspace(frequencies[0], frequencies[-1], len(frequencies))
    if not np.allclose(frequencies, canonical, rtol=0, atol=1e-3):
        raise ValueError("Frequency file must be uniformly spaced.")
    return canonical


def _spectra_db(path, expected_points):
    """Read one ID-prefixed dB reflection trace from each text line."""
    result = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            values = line.split()
            if not values:
                continue
            try:
                sample_id = int(values[0])
                spectrum = np.asarray(values[1:], dtype=float)
            except ValueError as error:
                raise ValueError("Spectrum rows must begin with an integer ID and numeric dB values.") from error
            if sample_id < 1 or sample_id in result or spectrum.shape != (expected_points,) or not np.isfinite(spectrum).all():
                raise ValueError("Spectrum IDs must be unique with one finite value per frequency.")
            result[sample_id] = spectrum
    if not result:
        raise ValueError("Spectrum file is empty.")
    return result


def _notch_ids(path):
    """Read IDs from the supplied notch table without treating it as source data."""
    result = set()
    with Path(path).open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or "id" not in reader.fieldnames:
            raise ValueError("Notch table must contain an id column.")
        for row in reader:
            try:
                sample_id = int(row["id"])
            except (TypeError, ValueError) as error:
                raise ValueError("Notch table has an invalid ID.") from error
            if sample_id < 1 or sample_id in result:
                raise ValueError("Notch IDs must be positive and unique.")
            result.add(sample_id)
    return result


def import_preliminary(samples_path, frequencies_path, spectra_path, notches_path,
                       output_path):
    """Build a raw training checkpoint from supplied sample labels and S11 traces.

    Args:
        samples_path (Path-like): CSV with ``id``, tile columns ``p11`` through
            ``p44``, and optional validated ``bits48`` labels.
        frequencies_path (Path-like): One increasing frequency in GHz per line.
        spectra_path (Path-like): One ``id`` plus one S11 dB trace per line.
        notches_path (Path-like): Existing notch CSV used only to verify ID linkage.
        output_path (Path-like): Raw checkpoint destination.

    Returns:
        Path: Atomic checkpoint containing labels, raw reflection power, and provenance.

    Raises:
        ValueError: If IDs, encodings, or trace dimensions disagree.
    """
    sample_codes = _sample_codes(samples_path)
    frequencies = _frequencies_hz(frequencies_path)
    spectra_db = _spectra_db(spectra_path, len(frequencies))
    spectrum_ids = tuple(spectra_db)
    if not set(spectrum_ids).issubset(sample_codes):
        raise ValueError("Every spectrum ID must have a geometry label.")
    if _notch_ids(notches_path) != set(spectrum_ids):
        raise ValueError("Notch and spectrum ID sets differ.")
    codes = np.asarray([sample_codes[sample_id] for sample_id in spectrum_ids], dtype=np.uint8)
    power = np.asarray([10 ** (spectra_db[sample_id] / 10) for sample_id in spectrum_ids])
    physics = {**PHYSICS, "frequency_min_hz": float(frequencies[0]),
               "frequency_max_hz": float(frequencies[-1]),
               "frequency_points": len(frequencies), "metal_model": "pec"}
    metadata = {"schema": 1, "stage": "simulations", "family": "paper-scale-rings-v1",
                "backend": "external_lumerical_import", "physics": physics,
                "available_traces": ["reflection_power"], "feature_trace": "reflection_power",
                "geometry_labels": "p11_to_p44_minus_one_from_amostras_csv",
                "geometry_pixel_manifest": "not_supplied",
                "export_physics": "repository_default_geometry_with_PEC_from_supplied_lsf",
                "reflection_trace": "S11_dB_supplied_as_total_or_co_polarization_unspecified",
                "source_files": {"samples": Path(samples_path).name,
                                 "frequencies": Path(frequencies_path).name,
                                 "spectra": Path(spectra_path).name,
                                 "notches": Path(notches_path).name}}
    state = {"metadata": np.asarray(canonical_json(metadata)), "codes": codes,
             "source_sample_ids": np.asarray(spectrum_ids, dtype=np.int64),
             "frequencies": frequencies, "reflection_power": power,
             "completed": np.ones(len(codes), dtype=bool), "seconds": np.zeros(len(codes)),
             "feature_trace": np.asarray("reflection_power")}
    destination = Path(output_path)
    save_npz(destination, state)
    return destination
