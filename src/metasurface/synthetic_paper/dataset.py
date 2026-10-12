"""Resumable raw datasets for the synthetic paper-scale surrogate."""

from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from metasurface.common.progress import Progress
from metasurface.common.storage import canonical_json, save_npz
from metasurface.paper.dataset import load_raw as load_paper_raw
from metasurface.paper.dataset import prepare as prepare_paper
from metasurface.paper.manifest import load_manifest
from metasurface.reduced.spectra import validate_spectrum
from metasurface.synthetic_paper.surrogate import SurrogateConfig, response, surrogate_identity


def load_raw(path):
    """Load a validated synthetic-surrogate raw or prepared checkpoint.

    Args:
        path (Path-like): Synthetic checkpoint path.

    Returns:
        dict: Validated arrays and checksum.

    Raises:
        ValueError: If the checkpoint does not have synthetic provenance.
    """
    state = load_paper_raw(path)
    if json.loads(state["metadata"].item()).get("backend") != "synthetic_surrogate":
        raise ValueError("Expected synthetic_surrogate provenance.")
    return state


def generate_spectra(manifest_path, directory, physics, surrogate_config=None, limit=None):
    """Generate and checkpoint deterministic synthetic reflection spectra.

    Args:
        manifest_path (Path-like): Complete paper-scale geometry manifest.
        directory (Path-like): Destination data directory.
        physics (dict): Frequency grid and retained physical specification.
        surrogate_config (SurrogateConfig or None): Surrogate constants.
        limit (int or None): Maximum additional rows, used for controlled runs.

    Returns:
        Path: Raw simulation-compatible checkpoint path.

    Raises:
        ValueError: If a saved checkpoint has a different identity.
    """
    manifest = load_manifest(manifest_path)
    selected = surrogate_config or SurrogateConfig()
    directory = Path(directory)
    path = directory / "simulation_checkpoint.npz"
    metadata = canonical_json({"schema": 1, "stage": "simulations",
        "family": "paper-scale-rings-v1", "backend": "synthetic_surrogate",
        "physics": physics, "geometry_id": manifest["checksum"].item(),
        "surrogate_id": surrogate_identity(selected), "surrogate": asdict(selected)})
    frequencies = np.linspace(physics["frequency_min_hz"], physics["frequency_max_hz"],
                              physics["frequency_points"])
    count = len(manifest["codes"])
    if path.exists():
        state = load_raw(path)
        if state["metadata"].item() != metadata or not np.array_equal(state["codes"], manifest["codes"]):
            raise ValueError("Synthetic dataset identity changed. Use a new run directory.")
    else:
        state = {"metadata": np.asarray(metadata), "codes": manifest["codes"].copy(),
                 "frequencies": frequencies, "completed": np.zeros(count, dtype=bool),
                 "seconds": np.zeros(count)}
        for key in ("co", "cross", "total"):
            state[key] = np.full((count, len(frequencies)), np.nan)
        save_npz(path, state)
    pending = np.flatnonzero(~state["completed"])
    if limit is not None:
        if not isinstance(limit, int) or limit < 0:
            raise ValueError("Generation limit must be a nonnegative integer or None.")
        pending = pending[:limit]
    progress = Progress("Synthetic paper spectra", count, int(state["completed"].sum()))
    for index in pending:
        started = time.monotonic()
        result = response(state["codes"][index], frequencies, selected)
        for key in ("co", "cross", "total"):
            _, state[key][index] = validate_spectrum(frequencies, result[key])
        if not np.allclose(state["total"][index], state["co"][index] + state["cross"][index]):
            raise ValueError("Synthetic total reflection is inconsistent.")
        state["seconds"][index] = time.monotonic() - started
        state["completed"][index] = True
        save_npz(path, state)
        progress.committed(int(state["completed"].sum()), f"layout {index:04d}")
    return path


def prepare(raw_path, output_path):
    """Extract paper-compatible notch descriptors from synthetic spectra.

    Args:
        raw_path (Path-like): Synthetic raw checkpoint.
        output_path (Path-like): Separate prepared dataset path.

    Returns:
        Path: Prepared dataset path.
    """
    load_raw(raw_path)
    result = prepare_paper(raw_path, output_path)
    load_raw(result)
    return result

