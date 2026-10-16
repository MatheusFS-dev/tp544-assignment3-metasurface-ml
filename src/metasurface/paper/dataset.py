"""Raw physical spectra and offline 24-descriptor preparation."""

import json
from pathlib import Path
import time

import numpy as np

from metasurface.common.progress import Progress
from metasurface.common.run_records import timed_part
from metasurface.common.storage import canonical_json, load_npz, save_npz
from metasurface.paper.features import extract_features
from metasurface.paper.geometry import rectangles
from metasurface.paper.manifest import load_manifest
from metasurface.reduced.spectra import validate_spectrum


def load_raw(path):
    """Load a paper-scale raw-spectrum checkpoint with integrity checks.

    Args:
        path (Path-like): Raw simulation checkpoint.

    Returns:
        dict: Validated saved spectra and metadata.

    Raises:
        ValueError: If arrays or provenance are invalid.
    """
    state = load_npz(path)
    metadata = json.loads(state["metadata"].item())
    count = len(state["codes"])
    p = metadata["physics"]
    expected = np.linspace(p["frequency_min_hz"], p["frequency_max_hz"], p["frequency_points"])
    available_traces = tuple(metadata.get("available_traces", ("co", "cross", "total")))
    feature_trace = metadata.get("feature_trace", "co")
    if (metadata.get("family") != "paper-scale-rings-v1" or metadata.get("stage") not in ("simulations", "training_dataset") or metadata.get("backend") not in ("lumerical", "test_fixture", "synthetic_surrogate", "external_lumerical_import")
            or state["codes"].shape != (count, 16) or state["completed"].shape != (count,)
            or state["seconds"].shape != (count,) or not np.array_equal(state["frequencies"], expected)
            or not np.isfinite(state["seconds"]).all() or np.any(state["seconds"] < 0)
            or not available_traces or feature_trace not in available_traces):
        raise ValueError("Invalid paper raw-spectrum checkpoint.")
    for key in available_traces:
        if key not in state:
            raise ValueError("Invalid paper spectrum shape.")
        if state[key].shape != (count, len(expected)):
            raise ValueError("Invalid paper spectrum shape.")
        for i in np.flatnonzero(state["completed"]):
            validate_spectrum(expected, state[key][i])
    return state


def simulate(manifest_path, directory, physics, limit=None, simulator=None,
             preflight_mode="full"):
    """Run Lumerical for sampled layouts and commit every raw spectrum.

    Args:
        manifest_path (Path-like): Complete paper geometry manifest.
        directory (Path-like): Paper data directory.
        physics (dict): Physical simulation settings.
        limit (int or None): Additional layouts to simulate.
        simulator (callable or None): Labeled test fixture, never for a real run.
        preflight_mode (str): ``full`` or reference-only solver preflight.

    Returns:
        Path: Raw simulation checkpoint.

    Raises:
        ValueError: If settings or saved identities differ.
        RuntimeError: If the physical preflight fails.
    """
    manifest = load_manifest(manifest_path)
    directory = Path(directory)
    path = directory / "simulation_checkpoint.npz"
    backend = "test_fixture" if simulator is not None else "lumerical"
    if preflight_mode not in ("full", "reference_only"):
        raise ValueError("Invalid preflight mode.")
    identity = {"schema": 1, "stage": "simulations", "family": "paper-scale-rings-v1", "backend": backend,
                "physics": physics, "geometry_id": manifest["checksum"].item()}
    if preflight_mode != "full":
        identity["preflight_mode"] = preflight_mode
    meta = canonical_json(identity)
    count = len(manifest["codes"])
    f = np.linspace(physics["frequency_min_hz"], physics["frequency_max_hz"], physics["frequency_points"])
    if path.exists():
        state = load_raw(path)
        if state["metadata"].item() != meta or not np.array_equal(state["codes"], manifest["codes"]):
            raise ValueError("Paper simulation identity changed. Use a new directory.")
    else:
        state = {"metadata": np.asarray(meta), "codes": manifest["codes"].copy(),
                 "frequencies": f, "completed": np.zeros(count, dtype=bool), "seconds": np.zeros(count)}
        for key in ("co", "cross", "total"):
            state[key] = np.full((count, len(f)), np.nan)
        save_npz(path, state)
    pending = np.flatnonzero(~state["completed"])
    if limit is not None:
        pending = pending[:limit]
    progress = Progress("Paper simulations", count, int(state["completed"].sum()))
    if len(pending) and simulator is None:
        from metasurface.reduced.lumerical import LumericalSolver
        solver = LumericalSolver(physics, rectangle_fn=rectangles, code_count=16)
        with timed_part("lumerical_preflight"):
            solver.preflight(directory, mode=preflight_mode)
        simulator = solver.simulate
    with timed_part("layout_collection"):
        for i in pending:
            started = time.monotonic()
            result = simulator(state["codes"][i], f)
            for key in ("co", "cross", "total"):
                _, state[key][i] = validate_spectrum(f, result[key])
            state["seconds"][i] = time.monotonic() - started
            state["completed"][i] = True
            save_npz(path, state)
            progress.committed(int(state["completed"].sum()), f"layout {i:04d}")
    return path


def prepare(raw_path, output_path):
    """Derive 24 descriptors offline while preserving source identity.

    Args:
        raw_path (Path-like): Saved raw spectra.
        output_path (Path-like): Prepared checkpoint destination.

    Returns:
        Path: Prepared paper dataset.
    """
    if Path(raw_path).resolve() == Path(output_path).resolve():
        raise ValueError("Prepared paper dataset must not overwrite raw spectra.")
    raw = load_raw(raw_path)
    meta = json.loads(raw["metadata"].item())
    if meta.get("backend") not in ("lumerical", "test_fixture", "synthetic_surrogate", "external_lumerical_import"):
        raise ValueError("Unsupported paper data provenance.")
    source_id = raw.pop("checksum").item()
    meta.update(stage="training_dataset", raw_simulation_id=source_id)
    raw["metadata"] = np.asarray(canonical_json(meta))
    destination = Path(output_path)
    if destination.exists():
        previous = load_npz(destination)
        if previous["metadata"].item() == raw["metadata"].item():
            return destination
    count = len(raw["codes"])
    raw["features"] = np.zeros((count, 24))
    raw["reason"] = np.full(count, "pending", dtype="U32")
    completed = np.flatnonzero(raw["completed"])
    progress = Progress("Paper dataset preparation", len(completed))
    for position, i in enumerate(completed, 1):
        trace = meta.get("feature_trace", "co")
        raw["features"][i], raw["reason"][i] = extract_features(raw["frequencies"], raw[trace][i])
        if position % 100 == 0 or position == len(completed):
            print(f"Paper dataset: {position}/{len(completed)} extracted", flush=True)
    save_npz(destination, raw)
    progress.committed(len(completed))
    return destination


def load_dataset(path):
    """Load and validate a prepared paper-scale dataset.

    Args:
        path (Path-like): Prepared checkpoint.

    Returns:
        dict: Validated data and fingerprint.

    Raises:
        ValueError: If descriptors or provenance disagree with spectra.
    """
    state = load_raw(path)
    meta = json.loads(state["metadata"].item())
    if meta.get("stage") != "training_dataset" or state["features"].shape != (len(state["codes"]), 24):
        raise ValueError("Invalid paper dataset identity or dimensions.")
    for i in np.flatnonzero(state["completed"]):
        trace = meta.get("feature_trace", "co")
        features, reason = extract_features(state["frequencies"], state[trace][i])
        if reason != state["reason"][i] or not np.array_equal(features, state["features"][i]):
            raise ValueError("Paper descriptors differ from source spectra.")
    return state
