"""Raw simulation collection and separate offline dataset preparation."""

import json
from pathlib import Path
import time

import numpy as np

from geometry import all_codes
from geometry_manifest import load_geometries
from spectra import extract_features, validate_spectrum
from storage import canonical_json, load_npz, save_npz


def load_simulations(path):
    """Load raw reflected powers without extracting or interpreting notches.

    Args:
        path (Path-like): Raw simulation checkpoint from stage two.
    Returns:
        dict: Geometry codes, frequencies, powers, runtime and provenance.
    Raises:
        FileNotFoundError: If no checkpoint exists.
        ValueError: If dimensions, physics, provenance or spectra are invalid.
    """
    state = load_npz(path)
    metadata = json.loads(state["metadata"].item())
    if (metadata.get("stage") != "simulations" or metadata.get("schema") != 2
            or metadata.get("backend") not in ("synthetic", "lumerical")
            or not isinstance(metadata.get("geometry_id"), str)):
        raise ValueError("Invalid simulation provenance.")
    p = metadata["physics"]
    f = np.linspace(p["frequency_min_hz"], p["frequency_max_hz"], p["frequency_points"])
    if (not np.array_equal(state["codes"], all_codes())
            or not np.array_equal(state["frequencies"], f)
            or state["completed"].shape != (256,)
            or state["completed"].dtype != np.bool_
            or state["seconds"].shape != (256,)
            or not np.isfinite(state["seconds"]).all() or np.any(state["seconds"] < 0)):
        raise ValueError("Invalid simulation geometry plan, dimensions or times.")
    for key in ("co", "cross", "total"):
        if state[key].shape != (256, len(f)):
            raise ValueError("Invalid simulation spectrum dimensions.")
        for i in np.flatnonzero(state["completed"]):
            validate_spectrum(f, state[key][i])
    return state


def simulate_geometries(geometry_path, directory, physics, backend="lumerical", limit=None, simulator=None):
    """Simulate saved layouts, durably committing each completed raw spectrum.

    Args:
        geometry_path (Path-like): Complete stage-one geometry checkpoint.
        directory (Path-like): Raw simulation and reference project directory.
        physics (dict): Material, source, mesh and frequency configuration.
        backend (str): Explicit 'lumerical' or software-only 'synthetic' backend.
        limit (int or None): Maximum additional simulations for this invocation.
        simulator (callable or None): Synthetic test response generator only.
    Returns:
        Path: Resumable raw simulation checkpoint, without notch descriptors.
    Raises:
        FileNotFoundError: If the saved geometry checkpoint is absent.
        ValueError: If geometry, provenance, settings or returned spectra differ.
        RuntimeError: If the physical solver or numerical acceptance gates fail.
        OSError: If checkpoint persistence fails.
    Notes:
        The saved masks must agree with the code-based solver construction.
        `seconds` includes the per-layout solver call and array validation but
        excludes the physical preflight and checkpoint serialization.
    Examples:
        simulate_geometries('runs/geometries/geometry_checkpoint.npz',
                            'runs/data', PHYSICS, limit=4)
    """
    plan = load_geometries(geometry_path)
    if json.loads(plan["metadata"].item())["pixel_m"] != physics["pixel_m"]:
        raise ValueError("Geometry pixel scale differs from simulation configuration.")
    if backend not in ("synthetic", "lumerical"):
        raise ValueError("Invalid simulation backend.")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("Simulation limit must be a positive integer or None.")
    if simulator is not None and backend != "synthetic":
        raise ValueError("Injected simulators are restricted to synthetic verification.")
    # Loading the plan first prevents stage two from silently inventing input.
    directory = Path(directory)
    path = directory / "simulation_checkpoint.npz"
    metadata = canonical_json({"schema": 2, "stage": "simulations",
                               "family": "mirrored-square-rings-v1", "backend": backend,
                               "physics": physics, "geometry_id": plan["checksum"].item()})
    f = np.linspace(physics["frequency_min_hz"], physics["frequency_max_hz"], physics["frequency_points"])
    if path.exists():
        state = load_simulations(path)
        if state["metadata"].item() != metadata:
            raise ValueError("Simulation configuration or geometry changed. Use a new directory.")
    else:
        state = {"metadata": np.asarray(metadata), "codes": plan["codes"].copy(), "frequencies": f,
                 "completed": np.zeros(256, dtype=bool), "seconds": np.zeros(256)}
        for key in ("co", "cross", "total"):
            state[key] = np.full((256, len(f)), np.nan)
        save_npz(path, state)
    pending = np.flatnonzero(~state["completed"])
    if limit is not None:
        pending = pending[:limit]
    if not len(pending):
        print("All requested saved geometries are already simulated.")
        return path
    if backend == "synthetic":
        from synthetic import response

        print("SYNTHETIC SOFTWARE TEST: spectra have no electromagnetic meaning.")
        simulator = response if simulator is None else simulator
    else:
        from lumerical import LumericalSolver

        solver = LumericalSolver(physics)
        solver.preflight(directory)
        simulator = solver.simulate
    for i in pending:
        start = time.monotonic()
        result = simulator(state["codes"][i], f)
        for key in ("co", "cross", "total"):
            _, state[key][i] = validate_spectrum(f, result[key])
        state["seconds"][i] = time.monotonic() - start
        state["completed"][i] = True
        save_npz(path, state)
        print(f"Simulation {i:03d}: {state['completed'].sum()}/256 saved, {state['seconds'][i]:.2f} s")
    return path


def prepare_dataset(simulation_path, output_path):
    """Derive notch descriptors offline while preserving every raw response.

    Args:
        simulation_path (Path-like): Saved raw simulation checkpoint.
        output_path (Path-like): Prepared training dataset NPZ destination.
    Returns:
        Path: Dataset with features, eligibility and raw-source fingerprint.
    Raises:
        FileNotFoundError: If raw simulations are unavailable.
        ValueError: If source data are invalid or output would overwrite them.
        OSError: If the atomic prepared snapshot cannot be saved.
    Notes:
        Preparation is inexpensive and deterministic. If interrupted before
        commit, rerunning repeats extraction from intact raw data. An unchanged
        rerun leaves the existing dataset and training identity untouched.
    Examples:
        prepare_dataset('runs/data/simulation_checkpoint.npz',
                        'runs/data/training_dataset.npz')
    """
    source, destination = Path(simulation_path), Path(output_path)
    if source.resolve() == destination.resolve():
        raise ValueError("Prepared dataset must not overwrite raw simulations.")
    state = load_simulations(source)
    source_id = state.pop("checksum").item()
    metadata = json.loads(state["metadata"].item())
    metadata.update(stage="training_dataset", raw_simulation_id=source_id)
    state["metadata"] = np.asarray(canonical_json(metadata))
    if destination.exists():
        previous = load_dataset(destination)
        if previous["metadata"].item() == state["metadata"].item():
            print(f"Prepared dataset is unchanged: {destination}")
            return destination
    state["features"] = np.zeros((256, 8))
    state["reason"] = np.full(256, "pending", dtype="U32")
    for i in np.flatnonzero(state["completed"]):
        state["features"][i], state["reason"][i] = extract_features(state["frequencies"], state["co"][i])
    save_npz(destination, state)
    eligible = int((state["completed"] & (state["reason"] == "ok")).sum())
    print(f"Prepared {eligible} eligible rows from {state['completed'].sum()} saved simulations: {destination}")
    return destination


def load_dataset(path):
    """Load and semantically validate a reduced-family dataset checkpoint.

    Args:
        path (Path-like): Saved dataset checkpoint.
    Returns:
        dict: Arrays, descriptors, eligibility, and provenance.
    Raises:
        ValueError: If integrity, geometry plan, or spectra are inconsistent.
        FileNotFoundError: If the checkpoint does not exist.
    """
    state = load_npz(path)
    if not np.array_equal(state["codes"], all_codes()):
        raise ValueError("Geometry plan differs from the canonical 256 layouts.")
    n = len(state["frequencies"])
    if state["completed"].shape != (256,) or state["features"].shape != (256, 8):
        raise ValueError("Invalid dataset dimensions.")
    for key in ("co", "cross", "total"):
        if state[key].shape != (256, n):
            raise ValueError("Invalid spectrum dimensions.")
    metadata = json.loads(state["metadata"].item())
    p = metadata["physics"]
    expected_f = np.linspace(p["frequency_min_hz"], p["frequency_max_hz"], p["frequency_points"])
    if not np.array_equal(state["frequencies"], expected_f):
        raise ValueError("Saved frequency band differs from configuration.")
    for i in np.flatnonzero(state["completed"]):
        for key in ("co", "cross", "total"):
            validate_spectrum(state["frequencies"], state[key][i])
        features, reason = extract_features(state["frequencies"], state["co"][i])
        if reason != state["reason"][i] or not np.array_equal(features, state["features"][i]):
            raise ValueError("Saved descriptors differ from their spectra.")
    return state


def generate(directory, physics, backend="lumerical", limit=None, simulator=None):
    """Collect spectra and durably save each completed geometry before continuing.

    Args:
        directory (Path-like): Experiment data directory.
        physics (dict): Simulation configuration.
        backend (str): Explicit 'lumerical' or 'synthetic' provenance.
        limit (int or None): Maximum additional geometries for this invocation.
        simulator (callable or None): Injected solver callable for synthetic tests.
    Returns:
        Path: Resumable dataset checkpoint, independent of lumapi for training.
    Raises:
        ValueError: If settings differ from a saved run or results are invalid.
        RuntimeError: If simulation or physical preflight fails.
        OSError: If a checkpoint cannot be saved.
    Notes:
        A crash repeats only the active, uncommitted geometry. Physical runs
        always pass the reference and convergence gate first.
    Examples:
        generate('runs/data', PHYSICS, 'lumerical', limit=4)
    """
    if backend not in ("lumerical", "synthetic") or (limit is not None and limit < 1):
        raise ValueError("Invalid backend or generation limit.")
    if simulator is not None and backend != "synthetic":
        raise ValueError("Injected simulators are restricted to synthetic verification.")
    directory = Path(directory)
    path = directory / "dataset_checkpoint.npz"
    metadata = canonical_json({"schema": 1, "family": "mirrored-square-rings-v1", "backend": backend, "physics": physics})
    f = np.linspace(physics["frequency_min_hz"], physics["frequency_max_hz"], physics["frequency_points"])
    if path.exists():
        state = load_dataset(path)
        if state["metadata"].item() != metadata:
            raise ValueError("Generation configuration changed. Use a new data directory.")
    else:
        state = {"metadata": np.asarray(metadata), "codes": all_codes(), "frequencies": f,
                 "completed": np.zeros(256, dtype=bool), "features": np.zeros((256, 8)),
                 "reason": np.full(256, "pending", dtype="U32"), "seconds": np.zeros(256)}
        for key in ("co", "cross", "total"):
            state[key] = np.full((256, len(f)), np.nan)
        save_npz(path, state)
    pending = np.flatnonzero(~state["completed"])
    if limit is not None:
        pending = pending[:limit]
    if not len(pending):
        print("All requested geometries are already completed.")
        return path
    if backend == "synthetic":
        from synthetic import response

        print("SYNTHETIC SOFTWARE TEST: results have no electromagnetic meaning.")
        simulator = response if simulator is None else simulator
    else:
        from lumerical import LumericalSolver

        solver = LumericalSolver(physics)
        solver.preflight(directory)
        simulator = solver.simulate
    for i in pending:
        start = time.monotonic()
        result = simulator(state["codes"][i], f)
        for key in ("co", "cross", "total"):
            _, values = validate_spectrum(f, result[key])
            state[key][i] = values
        features, reason = extract_features(f, state["co"][i])
        state["features"][i] = features
        state["reason"][i] = reason
        state["seconds"][i] = time.monotonic() - start
        state["completed"][i] = True
        save_npz(path, state)
        print(f"Layout {i:03d}: {reason}, {state['completed'].sum()}/256 saved, {state['seconds'][i]:.2f} s")
    return path
