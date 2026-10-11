"""Offline generation of durable, explicitly saved metasurface masks."""

import csv
import io
import json
from pathlib import Path

import numpy as np

from geometry import all_codes, pixel_layout
from storage import atomic_write, canonical_json, load_npz, save_npz


def load_geometries(path, require_complete=True):
    """Validate saved geometry codes and masks before simulation consumes them.

    Args:
        path (Path-like): Geometry checkpoint created by stage one.
        require_complete (bool): Require every candidate to be committed.
    Returns:
        dict: Checked codes, pixel masks, completion flags and geometry metadata.
    Raises:
        FileNotFoundError: If the checkpoint does not exist.
        ValueError: If its content, scale, family or completion is invalid.
    Notes:
        Validation recomputes masks only to detect encoding disagreements.
        Stage two does not invent or choose additional geometry candidates.
    """
    state = load_npz(path)
    metadata = json.loads(state["metadata"].item())
    if (metadata.get("stage") != "geometries" or metadata.get("schema") != 1
            or metadata.get("family") != "mirrored-square-rings-v1"
            or not np.isfinite(metadata["pixel_m"]) or metadata["pixel_m"] <= 0):
        raise ValueError("Invalid geometry configuration.")
    if (not np.array_equal(state["codes"], all_codes())
            or state["pixels"].shape != (256, 32, 32)
            or state["pixels"].dtype != np.uint8
            or state["completed"].shape != (256,)
            or state["completed"].dtype != np.bool_):
        raise ValueError("Invalid geometry checkpoint dimensions or codes.")
    for i in np.flatnonzero(state["completed"]):
        if not np.array_equal(state["pixels"][i], pixel_layout(state["codes"][i])):
            raise ValueError("Saved pixels differ from their pattern codes.")
    if require_complete and not state["completed"].all():
        raise ValueError("Geometry generation must be complete before simulation.")
    return state


def generate_geometries(directory, pixel_m, limit=None):
    """Generate all 256 masks offline, saving progress after every layout.

    Args:
        directory (Path-like): Geometry checkpoint and code-table directory.
        pixel_m (float): Positive pixel side in metres.
        limit (int or None): Maximum additional layouts, for recovery probes.
    Returns:
        Path: Geometry checkpoint, complete unless a limit stopped generation.
    Raises:
        ValueError: If scale, limit or resumed configuration is invalid.
        OSError: If durable output cannot be committed.
    Notes:
        No electromagnetic response is computed here. A crash repeats at most
        the active uncommitted layout. One writer must own this directory.
    Examples:
        generate_geometries('runs/geometries', 0.2e-3)
    """
    if not np.isfinite(pixel_m) or pixel_m <= 0:
        raise ValueError("Pixel length must be positive.")
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit < 1):
        raise ValueError("Geometry limit must be a positive integer or None.")
    path = Path(directory) / "geometry_checkpoint.npz"
    metadata = canonical_json({"schema": 1, "stage": "geometries",
                               "family": "mirrored-square-rings-v1", "pixel_m": float(pixel_m)})
    if path.exists():
        state = load_geometries(path, require_complete=False)
        if state["metadata"].item() != metadata:
            raise ValueError("Geometry configuration changed. Use a new directory.")
    else:
        state = {"metadata": np.asarray(metadata), "codes": all_codes(),
                 "pixels": np.zeros((256, 32, 32), dtype=np.uint8),
                 "completed": np.zeros(256, dtype=bool)}
        save_npz(path, state)
    pending = np.flatnonzero(~state["completed"])
    if limit is not None:
        pending = pending[:limit]
    for i in pending:
        state["pixels"][i] = pixel_layout(state["codes"][i])
        state["completed"][i] = True
        save_npz(path, state)
    if state["completed"].all():
        table = io.StringIO(newline="")
        writer = csv.writer(table, lineterminator="\n")
        writer.writerow(["geometry_id", "A", "B", "C", "D"])
        for i, codes in enumerate(state["codes"]):
            writer.writerow([i, *codes])
        payload = table.getvalue().encode("utf-8")
        destination = path.parent / "geometry_codes.csv"
        if not destination.exists() or destination.read_bytes() != payload:
            atomic_write(destination, lambda stream: stream.write(payload))
    print(f"Geometries saved: {state['completed'].sum()}/256. No solver was used. {path}")
    return path
