"""Independent, resumable paper-scale geometry manifest."""

from pathlib import Path
import time

import numpy as np

from metasurface.common.progress import Progress
from metasurface.common.storage import canonical_json, load_npz, save_npz
from metasurface.paper.geometry import masks, pixel_layout, sample_codes


def load_manifest(path, require_complete=True):
    """Load and validate saved 16-choice layouts.

    Args:
        path (Path-like): Manifest checkpoint.
        require_complete (bool): Require all sampled layouts to be saved.

    Returns:
        dict: Verified arrays and checksum.

    Raises:
        ValueError: If layout geometry or metadata are inconsistent.
    """
    state = load_npz(path)
    if (state["metadata"].item() != canonical_json({"schema": 1, "family": "paper-scale-rings-v1",
                                                      "count": len(state["codes"]), "seed": 2026})
            or state["codes"].shape != (len(state["codes"]), 16)
            or state["pixels"].shape != (len(state["codes"]), 32, 32)
            or state["completed"].shape != (len(state["codes"]),)
            or not np.array_equal(state["tiles"], masks())
            or not np.array_equal(state["codes"], sample_codes(len(state["codes"])))):
        raise ValueError("Paper geometry manifest identity mismatch.")
    if "seconds" in state and (state["seconds"].shape != (len(state["codes"]),) or
                               not np.isfinite(state["seconds"]).all() or
                               np.any(state["seconds"] < 0)):
        raise ValueError("Invalid paper geometry runtimes.")
    for i in np.flatnonzero(state["completed"]):
        if not np.array_equal(state["pixels"][i], pixel_layout(state["codes"][i])):
            raise ValueError("Saved paper mask differs from tile codes.")
    if require_complete and not state["completed"].all():
        raise ValueError("Paper geometry manifest is incomplete.")
    return state


def generate(directory, count=2000, limit=None):
    """Generate sampled paper-scale layouts with one commit per geometry.

    Args:
        directory (Path-like): Paper geometry output directory.
        count (int): Number of sampled layouts.
        limit (int or None): Additional layouts to save this invocation.

    Returns:
        Path: Manifest checkpoint path.

    Raises:
        ValueError: If a saved experiment differs from requested settings.
    """
    path = Path(directory) / "geometry_checkpoint.npz"
    if path.exists():
        state = load_manifest(path, require_complete=False)
        if len(state["codes"]) != count:
            raise ValueError("Paper layout count changed. Use a new directory.")
    else:
        state = {"metadata": np.asarray(canonical_json({"schema": 1, "family": "paper-scale-rings-v1",
                                                         "count": count, "seed": 2026})),
                 "tiles": masks(), "codes": sample_codes(count),
                 "pixels": np.zeros((count, 32, 32), dtype=np.uint8),
                 "completed": np.zeros(count, dtype=bool),
                 "seconds": np.zeros(count)}
        save_npz(path, state)
    progress = Progress("Paper geometries", count, int(state["completed"].sum()))
    pending = np.flatnonzero(~state["completed"])
    if limit is not None:
        pending = pending[:limit]
    if len(pending) and "seconds" not in state:
        state["seconds"] = np.zeros(count)
    for i in pending:
        started = time.perf_counter()
        state["pixels"][i] = pixel_layout(state["codes"][i])
        state["seconds"][i] = time.perf_counter() - started
        state["completed"][i] = True
        save_npz(path, state)
        progress.committed(int(state["completed"].sum()), f"layout {i:04d}")
    return path
