"""Persist console output, errors, and wall time for each stage invocation."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from contextvars import ContextVar
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
import time
import traceback

from metasurface.common.storage import atomic_write

_ACTIVE_PARTS = ContextVar("metasurface_active_parts", default=None)


class _Tee:
    """Write a console stream to its original target and a run log."""

    def __init__(self, original, log):
        self.original = original
        self.log = log

    def write(self, value):
        """Forward one write to both destinations.

        Args:
            value (str): Text to write.

        Returns:
            int: Number of characters written.
        """
        self.original.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        """Flush both output streams."""
        self.original.flush()
        self.log.flush()



@contextmanager
def timed_part(name):
    """Record one named part in the active stage status.

    Args:
        name (str): Part name shown in the status JSON and console log.

    Yields:
        None: Execute the timed part's body.
    """
    started = time.perf_counter()
    entry = {"name": name, "status": "completed"}
    try:
        yield
    except BaseException as error:
        entry["status"] = "failed"
        entry["error_type"] = type(error).__name__
        raise
    finally:
        entry["elapsed_seconds"] = time.perf_counter() - started
        parts = _ACTIVE_PARTS.get()
        if parts is not None:
            parts.append(entry)
        print(f"Part {name}: {entry['status']} after {entry['elapsed_seconds']:.3f}s", flush=True)

def run_stage(run_root, stage, action):
    """Run a stage with an append-only invocation log and atomic status file.

    Args:
        run_root (Path-like): Scenario output root under ``runs/``.
        stage (str): Stage name used for its log directory.
        action (callable): Zero-argument stage body.

    Returns:
        object: The action's return value.

    Raises:
        Exception: Re-raises action or persistence failures after logging.
    """
    root = Path(run_root)
    if stage in ("", ".", "..") or "/" in stage or "\\" in stage:
        raise ValueError("Stage name must be a simple directory name.")
    directory = root / "logs" / stage
    directory.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc)
    identifier = f"{started_at:%Y%m%dT%H%M%S%fZ}-{os.getpid()}"
    log_path = directory / f"{identifier}.log"
    status_path = directory / f"{identifier}.json"
    started = time.perf_counter()
    status = {"stage": stage, "started_at_utc": started_at.isoformat(),
              "status": "running", "log": str(log_path.relative_to(root)),
              "parts": []}
    with log_path.open("w", encoding="utf-8", buffering=1) as log:
        with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
            print(f"Stage {stage}: started {status['started_at_utc']}", flush=True)
            token = _ACTIVE_PARTS.set(status["parts"])
            try:
                result = action()
            except BaseException as error:
                status["status"] = "failed"
                status["error_type"] = type(error).__name__
                status["error_message"] = str(error)
                traceback.print_exc()
                raise
            else:
                status["status"] = "completed"
                return result
            finally:
                _ACTIVE_PARTS.reset(token)
                status["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
                status["elapsed_seconds"] = time.perf_counter() - started
                print(f"Stage {stage}: {status['status']} after {status['elapsed_seconds']:.3f}s", flush=True)
                atomic_write(status_path, lambda stream: stream.write(
                    json.dumps(status, indent=2).encode("utf-8")))


def write_training_metrics(checkpoint_path):
    """Export checkpointed training metrics as readable JSON beside the model.

    Args:
        checkpoint_path (Path-like): Training ``last.pt`` checkpoint.

    Returns:
        Path: Atomic ``training_metrics.json`` output.

    Raises:
        ValueError: If the checkpoint is invalid.
    """
    from metasurface.reduced.learning import load_checkpoint

    path = Path(checkpoint_path)
    state = load_checkpoint(path)
    batches = state.get("batch_history", [])
    best_loss = state.get("best_loss")
    record = {"backend": state["metadata"]["backend"],
              "dataset_id": state["dataset_id"],
              "options": json.loads(state["options_json"]),
              "split_sizes": {name: len(rows) for name, rows in state["splits"].items()},
              "steps": state["step"], "completed_epochs": state["epoch"],
              "finished": state["finished"],
              "best_validation_loss": best_loss if best_loss != float("inf") else None,
              "minibatch_history_available": len(batches) == state["step"],
              "minibatch_compute_seconds": sum(row["compute_seconds"] for row in batches),
              "minibatches": batches,
              "epochs": state["history"]}
    destination = path.parent / "training_metrics.json"
    atomic_write(destination, lambda stream: stream.write(json.dumps(record, indent=2).encode("utf-8")))
    return destination


def _save_json(path, value):
    """Atomically save a JSON-compatible result record.

    Args:
        path (Path-like): Output path.
        value (dict): Serializable metrics.

    Returns:
        Path: Saved path.
    """
    destination = Path(path)
    atomic_write(destination, lambda stream: stream.write(json.dumps(value, indent=2).encode("utf-8")))
    return destination


def write_geometry_metrics(checkpoint_path):
    """Summarize committed geometries and available per-layout runtimes.

    Args:
        checkpoint_path (Path-like): Geometry NPZ checkpoint.

    Returns:
        Path: Geometry metrics JSON under the same run directory.
    """
    from metasurface.common.storage import load_npz

    path = Path(checkpoint_path)
    state = load_npz(path)
    seconds = state.get("seconds")
    completed = state["completed"]
    record = {"geometry_id": state["checksum"].item(),
              "metadata": json.loads(state["metadata"].item()),
              "completed": int(completed.sum()), "total": len(completed),
              "per_layout_seconds": [float(seconds[i]) if completed[i] and seconds is not None and seconds[i] > 0 else None
                                     for i in range(len(completed))]}
    record["measured_layouts"] = sum(value is not None for value in record["per_layout_seconds"])
    record["measured_seconds"] = sum(value for value in record["per_layout_seconds"] if value is not None)
    return _save_json(path.parent / "geometry_metrics.json", record)


def write_simulation_metrics(checkpoint_path):
    """Summarize saved raw spectra and per-layout solver runtimes.

    Args:
        checkpoint_path (Path-like): Raw simulation NPZ checkpoint.

    Returns:
        Path: Simulation metrics JSON under the same run directory.
    """
    from metasurface.common.storage import load_npz

    path = Path(checkpoint_path)
    state = load_npz(path)
    completed = state["completed"]
    seconds = state["seconds"]
    record = {"metadata": json.loads(state["metadata"].item()),
              "completed": int(completed.sum()), "total": len(completed),
              "total_layout_seconds": float(seconds[completed].sum()),
              "per_layout_seconds": [float(seconds[i]) if completed[i] else None
                                     for i in range(len(completed))]}
    return _save_json(path.parent / "simulation_metrics.json", record)


def write_dataset_metrics(checkpoint_path):
    """Summarize prepared eligibility and reasons without hiding raw rows.

    Args:
        checkpoint_path (Path-like): Prepared dataset NPZ checkpoint.

    Returns:
        Path: Dataset metrics JSON under the same run directory.
    """
    from collections import Counter

    from metasurface.common.storage import load_npz

    path = Path(checkpoint_path)
    state = load_npz(path)
    completed = state["completed"]
    reasons = Counter(str(value) for value in state["reason"][completed])
    record = {"dataset_id": state["checksum"].item(),
              "metadata": json.loads(state["metadata"].item()),
              "completed": int(completed.sum()), "total": len(completed),
              "eligible": reasons.get("ok", 0), "reasons": dict(sorted(reasons.items()))}
    return _save_json(path.parent / "dataset_metrics.json", record)
