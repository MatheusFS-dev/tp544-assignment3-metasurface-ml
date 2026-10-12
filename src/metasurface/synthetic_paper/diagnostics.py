"""Truthful dataset diagnostics for the synthetic paper surrogate."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from metasurface.paper.dataset import load_dataset


def write_diagnostics(dataset_path, output_dir):
    """Write summary diagnostics without implying physical validation.

    Args:
        dataset_path (Path-like): Prepared synthetic dataset.
        output_dir (Path-like): Destination analysis directory.

    Returns:
        Path: JSON summary path.

    Raises:
        ValueError: If the dataset provenance is not synthetic.
    """
    state = load_dataset(dataset_path)
    metadata = json.loads(state["metadata"].item())
    if metadata.get("backend") != "synthetic_surrogate":
        raise ValueError("Diagnostics require synthetic_surrogate provenance.")
    completed = np.flatnonzero(state["completed"])
    reasons, reason_counts = np.unique(state["reason"][completed], return_counts=True)
    violations = 0
    for key in ("co", "cross", "total"):
        values = state[key][completed]
        violations += int((~np.isfinite(values) | (values < 0) | (values > 1)).sum())
    violations += int((~np.isclose(state["total"][completed],
                                   state["co"][completed] + state["cross"][completed])).sum())
    spectrum_hashes = [hashlib.sha256(state["total"][i].tobytes()).hexdigest() for i in completed]
    feature_hashes = [hashlib.sha256(state["features"][i].tobytes()).hexdigest() for i in completed]
    report = {"scenario": "synthetic_surrogate", "physical_validation": "not_run",
              "completed": int(len(completed)),
              "eligible": int(np.sum(state["reason"][completed] == "ok")),
              "reasons": {str(key): int(value) for key, value in zip(reasons, reason_counts)},
              "passivity_violations": violations,
              "duplicate_spectra": len(spectrum_hashes) - len(set(spectrum_hashes)),
              "duplicate_feature_vectors": len(feature_hashes) - len(set(feature_hashes)),
              "tile_counts": np.bincount(state["codes"][completed].ravel(), minlength=8).tolist()}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "synthetic_dataset_diagnostics.json"
    path.write_text(json.dumps(report, indent=2))
    with (output / "synthetic_dataset_summary.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["metric", "value"])
        for key in ("completed", "eligible", "passivity_violations",
                    "duplicate_spectra", "duplicate_feature_vectors"):
            writer.writerow([key, report[key]])
        for reason, count in report["reasons"].items():
            writer.writerow([f"reason_{reason}", count])
    return path
