"""Optional Lumerical validation of one exported synthetic-paper design."""

import hashlib
import json
from pathlib import Path

import numpy as np

from metasurface.common.storage import canonical_json
from metasurface.paper.features import extract_features


def _digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _notch_errors(target, observed):
    wanted = np.asarray(target).reshape(8, 3)
    actual = np.asarray(observed).reshape(8, 3)
    wanted = wanted[wanted[:, 0] > 0]
    actual = actual[actual[:, 0] > 0]
    result = {"target_count": len(wanted), "observed_count": len(actual)}
    if len(wanted) == len(actual):
        result.update(frequency_error_ghz=np.abs(wanted[:, 0] - actual[:, 0]).tolist(),
                      depth_error_db=np.abs(wanted[:, 1] - actual[:, 1]).tolist(),
                      width_error_ghz=np.abs(wanted[:, 2] - actual[:, 2]).tolist())
    return result


def validate_export(design_dir, output_dir=None, solver_factory=None):
    """Run a fresh solver check for an unchanged exported geometry.

    Args:
        design_dir (Path-like): Directory created by synthetic evaluation.
        output_dir (Path-like or None): Validation destination, defaulting to design directory.
        solver_factory (callable or None): Injectable test factory receiving physics.

    Returns:
        dict: Provenance-separated physical comparison report.

    Raises:
        ValueError: If export provenance, identity, or artifacts changed.
    """
    design = Path(design_dir)
    spec = json.loads((design / "design_spec.json").read_text())
    if spec.get("backend") != "synthetic_surrogate":
        raise ValueError("Validation requires a synthetic_surrogate export.")
    checksum = spec.pop("design_checksum", None)
    if checksum != hashlib.sha256(canonical_json(spec).encode()).hexdigest():
        raise ValueError("Design checksum mismatch.")
    for name, expected in spec["artifact_checksums"].items():
        if _digest(design / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
    physics = spec["physics"]
    codes = np.asarray(spec["pattern_codes"], dtype=np.uint8)
    destination = Path(output_dir) if output_dir is not None else design
    destination.mkdir(parents=True, exist_ok=True)
    if solver_factory is None:
        from metasurface.paper.geometry import rectangles
        from metasurface.reduced.lumerical import LumericalSolver
        solver = LumericalSolver(physics, rectangle_fn=rectangles, code_count=16)
    else:
        solver = solver_factory(physics)
    solver.preflight(destination)
    raw = solver._run(codes, project_path=destination / "design.fsp")
    powers = solver._powers(raw)
    frequencies = np.linspace(physics["frequency_min_hz"], physics["frequency_max_hz"],
                              physics["frequency_points"])
    physical_features, reason = extract_features(frequencies, powers["co"])
    np.savetxt(destination / "design_validation_spectrum.csv",
               np.column_stack([frequencies, powers["co"], powers["cross"], powers["total"]]),
               delimiter=",", header="frequency_hz,R_co,R_cross,R_total", comments="")
    report = {"scenario": "synthetic_surrogate_validated_once",
              "synthetic_backend": "synthetic_surrogate",
              "physical_validation": "fresh_lumerical_run", "physical_reason": reason,
              "design_checksum": checksum,
              "target_to_physical_errors": _notch_errors(spec["target_features"], physical_features),
              "physical_errors": _notch_errors(spec["predicted_features"], physical_features)}
    (destination / "validation_report.json").write_text(json.dumps(report, indent=2))
    return report
