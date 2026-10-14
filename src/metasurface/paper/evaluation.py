"""Held-out paper-scale restricted evaluation and one-design export."""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from metasurface.paper.dataset import load_dataset
from metasurface.paper.features import extract_features
from metasurface.paper.geometry import masks, pixel_layout, rectangles
from metasurface.paper.learning import RestrictedNet, decode_bits, encode_bits
from metasurface.reduced.learning import load_checkpoint
from metasurface.reduced.lumerical import LumericalSolver, project_script


def _notch_errors(target, observed):
    """Compare matched notch triples without inventing missing notch errors.

    Args:
        target (array-like): Target 24 descriptors.
        observed (array-like): Observed 24 descriptors.

    Returns:
        dict: Counts and absolute error arrays when counts match.
    """
    wanted = np.asarray(target).reshape(8, 3)
    actual = np.asarray(observed).reshape(8, 3)
    wanted = wanted[wanted[:, 0] > 0]
    actual = actual[actual[:, 0] > 0]
    result = {"target_count": len(wanted), "observed_count": len(actual)}
    if len(wanted) == len(actual):
        result.update(frequency_error_ghz=abs(wanted[:, 0] - actual[:, 0]).tolist(),
                      depth_error_db=abs(wanted[:, 1] - actual[:, 1]).tolist(),
                      width_error_ghz=abs(wanted[:, 2] - actual[:, 2]).tolist())
    return result


def evaluate(checkpoint_path, dataset_path, output_dir, run_lumerical=False,
             preflight_mode="full"):
    """Score held-out layouts and export one paper-scale predicted design.

    Args:
        checkpoint_path (Path-like): Finished or resumable paper checkpoint.
        dataset_path (Path-like): Unchanged prepared paper dataset.
        output_dir (Path-like): Independent paper export directory.
        run_lumerical (bool): Run fresh physical validation if explicitly enabled.
        preflight_mode (str): ``full`` or reference-only solver preflight.

    Returns:
        dict: Labeled held-out and physical comparison results.

    Raises:
        ValueError: If dataset identity or provenance is invalid.
    """
    if preflight_mode not in ("full", "reference_only"):
        raise ValueError("Invalid preflight mode.")
    state = load_checkpoint(checkpoint_path)
    data = load_dataset(dataset_path)
    if state["dataset_id"] != data["checksum"].item() or "model" not in state:
        raise ValueError("Paper checkpoint and dataset differ.")
    if run_lumerical and state["metadata"]["backend"] != "lumerical":
        raise ValueError("Test fixture predictions cannot be physically validated as paper results.")
    model = RestrictedNet(json.loads(state["options_json"])["dropout"])
    model.load_state_dict(state["model"])
    model.eval()
    x = (torch.tensor(data["features"], dtype=torch.float32) - state["mean"]) / state["std"]
    test_ids = state["splits"]["test"]
    with torch.no_grad():
        outputs = model(x[test_ids])
        predicted = decode_bits(outputs).numpy()
    truth = data["codes"][test_ids]
    bits = encode_bits(truth)
    report = {"scenario": "paper_scale_approximation", "backend": state["metadata"]["backend"],
              "physical_validation": "not_run", "physical_preflight_mode": preflight_mode,
              "training_complete": state["finished"],
              "test_sample_count": len(test_ids),
              "bit_accuracy": float(((outputs >= 0.5) == bits.bool()).float().mean()),
              "tile_accuracy": float((predicted == truth).mean()),
              "exact_layout_accuracy": float((predicted == truth).all(axis=1).mean())}
    target_id = test_ids[0]
    with torch.no_grad():
        codes = decode_bits(model(x[[target_id]]))[0].numpy()
    report["target_row"] = target_id
    report["predicted_codes"] = codes.tolist()
    observed_id = next((i for i, row in enumerate(data["codes"]) if np.array_equal(row, codes)), None)
    if observed_id is not None and data["completed"][observed_id]:
        report["cached_descriptor_comparison"] = _notch_errors(data["features"][target_id], data["features"][observed_id])
        report["cached_spectrum_provenance"] = report["backend"]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in ("design.fsp", "design_validation_spectrum.csv", "report.json"):
        (output / name).unlink(missing_ok=True)
    mask = pixel_layout(codes)
    np.savetxt(output / "design_pixels.csv", mask, delimiter=",", fmt="%d")
    with (output / "design_rectangles_m.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x_center_m", "y_center_m", "x_span_m", "y_span_m", "z_min_m", "z_max_m"])
        for row in rectangles(codes, state["metadata"]["physics"]["pixel_m"]):
            writer.writerow([*row, 0, state["metadata"]["physics"]["metal_thickness_m"]])
    physics = state["metadata"]["physics"]
    (output / "build_design.lsf").write_text(project_script(codes, physics, rectangle_fn=rectangles) + '\nsave("design.fsp");\n')
    (output / "design_spec.json").write_text(json.dumps({"scenario": report["scenario"],
        "backend": report["backend"], "pattern_codes": codes.tolist(), "tile_masks": masks().tolist(),
        "physics": physics, "target_features": data["features"][target_id].tolist()}, indent=2))
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mask, origin="lower", cmap="Greys", interpolation="nearest")
    ax.set(title="Paper-scale approximation: predicted copper mask", xlabel="x pixel", ylabel="y pixel")
    fig.tight_layout()
    fig.savefig(output / "design_preview.png", dpi=180)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot([row["epoch"] for row in state["history"]], [row["train_loss"] for row in state["history"]])
    ax.set(xlabel="Epoch", ylabel="Training MSE")
    fig.tight_layout()
    fig.savefig(output / "training_loss.png", dpi=160)
    plt.close(fig)
    if run_lumerical:
        solver = LumericalSolver(physics, rectangle_fn=rectangles, code_count=16)
        solver.preflight(Path(dataset_path).parent, mode=preflight_mode)
        raw = solver._run(codes, project_path=output / "design.fsp")
        power = solver._powers(raw)
        features, reason = extract_features(data["frequencies"], power["co"])
        np.savetxt(output / "design_validation_spectrum.csv",
                   np.column_stack([data["frequencies"], power["co"], power["cross"], power["total"]]),
                   delimiter=",", header="frequency_hz,R_co,R_cross,R_total", comments="")
        report["physical_validation"] = "fresh_lumerical_run"
        report["physical_reason"] = reason
        report["physical_errors"] = _notch_errors(data["features"][target_id], features)
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    return report
