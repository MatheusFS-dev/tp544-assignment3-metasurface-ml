"""Held-out evaluation and export of exactly one predicted geometry."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from metasurface.reduced.dataset import load_dataset
from metasurface.reduced.geometry import pixel_layout, rectangles
from metasurface.reduced.learning import InverseNet, load_checkpoint, metrics
from metasurface.reduced.lumerical import LumericalSolver, project_script
from metasurface.reduced.spectra import extract_features
from metasurface.common.storage import canonical_json


def target_features(notches, physics):
    """Validate and encode one or two user-specified -10 dB notch targets.

    Args:
        notches (list[tuple]): Frequency GHz, depth dB, and width GHz per notch.
        physics (dict): Simulation band and sampling configuration.
    Returns:
        ndarray: Eight ordered descriptors including presence flags.
    Raises:
        ValueError: If targets are invalid, unresolved, overlapping, or outside band.
    """
    values = np.asarray(notches, dtype=float)
    if values.shape not in ((1, 3), (2, 3)) or not np.isfinite(values).all():
        raise ValueError("Specify one or two finite triples: GHz, dB, GHz.")
    values = values[np.argsort(values[:, 0])]
    low, high = physics["frequency_min_hz"] / 1e9, physics["frequency_max_hz"] / 1e9
    minimum_width = 3 * (high - low) / (physics["frequency_points"] - 1)
    features = np.zeros(8)
    for i, (frequency, depth, width) in enumerate(values):
        if depth > -10.5 or width < minimum_width or frequency - width / 2 <= low or frequency + width / 2 >= high:
            raise ValueError("Target must contain a resolved notch fully inside the frequency band.")
        if i and values[i - 1, 0] + values[i - 1, 2] / 2 >= frequency - width / 2:
            raise ValueError("Target notch intervals must not overlap.")
        features[i * 4:i * 4 + 4] = [frequency, depth, width, 1]
    return features


def physical_errors(target, observed, reason):
    """Report descriptor errors only when the observed notch count is acceptable.

    Args:
        target (array-like): Eight requested descriptors.
        observed (array-like): Eight simulated descriptors.
        reason (str): Eligibility reason from extraction.
    Returns:
        dict: Notch-count agreement and absolute errors, or failure reason.
    Raises:
        None.
    """
    target = np.asarray(target).reshape(2, 4)
    observed = np.asarray(observed).reshape(2, 4)
    matches = reason == "ok" and np.array_equal(target[:, 3], observed[:, 3])
    if not matches:
        return {"notch_count_matches": False, "reason": reason}
    delta = abs(target[:, :3] - observed[:, :3])[target[:, 3] == 1]
    return {"notch_count_matches": True, "frequency_error_ghz": delta[:, 0].tolist(),
            "depth_error_db": delta[:, 1].tolist(), "width_error_ghz": delta[:, 2].tolist()}


def evaluate(checkpoint_path, dataset_path, output_dir, target=None, run_lumerical=False):
    """Evaluate saved best weights and export one NN-selected Lumerical layout.

    Args:
        checkpoint_path (Path-like): Training last.pt with best validation weights.
        dataset_path (Path-like): Unchanged generated dataset.
        output_dir (Path-like): Directory for one design, reports, and plots.
        target (list[tuple] or None): Optional custom notch specifications. Otherwise
            use the first held-out test row, without selecting by performance.
        run_lumerical (bool): Re-simulate this one exported NN geometry if true.
    Returns:
        dict: Test/baseline metrics, provenance, selected codes, and validation status.
    Raises:
        ValueError: If dataset identity, checkpoint, or target is invalid.
        RuntimeError: If requested real solver validation fails.
        OSError: If export cannot be written.
    Notes:
        Cached lookup is labeled separately from a fresh physical run. A
        synthetic checkpoint always remains synthetic, even when exported.
    Examples:
        evaluate('runs/training/last.pt', 'runs/data/dataset_checkpoint.npz', 'runs/design')
    """
    print("Export: loading checkpoint and dataset", flush=True)
    state = load_checkpoint(checkpoint_path)
    data = load_dataset(dataset_path)
    if state["dataset_id"] != data["checksum"].item() or state["best_model"] is None:
        raise ValueError("Checkpoint dataset mismatch or no validated best model.")
    model = InverseNet()
    model.load_state_dict(state["best_model"])
    model.eval()
    mean, std = state["mean"], state["std"]
    x = (torch.tensor(data["features"], dtype=torch.float32) - mean) / std
    labels = torch.tensor(data["codes"], dtype=torch.long)
    test_ids, train_ids = state["splits"]["test"], state["splits"]["train"]
    report = {"backend": state["metadata"]["backend"], "physical_validation": "not_run",
              "test": metrics(model, x, labels, test_ids), "test_sample_count": len(test_ids),
              "training_complete": state["finished"], "checkpoint_sha256": hashlib.sha256(Path(checkpoint_path).read_bytes()).hexdigest()}
    distances = torch.cdist(x[test_ids], x[train_ids])
    nearest = labels[train_ids][distances.argmin(dim=1)]
    truth = labels[test_ids]
    report["nearest_neighbor"] = {"position_accuracy": (nearest == truth).float().mean().item(),
                                   "exact_accuracy": (nearest == truth).all(dim=1).float().mean().item()}
    physics = state["metadata"]["physics"]
    requested = data["features"][test_ids[0]].copy() if target is None else target_features(target, physics)
    scaled = (torch.tensor(requested, dtype=torch.float32) - mean) / std
    with torch.no_grad():
        probabilities = model(scaled.reshape(1, 8)).softmax(dim=-1)[0]
        codes = probabilities.argmax(dim=-1).numpy()
    report["target_features"] = requested.tolist()
    report["target_source"] = f"held_out_row_{test_ids[0]}" if target is None else "user_specified"
    report["predicted_codes"] = codes.tolist()
    report["class_probabilities"] = probabilities.tolist()
    report["out_of_training_range"] = bool(np.any(requested < data["features"][train_ids].min(axis=0)) or np.any(requested > data["features"][train_ids].max(axis=0)))
    geometry_id = int(codes @ np.array([64, 16, 4, 1]))
    if data["completed"][geometry_id]:
        report["cached_descriptor_comparison"] = physical_errors(requested, data["features"][geometry_id], data["reason"][geometry_id])
        report["cached_spectrum_provenance"] = report["backend"]
    print("Export: writing predicted design and comparisons", flush=True)
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    # Invalidate prior physical artifacts before writing a different target.
    for name in ("design.fsp", "design_validation_spectrum.csv", "report.json", "spectrum_comparison.png"):
        (directory / name).unlink(missing_ok=True)
    mask = pixel_layout(codes)
    np.savetxt(directory / "design_pixels.csv", mask, delimiter=",", fmt="%d")
    with (directory / "design_rectangles_m.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x_center_m", "y_center_m", "x_span_m", "y_span_m", "z_min_m", "z_max_m"])
        for row in rectangles(codes, physics["pixel_m"]):
            writer.writerow([*row, 0, physics["metal_thickness_m"]])
    script = project_script(codes, physics) + '\n# Run this script from the export directory, then inspect the saved project.\nsave("design.fsp");\n'
    (directory / "build_design.lsf").write_text(script)
    spec = {"pattern_codes": codes.tolist(), "physics": physics, "pixel_csv_row_direction": "increasing_y",
            "layers": {"substrate_z_m": [-physics["substrate_thickness_m"], 0],
                       "ground_z_m": [-physics["substrate_thickness_m"] - physics["metal_thickness_m"], -physics["substrate_thickness_m"]],
                       "pattern_z_m": [0, physics["metal_thickness_m"]]},
            "training_backend": report["backend"], "target": requested.tolist(), "pattern_names": ["empty", "small_square_ring", "large_square_ring", "thick_square_ring"]}
    (directory / "design_spec.json").write_text(json.dumps(spec, indent=2))
    fig, ax = plt.subplots(figsize=(5, 5))
    half_cell_mm = 16 * physics["pixel_m"] * 1e3
    ax.imshow(mask, origin="lower", cmap="Greys", interpolation="nearest",
              extent=[-half_cell_mm, half_cell_mm, -half_cell_mm, half_cell_mm])
    ax.set(xlabel="x (mm)", ylabel="y (mm)", title=f"NN geometry A,B,C,D = {codes.tolist()}")
    fig.tight_layout()
    fig.savefig(directory / "design_preview.png", dpi=180)
    plt.close(fig)
    if data["completed"][geometry_id]:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(data["frequencies"] / 1e9, 10 * np.log10(np.maximum(data["co"][geometry_id], 1e-12)),
                label=f"Cached NN layout ({report['backend']})")
        active = requested.reshape(2, 4)
        active = active[active[:, 3] == 1]
        ax.scatter(active[:, 0], active[:, 1], marker="x", color="red", label="Target minima")
        ax.axhline(-10, color="grey", linestyle="--", linewidth=0.8)
        ax.set(xlabel="Frequency (GHz)", ylabel="Co-polar reflection (dB)",
               title="Cached response comparison; no fresh physical validation")
        ax.legend()
        fig.tight_layout()
        fig.savefig(directory / "spectrum_comparison.png", dpi=160)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    history = state["history"]
    ax.plot([r["epoch"] for r in history], [r["train_loss"] for r in history], label="Training")
    ax.plot([r["epoch"] for r in history], [r["loss"] for r in history], label="Validation")
    ax.set(xlabel="Epoch", ylabel="Cross-entropy loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(directory / "training_loss.png", dpi=160)
    plt.close(fig)
    if run_lumerical:
        print("Export: running Lumerical preflight and fresh validation", flush=True)
        solver = LumericalSolver(physics)
        solver.preflight(Path(dataset_path).parent)
        raw = solver._run(codes, project_path=directory / "design.fsp")
        power = solver._powers(raw)
        observed, reason = extract_features(data["frequencies"], power["co"])
        np.savetxt(directory / "design_validation_spectrum.csv", np.column_stack([data["frequencies"], power["co"], power["cross"], power["total"]]),
                   delimiter=",", header="frequency_hz,R_co,R_cross,R_total", comments="")
        report["physical_validation"] = "fresh_lumerical_run"
        report["physical_errors"] = physical_errors(requested, observed, reason)
    (directory / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Export: report saved to {directory / 'report.json'}", flush=True)
    print(json.dumps(report, indent=2))
    return report
