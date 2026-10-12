"""Held-out evaluation and export for the synthetic paper workflow."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from metasurface.common.storage import canonical_json
from metasurface.paper.dataset import load_dataset
from metasurface.paper.features import extract_features
from metasurface.paper.geometry import masks, pixel_layout, rectangles
from metasurface.paper.learning import RestrictedNet, decode_bits, encode_bits
from metasurface.reduced.learning import load_checkpoint
from metasurface.reduced.lumerical import project_script
from metasurface.synthetic_paper.surrogate import response


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(checkpoint_path, dataset_path, output_dir):
    """Evaluate synthetic held-out data and export one predicted design.

    Args:
        checkpoint_path (Path-like): Restricted-network checkpoint.
        dataset_path (Path-like): Prepared synthetic dataset.
        output_dir (Path-like): Design export directory.

    Returns:
        dict: Synthetic-only evaluation report.

    Raises:
        ValueError: If checkpoint, dataset, or provenance differs.
    """
    state = load_checkpoint(checkpoint_path)
    data = load_dataset(dataset_path)
    metadata = json.loads(data["metadata"].item())
    if metadata.get("backend") != "synthetic_surrogate":
        raise ValueError("Synthetic evaluation requires synthetic_surrogate data.")
    if state["dataset_id"] != data["checksum"].item() or "model" not in state:
        raise ValueError("Checkpoint and synthetic dataset differ.")
    options = json.loads(state["options_json"])
    model = RestrictedNet(options["dropout"])
    model.load_state_dict(state["model"])
    model.eval()
    x = (torch.tensor(data["features"], dtype=torch.float32) - state["mean"]) / state["std"]
    test_ids = state["splits"]["test"]
    with torch.no_grad():
        outputs = model(x[test_ids])
    predicted = decode_bits(outputs).numpy()
    truth = data["codes"][test_ids]
    targets = encode_bits(truth)
    confusion = np.zeros((8, 8), dtype=int)
    for actual, guess in zip(truth.ravel(), predicted.ravel()):
        confusion[int(actual), int(guess)] += 1
    report = {"scenario": "synthetic_surrogate", "backend": "synthetic_surrogate",
              "physical_validation": "not_run", "training_complete": bool(state["finished"]),
              "test_sample_count": len(test_ids),
              "test_mse": float(torch.nn.functional.mse_loss(outputs, targets)),
              "bit_accuracy": float(((outputs >= 0.5) == targets.bool()).float().mean()),
              "tile_accuracy": float((predicted == truth).mean()),
              "exact_layout_accuracy": float((predicted == truth).all(axis=1).mean()),
              "confusion_matrix": confusion.tolist()}
    target_id = int(test_ids[0])
    codes = predicted[0]
    frequencies = data["frequencies"]
    predicted_spectrum = response(codes, frequencies)
    predicted_features, predicted_reason = extract_features(frequencies, predicted_spectrum["co"])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    mask = pixel_layout(codes)
    pixels_path = output / "design_pixels.csv"
    rectangles_path = output / "design_rectangles_m.csv"
    np.savetxt(pixels_path, mask, delimiter=",", fmt="%d")
    with rectangles_path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["x_center_m", "y_center_m", "x_span_m", "y_span_m", "z_min_m", "z_max_m"])
        for row in rectangles(codes, metadata["physics"]["pixel_m"]):
            writer.writerow([*row, 0, metadata["physics"]["metal_thickness_m"]])
    np.savetxt(output / "target_synthetic_spectrum.csv",
               np.column_stack([frequencies, data["co"][target_id], data["cross"][target_id],
                                data["total"][target_id]]), delimiter=",",
               header="frequency_hz,R_co,R_cross,R_total", comments="")
    np.savetxt(output / "predicted_synthetic_spectrum.csv",
               np.column_stack([frequencies, predicted_spectrum["co"], predicted_spectrum["cross"],
                                predicted_spectrum["total"]]), delimiter=",",
               header="frequency_hz,R_co,R_cross,R_total", comments="")
    (output / "build_design.lsf").write_text(
        project_script(codes, metadata["physics"], rectangle_fn=rectangles) + '\nsave("design.fsp");\n')
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(mask, origin="lower", cmap="Greys", interpolation="nearest")
    ax.set(title="Synthetic-paper predicted copper mask", xlabel="x pixel", ylabel="y pixel")
    fig.tight_layout()
    fig.savefig(output / "design_preview.png", dpi=180)
    plt.close(fig)
    core = {"scenario": "synthetic_surrogate", "backend": "synthetic_surrogate",
            "pattern_codes": codes.tolist(), "tile_masks": masks().tolist(),
            "physics": metadata["physics"], "surrogate_id": metadata["surrogate_id"],
            "target_row": target_id, "target_features": data["features"][target_id].tolist(),
            "predicted_features": predicted_features.tolist(),
            "predicted_reason": predicted_reason,
            "artifact_checksums": {"design_pixels.csv": _file_digest(pixels_path),
                                   "design_rectangles_m.csv": _file_digest(rectangles_path)}}
    core["design_checksum"] = hashlib.sha256(canonical_json(core).encode()).hexdigest()
    (output / "design_spec.json").write_text(json.dumps(core, indent=2))
    report.update(target_row=target_id, predicted_codes=codes.tolist(),
                  design_checksum=core["design_checksum"])
    (output / "report.json").write_text(json.dumps(report, indent=2))
    return report

