"""Import contracts for externally supplied paper-scale preliminary data."""

import csv
import json

import numpy as np


def _write_inputs(directory):
    """Write a small, traceable preliminary-data fixture.

    Args:
        directory: Directory that receives the source fixture files.

    Returns:
        tuple: Paths for samples, frequencies, spectra, and notch records.
    """
    samples = directory / "amostras.csv"
    fields = ["id", "origem"] + [f"p{row}{column}"
                                  for row in range(1, 5)
                                  for column in range(1, 5)] + ["bits48", "fracao_cobre"]
    rows = []
    for sample_id, values in ((11, [1, 2] * 8), (12, [8, 7] * 8)):
        rows.append({"id": sample_id, "origem": "fixture",
                     **{field: value for field, value in zip(fields[2:18], values)},
                     "bits48": "".join(f"{value - 1:03b}" for value in values),
                     "fracao_cobre": 0.5})
    with samples.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    frequencies = directory / "frequencias.txt"
    frequencies.write_text("4\n5\n6\n7\n8\n9\n10\n11\n12\n13\n14\n")
    spectra = directory / "s11_todos.txt"
    spectra.write_text("11 -2 -2 -2 -2 -11 -15 -11 -2 -2 -2 -2\n"
                       "12 -2 -2 -2 -2 -11 -16 -11 -2 -2 -2 -2\n")
    notches = directory / "preliminar_notches.csv"
    notches.write_text("id,n_notches\n11,1\n12,1\n")
    return samples, frequencies, spectra, notches


def test_imported_preliminary_data_prepares_from_the_supplied_s11_trace(tmp_path):
    """Import labels and power traces without fabricating absent channels."""
    from metasurface.paper.dataset import load_dataset, load_raw, prepare
    from metasurface.paper.preliminary_import import import_preliminary

    paths = _write_inputs(tmp_path)
    raw_path = import_preliminary(*paths, tmp_path / "simulation_checkpoint.npz")
    raw = load_raw(raw_path)

    assert raw["codes"].tolist() == [[0, 1] * 8, [7, 6] * 8]
    assert raw["source_sample_ids"].tolist() == [11, 12]
    assert raw["feature_trace"].item() == "reflection_power"
    assert set(raw) >= {"reflection_power", "codes", "frequencies", "completed"}
    assert "co" not in raw and "cross" not in raw and "total" not in raw
    assert np.allclose(raw["reflection_power"][0], 10 ** (np.array(
        [-2, -2, -2, -2, -11, -15, -11, -2, -2, -2, -2]) / 10))
    from metasurface.paper.geometry import rectangles
    from metasurface.reduced.lumerical import project_script
    assert 'addfdtd' in project_script(raw["codes"][0],
                                       json.loads(raw["metadata"].item())["physics"],
                                       rectangle_fn=rectangles)

    prepared_path = prepare(raw_path, tmp_path / "training_dataset.npz")
    prepared = load_dataset(prepared_path)
    assert prepared["reason"].tolist() == ["ok", "ok"]
    assert prepared["features"].shape == (2, 24)
