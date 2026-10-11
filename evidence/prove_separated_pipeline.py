"""Reproduce the independent dummy workflow in a temporary directory."""

import json
from pathlib import Path
import runpy
import tempfile

from metasurface import config
from metasurface.reduced.dataset import load_simulations
from metasurface.reduced.learning import load_checkpoint

ROOT = Path(__file__).resolve().parents[1]

with tempfile.TemporaryDirectory(prefix="metasurface-dummy-proof-") as temporary:
    config.DUMMY_ROOT = Path(temporary) / "dummy"
    config.TRAINING = {**config.TRAINING, "max_epochs": 4, "patience": 30}
    runpy.run_path(str(ROOT / "src/dummy/01_generate_geometries.py"), run_name="__main__")
    runpy.run_path(str(ROOT / "src/dummy/02.1_simulate_geometries.py"), run_name="__main__")
    raw = load_simulations(config.DUMMY_ROOT / "data/simulation_checkpoint.npz")
    assert raw["completed"].all()
    runpy.run_path(str(ROOT / "src/dummy/03.1_dummy_train.py"), run_name="__main__")
    assert load_checkpoint(config.DUMMY_ROOT / "training/last.pt")["finished"]
    runpy.run_path(str(ROOT / "src/dummy/04.1_dummy_test_and_export.py"), run_name="__main__")
    report = json.loads((config.DUMMY_ROOT / "design/report.json").read_text())
    assert report["backend"] == "synthetic" and report["physical_validation"] == "not_run"
    print("PASS: independent dummy entrypoints, saved spectra, training, and labeled export.")
