"""Reproduce the software-only entrypoint proof using artificial CPU fixtures."""

import json
from pathlib import Path
import runpy
import shutil
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
from dataset import load_simulations, prepare_dataset
from learning import load_checkpoint, train

# In-memory overrides cannot change the delivered physical/CUDA defaults.
sys.modules["lumapi"] = None
with tempfile.TemporaryDirectory(prefix="metasurface-software-proof-") as temporary:
    work = Path(temporary)
    config.GEOMETRY_DIR = work / "geometries"
    config.DATA_DIR = work / "data"
    config.TRAIN_DIR = work / "resumed"
    config.EXPORT_DIR = work / "export"
    config.BACKEND = "synthetic"
    config.TRAINING = {**config.TRAINING, "device": "cpu", "allow_synthetic": True,
                       "max_epochs": 4, "patience": 30}
    config.RUN_EXPORTED_DESIGN = False
    runpy.run_path(str(ROOT / "01_generate_geometries.py"), run_name="__main__")
    geometry_bytes = (config.GEOMETRY_DIR / "geometry_checkpoint.npz").read_bytes()
    runpy.run_path(str(ROOT / "01_generate_geometries.py"), run_name="__main__")
    assert (config.GEOMETRY_DIR / "geometry_checkpoint.npz").read_bytes() == geometry_bytes
    config.GENERATION_LIMIT = 4
    runpy.run_path(str(ROOT / "02_simulate_geometries.py"), run_name="__main__")
    config.GENERATION_LIMIT = 2
    runpy.run_path(str(ROOT / "02_simulate_geometries.py"), run_name="__main__")
    raw = config.DATA_DIR / "simulation_checkpoint.npz"
    assert load_simulations(raw)["completed"].sum() == 6
    config.GENERATION_LIMIT = None
    runpy.run_path(str(ROOT / "02_simulate_geometries.py"), run_name="__main__")
    assert load_simulations(raw)["completed"].all()
    assert "features" not in load_simulations(raw)
    raw_bytes = raw.read_bytes()
    prepared = prepare_dataset(raw, config.DATA_DIR / "training_dataset.npz")
    train(prepared, config.TRAIN_DIR, config.TRAINING, max_steps=3)
    assert load_checkpoint(config.TRAIN_DIR / "last.pt")["step"] == 3
    # A blocked solver module proves training consumes saved arrays only.
    sys.modules["lumerical"] = None
    runpy.run_path(str(ROOT / "03_train.py"), run_name="__main__")
    sys.modules.pop("lumerical")
    resumed = load_checkpoint(config.TRAIN_DIR / "last.pt")
    full = load_checkpoint(train(prepared, work / "full", config.TRAINING))
    import torch

    assert resumed["history"] == full["history"]
    assert resumed["step"] == full["step"]
    for name in resumed["model"]:
        assert torch.equal(resumed["model"][name], full["model"][name])
    prepared_id = prepared.read_bytes()
    runpy.run_path(str(ROOT / "03_train.py"), run_name="__main__")
    assert prepared.read_bytes() == prepared_id
    assert raw.read_bytes() == raw_bytes
    runpy.run_path(str(ROOT / "04_test_and_export.py"), run_name="__main__")
    assert len(list(config.EXPORT_DIR.glob("*_pixels.csv"))) == 1
    report = json.loads((config.EXPORT_DIR / "report.json").read_text())
    assert report["backend"] == "synthetic" and report["physical_validation"] == "not_run"
    assert not (config.EXPORT_DIR / "design.fsp").exists()
    destination = ROOT / "evidence" / "software-proof"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(config.EXPORT_DIR, destination)
    (destination / "NOTICE.txt").write_text(
        "SYNTHETIC CPU SOFTWARE PROOF ONLY. This is not a physical Lumerical result.\n")
    print("PASS: separate entrypoints, resume, stable prepared data, exact CPU weights, one tagged export.")
