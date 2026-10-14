"""Contracts for the independent narrowband physical workflows."""

import importlib.util
import inspect
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from metasurface import config


ROOT = Path(__file__).parents[1]


def _load_script(workflow, name):
    """Import a numbered workflow script without executing its stage.

    Args:
        workflow (str): Directory below ``src``.
        name (str): Script filename.

    Returns:
        module: Imported entrypoint module.
    """
    path = ROOT / "src" / workflow / name
    spec = importlib.util.spec_from_file_location(f"{workflow}_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _flat_response(codes, frequencies):
    """Return a valid passive fixture spectrum for checkpoint tests."""
    del codes
    co = np.full(len(frequencies), 0.5)
    cross = np.zeros(len(frequencies))
    return {"co": co, "cross": cross, "total": co.copy()}


def _complete_paper_manifest(directory, count):
    """Create a complete paper manifest in one test-only atomic write."""
    from metasurface.common.storage import canonical_json, save_npz
    from metasurface.paper.geometry import masks, pixel_layout, sample_codes

    codes = sample_codes(count)
    path = directory / "geometry_checkpoint.npz"
    save_npz(path, {
        "metadata": np.asarray(canonical_json({
            "schema": 1, "family": "paper-scale-rings-v1", "count": count,
            "seed": 2026,
        })),
        "tiles": masks(),
        "codes": codes,
        "pixels": np.asarray([pixel_layout(code) for code in codes]),
        "completed": np.ones(count, dtype=bool),
        "seconds": np.zeros(count),
    })
    return path


def test_narrowband_configuration_is_exact_and_independent():
    """Catch frequency drift, shared dictionaries, paths, or sample counts."""
    physics = config.NARROWBAND_PHYSICS
    frequencies = np.linspace(physics["frequency_min_hz"],
                              physics["frequency_max_hz"],
                              physics["frequency_points"])
    assert frequencies[[0, -1]].tolist() == [22e9, 28e9]
    assert len(frequencies) == 301
    assert np.array_equal(np.diff(frequencies), np.full(300, 20e6))
    assert {key: value for key, value in physics.items()
            if key not in {"frequency_min_hz", "frequency_max_hz", "frequency_points"}} == {
                key: value for key, value in config.PHYSICS.items()
                if key not in {"frequency_min_hz", "frequency_max_hz", "frequency_points"}
            }
    assert config.NARROWBAND_ROOT == config.ROOT / "runs" / "real_narrowband"
    assert config.PAPER_NARROWBAND_ROOT == config.ROOT / "runs" / "real_paper_narrowband"
    assert config.PAPER_NARROWBAND_COUNT == 2000
    assert config.NARROWBAND_TRAINING == config.TRAINING
    assert config.NARROWBAND_TRAINING is not config.TRAINING
    assert config.PAPER_NARROWBAND_TRAINING == config.PAPER_TRAINING
    assert config.PAPER_NARROWBAND_TRAINING is not config.PAPER_TRAINING


def test_narrowband_entrypoints_import_without_solver_api():
    """Catch eager lumapi imports and missing mirrored entrypoints."""
    sys.modules.pop("lumapi", None)
    workflows = {
        "real_narrowband": ("00_analyze_networks.py", "01_generate_geometries.py",
                            "02_simulate_geometries.py", "03_train.py",
                            "04_test_and_export.py", "05_training_graphs.py"),
        "real_paper_narrowband": ("01_generate_geometries.py",
                                  "02_simulate_geometries.py", "03_train.py",
                                  "04_test_and_export.py", "05_training_graphs.py"),
    }
    for workflow, names in workflows.items():
        for name in names:
            assert callable(_load_script(workflow, name)._run)
            assert "lumapi" not in sys.modules


def test_narrowband_simulation_shapes_identity_and_resume(tmp_path):
    """Catch wrong band sizes and preflight-mode loss from checkpoint identity."""
    from metasurface.paper.dataset import load_raw, simulate
    from metasurface.reduced.dataset import load_simulations, simulate_geometries
    from metasurface.reduced.geometry_manifest import generate_geometries

    reduced_manifest = generate_geometries(tmp_path / "reduced-geometries",
                                           config.NARROWBAND_PHYSICS["pixel_m"])
    reduced_path = simulate_geometries(
        reduced_manifest, tmp_path / "reduced-data", config.NARROWBAND_PHYSICS,
        backend="synthetic", limit=1, simulator=_flat_response,
        preflight_mode="reference_only")
    reduced = load_simulations(reduced_path)
    assert reduced["co"].shape == (256, 301)
    assert json.loads(reduced["metadata"].item())["preflight_mode"] == "reference_only"
    assert simulate_geometries(
        reduced_manifest, tmp_path / "reduced-data", config.NARROWBAND_PHYSICS,
        backend="synthetic", limit=1, simulator=_flat_response,
        preflight_mode="reference_only") == reduced_path
    with pytest.raises(ValueError, match="configuration or geometry changed"):
        simulate_geometries(
            reduced_manifest, tmp_path / "reduced-data", config.NARROWBAND_PHYSICS,
            backend="synthetic", limit=1, simulator=_flat_response,
            preflight_mode="full")

    paper_manifest = _complete_paper_manifest(tmp_path / "paper-geometries", 2000)
    paper_path = simulate(
        paper_manifest, tmp_path / "paper-data", config.NARROWBAND_PHYSICS,
        limit=1, simulator=_flat_response, preflight_mode="reference_only")
    paper = load_raw(paper_path)
    assert paper["co"].shape == (2000, 301)
    assert json.loads(paper["metadata"].item())["preflight_mode"] == "reference_only"
    assert simulate(
        paper_manifest, tmp_path / "paper-data", config.NARROWBAND_PHYSICS,
        limit=1, simulator=_flat_response,
        preflight_mode="reference_only") == paper_path
    with pytest.raises(ValueError, match="identity changed"):
        simulate(paper_manifest, tmp_path / "paper-data", config.NARROWBAND_PHYSICS,
                 limit=1, simulator=_flat_response, preflight_mode="full")


def test_paper_full_preflight_keeps_legacy_checkpoint_identity(tmp_path):
    """Catch accidental invalidation of existing paper simulation checkpoints."""
    from metasurface.common.storage import canonical_json
    from metasurface.paper.dataset import simulate
    from metasurface.paper.manifest import generate, load_manifest

    manifest_path = generate(tmp_path / "geometries", count=2)
    manifest = load_manifest(manifest_path)
    expected = canonical_json({
        "schema": 1,
        "stage": "simulations",
        "family": "paper-scale-rings-v1",
        "backend": "test_fixture",
        "physics": config.PHYSICS,
        "geometry_id": manifest["checksum"].item(),
    })
    path = simulate(manifest_path, tmp_path / "data", config.PHYSICS,
                    simulator=_flat_response)
    with np.load(path, allow_pickle=False) as saved:
        assert saved["metadata"].item() == expected


def test_extended_interfaces_default_to_full_preflight():
    """Catch backward-incompatible defaults on public physical interfaces."""
    from metasurface.paper.dataset import simulate
    from metasurface.paper.evaluation import evaluate as evaluate_paper
    from metasurface.reduced.evaluation import evaluate as evaluate_reduced

    assert inspect.signature(simulate).parameters["preflight_mode"].default == "full"
    assert inspect.signature(evaluate_reduced).parameters["preflight_mode"].default == "full"
    assert inspect.signature(evaluate_paper).parameters["preflight_mode"].default == "full"


@pytest.mark.parametrize("module_name", [
    "metasurface.reduced.evaluation",
    "metasurface.paper.evaluation",
])
def test_evaluators_reject_invalid_preflight_before_loading_files(module_name):
    """Catch silently accepted invalid preflight modes when validation is off."""
    import importlib

    evaluate = importlib.import_module(module_name).evaluate
    with pytest.raises(ValueError, match="Invalid preflight mode"):
        evaluate("missing.pt", "missing.npz", "unused",
                 preflight_mode="invalid")


@pytest.mark.parametrize(
    ("workflow", "script", "module_name", "function_name", "metrics_name"),
    [
        ("real_narrowband", "02_simulate_geometries.py",
         "metasurface.reduced.dataset", "simulate_geometries",
         "write_simulation_metrics"),
        ("real_paper_narrowband", "02_simulate_geometries.py",
         "metasurface.paper.dataset", "simulate", "write_simulation_metrics"),
        ("real_narrowband", "04_test_and_export.py",
         "metasurface.reduced.evaluation", "evaluate", None),
        ("real_paper_narrowband", "04_test_and_export.py",
         "metasurface.paper.evaluation", "evaluate", None),
    ],
)
def test_narrowband_physical_entrypoints_request_reference_only(
        monkeypatch, workflow, script, module_name, function_name, metrics_name):
    """Catch collection or validation entrypoints reverting to full preflight."""
    import importlib

    calls = []
    target_module = importlib.import_module(module_name)

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return Path("fixture.npz")

    monkeypatch.setattr(target_module, function_name, capture)
    if metrics_name:
        records = importlib.import_module("metasurface.common.run_records")
        monkeypatch.setattr(records, metrics_name, lambda path: None)
    _load_script(workflow, script)._run()
    assert len(calls) == 1
    assert calls[0][1]["preflight_mode"] == "reference_only"


def test_run_commands_lists_every_narrowband_stage():
    """Catch a runbook that omits any executable narrowband stage."""
    text = (ROOT / "RUN_COMMANDS.md").read_text()
    for workflow, first in (("real_narrowband", 0), ("real_paper_narrowband", 1)):
        for stage in range(first, 6):
            names = list((ROOT / "src" / workflow).glob(f"{stage:02d}_*.py"))
            assert len(names) == 1
            assert f"src/{workflow}/{names[0].name}" in text
