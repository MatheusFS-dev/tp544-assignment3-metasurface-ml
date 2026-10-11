"""Durable stage logs and metrics records."""

import json
from pathlib import Path

import pytest

from metasurface.common.run_records import run_stage, timed_part


def test_stage_records_output_and_elapsed_time(tmp_path, capsys):
    """Save stdout, stderr, and a completed status inside the run directory.

    Args:
        tmp_path (Path): Temporary run root.
        capsys (fixture): Console capture.
    """
    def work():
        with timed_part("compute"):
            print("metric=0.25")
        return 7

    assert run_stage(tmp_path, "train", work) == 7
    status_files = list((tmp_path / "logs" / "train").glob("*.json"))
    assert len(status_files) == 1
    status = json.loads(status_files[0].read_text())
    assert status["status"] == "completed"
    assert status["elapsed_seconds"] >= 0
    assert status["parts"][0]["name"] == "compute"
    assert status["parts"][0]["elapsed_seconds"] >= 0
    assert "metric=0.25" in status_files[0].with_suffix(".log").read_text()
    assert "metric=0.25" in capsys.readouterr().out


def test_stage_records_exception_and_raises(tmp_path):
    """Persist the traceback and failed status before propagating an error.

    Args:
        tmp_path (Path): Temporary run root.
    """
    def broken():
        with timed_part("solver_call"):
            raise RuntimeError("solver failed")

    with pytest.raises(RuntimeError, match="solver failed"):
        run_stage(tmp_path, "simulate", broken)
    status_file = next((tmp_path / "logs" / "simulate").glob("*.json"))
    status = json.loads(status_file.read_text())
    assert status["status"] == "failed"
    assert status["error_type"] == "RuntimeError"
    assert status["parts"][0]["status"] == "failed"
    assert "solver failed" in status_file.with_suffix(".log").read_text()


def test_new_geometry_checkpoints_store_per_layout_runtime(tmp_path):
    """Keep timing next to each committed geometry result.

    Args:
        tmp_path (Path): Temporary run root.
    """
    from metasurface.reduced.geometry_manifest import generate_geometries, load_geometries
    from metasurface.paper.manifest import generate, load_manifest

    reduced = load_geometries(generate_geometries(tmp_path / "reduced", 0.2e-3, limit=1),
                              require_complete=False)
    paper = load_manifest(generate(tmp_path / "paper", count=2, limit=1),
                          require_complete=False)
    assert reduced["seconds"].shape == (256,)
    assert paper["seconds"].shape == (2,)
    assert reduced["seconds"][0] >= 0
    assert paper["seconds"][0] >= 0


def test_reduced_training_commits_minibatch_metrics(tmp_path):
    """Save minibatch loss and runtime in the same durable checkpoint.

    Args:
        tmp_path (Path): Temporary run root.
    """
    from metasurface import config
    from metasurface.reduced.dataset import generate
    from metasurface.reduced.learning import load_checkpoint, train

    data = generate(tmp_path / "data", config.PHYSICS, "synthetic", limit=32)
    options = {**config.TRAINING, "device": "cpu", "allow_synthetic": True,
               "max_epochs": 1}
    state = load_checkpoint(train(data, tmp_path / "training", options, max_steps=1))
    assert len(state["batch_history"]) == 1
    assert state["batch_history"][0]["step"] == 1
    assert state["batch_history"][0]["loss"] >= 0
    assert state["batch_history"][0]["compute_seconds"] >= 0
    from metasurface.common.run_records import write_training_metrics
    output = write_training_metrics(tmp_path / "training/last.pt")
    record = json.loads(output.read_text())
    assert record["steps"] == 1
    assert record["minibatches"][0]["step"] == 1


def test_dataset_summary_counts_saved_eligibility(tmp_path):
    """Export a readable eligibility summary from saved dataset arrays.

    Args:
        tmp_path (Path): Temporary run root.
    """
    from metasurface import config
    from metasurface.common.run_records import write_dataset_metrics
    from metasurface.reduced.dataset import generate

    data = generate(tmp_path / "data", config.PHYSICS, "synthetic", limit=32)
    summary = json.loads(write_dataset_metrics(data).read_text())
    assert summary["completed"] == 32
    assert summary["eligible"] == 32
    assert summary["reasons"]["ok"] == 32


def test_geometry_and_simulation_summaries_keep_layout_timings(tmp_path):
    """Expose committed layout counts and measured runtimes as JSON.

    Args:
        tmp_path (Path): Temporary run root.
    """
    from metasurface import config
    from metasurface.common.run_records import write_geometry_metrics, write_simulation_metrics
    from metasurface.reduced.dataset import simulate_geometries
    from metasurface.reduced.geometry_manifest import generate_geometries

    manifest = generate_geometries(tmp_path / "geometry", config.PHYSICS["pixel_m"])
    geometry = json.loads(write_geometry_metrics(manifest).read_text())
    assert geometry["completed"] == 256
    assert geometry["measured_layouts"] == 256
    raw = simulate_geometries(manifest, tmp_path / "data", config.PHYSICS,
                              backend="synthetic", limit=1)
    simulation = json.loads(write_simulation_metrics(raw).read_text())
    assert simulation["completed"] == 1
    assert simulation["per_layout_seconds"][0] >= 0
    assert simulation["per_layout_seconds"][1] is None
