"""Contracts for the six no-argument synthetic-paper entry scripts."""

import importlib.util
from pathlib import Path
import sys

from metasurface import config


ROOT = Path(__file__).parents[1]


def _load_script(name):
    path = ROOT / "src/synthetic_paper" / name
    spec = importlib.util.spec_from_file_location(f"synthetic_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_configuration_matches_approved_paper_scale_workflow():
    """Keep full-run values configured without executing them in tests."""
    assert config.SYNTHETIC_PAPER_COUNT == 2000
    assert config.SYNTHETIC_PAPER_GENERATION_LIMIT is None
    assert config.SYNTHETIC_PAPER_TRAINING["max_epochs"] == 10000
    assert config.SYNTHETIC_PAPER_TRAINING["batch_size"] == 30
    assert config.SYNTHETIC_PAPER_TRAINING["device"] == "cuda"
    assert config.SYNTHETIC_PAPER_TRAINING["allow_synthetic_surrogate"] is True
    assert "allow_synthetic_surrogate" not in config.PAPER_TRAINING


def test_offline_scripts_import_without_solver_api():
    """Keep solver access isolated to the explicit validation script."""
    sys.modules.pop("lumapi", None)
    for name in ("01_generate_geometries.py", "02_generate_synthetic_spectra.py",
                 "03_train.py", "04_test_and_export.py", "05_training_graphs.py"):
        module = _load_script(name)
        assert callable(module._run)
        assert "lumapi" not in sys.modules


def test_validation_script_is_disabled_by_default():
    """Require an explicit configuration change before a solver run."""
    module = _load_script("06_validate_in_lumerical.py")
    assert config.SYNTHETIC_PAPER_RUN_VALIDATION is False
    try:
        module._run()
    except RuntimeError as error:
        assert "disabled" in str(error).lower()
    else:
        raise AssertionError("Validation script should refuse the default configuration.")


def test_run_commands_lists_the_complete_synthetic_workflow():
    """Keep the other-PC runbook synchronized with all numbered stages."""
    text = (ROOT / "RUN_COMMANDS.md").read_text()
    for name in ("01_generate_geometries.py", "02_generate_synthetic_spectra.py",
                 "03_train.py", "04_test_and_export.py", "05_training_graphs.py",
                 "06_validate_in_lumerical.py"):
        assert f"src/synthetic_paper/{name}" in text
    assert "SYNTHETIC_PAPER_RUN_VALIDATION" in text
