"""Edit these constants, then run the four workflow scripts without arguments."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GEOMETRY_DIR = ROOT / "runs" / "real" / "geometries"
DATA_DIR = ROOT / "runs" / "real" / "data"
TRAIN_DIR = ROOT / "runs" / "real" / "training"
EXPORT_DIR = ROOT / "runs" / "real" / "design"
LUMAPI_DIR = Path(r"C:\Program Files\Lumerical\v252\api\python")
GENERATION_LIMIT = None  # Simulate all remaining layouts; reruns resume from the checkpoint.
PREFLIGHT_MODE = "reference_only"  # Fast calibration; skips PEC and convergence validation.
ALLOW_SYNTHETIC_TRAINING = False
RUN_EXPORTED_DESIGN = False
TARGET_NOTCHES = None  # Example: [(30.0, -20.0, 0.5)], in GHz, dB, GHz.

PHYSICS = {
    "pixel_m": 0.2e-3,
    "substrate_thickness_m": 1.5e-3,
    "metal_thickness_m": 18e-6,
    "epsilon_real": 4.2,
    "epsilon_imag": 0.025,
    "metal_model": "copper",
    "copper_conductivity_s_m": 5.8e7,
    "frequency_min_hz": 4e9,
    "frequency_max_hz": 45e9,
    "frequency_points": 1641,
    "mesh_xy_m": 0.1e-3,
    "mesh_z_m": 0.1e-3,
    "simulation_time_s": 30e-9,
    "autoshutoff": 1e-6,
    "source_z_m": 4e-3,
    "incident_z_m": 3e-3,
    "reflection_z_m": 6e-3,
    "z_min_m": -5e-3,
    "z_max_m": 10e-3,
    "minimum_pulse_power": 1e-3,
    "material_fit_tolerance": 0.01,
    "convergence_power_tolerance": 0.02,
}

NARROWBAND_PHYSICS = {
    **PHYSICS,
    "frequency_min_hz": 22e9,
    "frequency_max_hz": 28e9,
    "frequency_points": 301,
}

TRAINING = {
    "seed": 2026,
    "batch_size": 16,
    "learning_rate": 0.001,
    "max_epochs": 500,
    "patience": 30,
    "device": "cuda",  # NVIDIA GPU required; explicitly choose cpu for software tests.
    "allow_synthetic": ALLOW_SYNTHETIC_TRAINING,
}

NARROWBAND_ROOT = ROOT / "runs" / "real_narrowband"
NARROWBAND_GEOMETRY_DIR = NARROWBAND_ROOT / "geometries"
NARROWBAND_DATA_DIR = NARROWBAND_ROOT / "data"
NARROWBAND_TRAIN_DIR = NARROWBAND_ROOT / "training"
NARROWBAND_EXPORT_DIR = NARROWBAND_ROOT / "design"
NARROWBAND_GENERATION_LIMIT = None
NARROWBAND_PREFLIGHT_MODE = "reference_only"
NARROWBAND_RUN_EXPORTED_DESIGN = False
NARROWBAND_TARGET_NOTCHES = None
NARROWBAND_TRAINING = dict(TRAINING)

DUMMY_ROOT = ROOT / "runs" / "dummy"
PAPER_ROOT = ROOT / "runs" / "real_paper"
SYNTHETIC_PAPER_ROOT = ROOT / "runs" / "synthetic_paper"

PAPER_COUNT = 2000
PAPER_GENERATION_LIMIT = None
PAPER_RUN_EXPORTED_DESIGN = False
PAPER_TRAINING = {
    "seed": 2026,
    "batch_size": 30,
    "learning_rate": 0.001,
    "max_epochs": 10000,
    "patience": 30,
    "dropout": 0.5,
    "device": "cuda",
    "allow_test_fixture": False,
}

PAPER_NARROWBAND_ROOT = ROOT / "runs" / "real_paper_narrowband"
PAPER_NARROWBAND_COUNT = 2000
PAPER_NARROWBAND_GENERATION_LIMIT = None
PAPER_NARROWBAND_PREFLIGHT_MODE = "reference_only"
PAPER_NARROWBAND_RUN_EXPORTED_DESIGN = False
PAPER_NARROWBAND_TRAINING = dict(PAPER_TRAINING)

SYNTHETIC_PAPER_COUNT = 2000
SYNTHETIC_PAPER_GENERATION_LIMIT = None
SYNTHETIC_PAPER_RUN_VALIDATION = False
SYNTHETIC_PAPER_SURROGATE = {
    "schema": 1,
    "effective_permittivity": 2.6,
    "frequency_scale_ghz_pixel": 285.0,
    "internal_loss_ghz": 0.16,
    "external_loss_ghz": 0.22,
    "neighbour_shift_ghz": 0.13,
    "position_shift_ghz": 0.08,
    "merge_distance_ghz": 0.30,
    "maximum_modes": 8,
}
SYNTHETIC_PAPER_TRAINING = {
    "seed": 2026,
    "batch_size": 30,
    "learning_rate": 0.001,
    "max_epochs": 10000,
    "patience": 30,
    "dropout": 0.5,
    "device": "cuda",
    "allow_test_fixture": False,
    "allow_synthetic_surrogate": True,
}
