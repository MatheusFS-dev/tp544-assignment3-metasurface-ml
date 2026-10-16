# Dummy

```bash
uv run --locked python src/dummy/01_generate_geometries.py
uv run --locked python src/dummy/02.1_simulate_geometries.py
uv run --locked python src/dummy/03.1_dummy_train.py
uv run --locked python src/dummy/04.1_dummy_test_and_export.py
uv run --locked python src/dummy/05.1_dummy_training_graphs.py
```

# Real

All Lumerical-backed stages require an active FDTD GPU resource. The shared
solver runs the GPU system check and launches `run("FDTD", "GPU")`; it raises
an error instead of falling back to CPU.

## Standalone FDTD GUI scripts

Each physical workflow also has one `.lsf` that generates its tile codes and
32 × 32 copper masks, constructs each FDTD model, runs it on GPU, and writes
the geometry manifest and one MAT result per completed sample. These scripts
run directly in the Lumerical FDTD Script File Editor. Set the GUI working
directory to a new, empty experiment directory before pressing Run. Configure
an active GPU resource first. An interrupted run can resume in the same
directory. Change the directory and `run_prefix` when changing physics or tile
definitions.

| Workflow | Standalone script |
|---|---|
| Real | `src/real/02_simulate_in_fdtd_gui.lsf` |
| Real paper | `src/real_paper/02_simulate_in_fdtd_gui.lsf` |
| Real narrowband | `src/real_narrowband/02_simulate_in_fdtd_gui.lsf` |
| Real paper narrowband | `src/real_paper_narrowband/02_simulate_in_fdtd_gui.lsf` |

Each script saves `<run_prefix>_geometry_manifest.mat`,
`<run_prefix>_preflight.mat`, and `<run_prefix>_sample_<id>.mat` files.
Sample files contain tile codes, the binary pixel mask, frequency in Hz,
co-polarized, cross-polarized, and total reflected power, and solver runtime.
A `.done` file marks a completed export. A failed reference or pilot stops
collection. Only the wideband paper script enables the full PEC and mesh/time
pilot gate, matching its existing Python workflow. The other configurations
use empty-reference calibration only and therefore do not establish numerical
convergence.

The LSF results are an **independent dataset**. They do not create Python
`simulation_checkpoint.npz` files for Stage 3. In particular, Lumerical's
seeded random generator gives the paper-scale scripts a different 2,000-layout
sample from NumPy's generator in the Python workflow. Do not combine their
results or training identities without an explicit import and provenance check.

```bash
uv run --locked python src/real/00_analyze_networks.py
uv run --locked python src/real/01_generate_geometries.py
uv run --locked python src/real/02_simulate_geometries.py
uv run --locked python src/real/03_train.py
uv run --locked python src/real/04_test_and_export.py
uv run --locked python src/real/05_training_graphs.py
```

# Real paper

```bash
uv run --locked python src/real_paper/01_generate_geometries.py
uv run --locked python src/real_paper/02_simulate_geometries.py
uv run --locked python src/real_paper/03_train.py
uv run --locked python src/real_paper/04_test_and_export.py
uv run --locked python src/real_paper/05_training_graphs.py
```

# Real narrowband

This independent reduced workflow uses 301 uniformly spaced samples from 22 to
28 GHz (20 MHz spacing). It keeps the 256 geometries and reduced network, writes
only under `runs/real_narrowband`, and explicitly requests `reference_only`
preflight for collection and optional fresh validation.

```bash
uv run --locked python src/real_narrowband/00_analyze_networks.py
uv run --locked python src/real_narrowband/01_generate_geometries.py
uv run --locked python src/real_narrowband/02_simulate_geometries.py
uv run --locked python src/real_narrowband/03_train.py
uv run --locked python src/real_narrowband/04_test_and_export.py
uv run --locked python src/real_narrowband/05_training_graphs.py
```

# Real paper narrowband

This independent paper-scale approximation keeps 2,000 sampled layouts and the
restricted network, uses the same 22--28 GHz grid, and writes only under
`runs/real_paper_narrowband`. Its Lumerical collection and optional design
validation also request `reference_only` preflight.

```bash
uv run --locked python src/real_paper_narrowband/01_generate_geometries.py
uv run --locked python src/real_paper_narrowband/02_simulate_geometries.py
uv run --locked python src/real_paper_narrowband/03_train.py
uv run --locked python src/real_paper_narrowband/04_test_and_export.py
uv run --locked python src/real_paper_narrowband/05_training_graphs.py
```

Both narrowband variants are focused studies near the resonances around 23.5
and 25.3 GHz. They do not force those notches as targets and do not replace the
4--45 GHz analyses. A notch crossing either band edge remains truncated.
`reference_only` skips the PEC and convergence checks, so these runs do not by
themselves establish full numerical validation.

# Synthetic paper

This workflow creates 2,000 deterministic surrogate spectra and trains the
paper-scale restricted network. Its spectra, metrics, and predicted design are
synthetic software results, not electromagnetic reproduction results.

Run one command at a time, in order, from the repository root. Repeating the
same command resumes a compatible checkpoint. Do not run two processes against
the same `runs/synthetic_paper` directory. Changes to geometry, surrogate,
physics, or training identity require a new run directory.

Stages 1 through 5 do not use Lumerical:

```bash
uv run --locked python src/synthetic_paper/01_generate_geometries.py
uv run --locked python src/synthetic_paper/02_generate_synthetic_spectra.py
uv run --locked python src/synthetic_paper/03_train.py
uv run --locked python src/synthetic_paper/04_test_and_export.py
uv run --locked python src/synthetic_paper/05_training_graphs.py
```

Stage 3 uses CUDA by default and is configured for up to 10,000 epochs. Edit
only `src/metasurface/config.py` before starting a new run identity.

Stage 6 is optional and is the only command that requires licensed Lumerical
and a working `lumapi` configuration. First set
`SYNTHETIC_PAPER_RUN_VALIDATION = True` in `src/metasurface/config.py`, then run:

```bash
uv run --locked python src/synthetic_paper/06_validate_in_lumerical.py
```

This final command reconstructs only the exported predicted geometry, saves the
Lumerical project and spectrum, and compares its physical notches with the
synthetic target. Disagreement is expected and does not convert the synthetic
dataset into physical evidence.
