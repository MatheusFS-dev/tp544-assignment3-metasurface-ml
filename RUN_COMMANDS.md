# Dummy

```bash
uv run --locked python src/dummy/01_generate_geometries.py
uv run --locked python src/dummy/02.1_simulate_geometries.py
uv run --locked python src/dummy/03.1_dummy_train.py
uv run --locked python src/dummy/04.1_dummy_test_and_export.py
uv run --locked python src/dummy/05.1_dummy_training_graphs.py
```

# Real

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
