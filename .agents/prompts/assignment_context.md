# TP544 Assignment 3: reduced metasurface reproduction using machine learning

## Exercise context

Implement and evaluate a reduced reproduction of the inverse-design method in
Ghorbani et al., Deep neural network-based automatic metasurface design with a
wide frequency range, Scientific Reports 11:7102 (2021).
DOI: https://doi.org/10.1038/s41598-021-86588-2.

This file records the exercise agreed in this conversation. An official instructor
statement or grading rubric was not supplied. Do not invent course requirements,
grading weights, hardware measurements, or results from the original authors.
The project name is `tp544-assignment3-metasurface-ml`.

The central task is to learn an inverse mapping: given desired reflection-notch
properties, predict a metasurface geometry. Lumerical FDTD supplies the physical
training examples and checks the predicted design. PyTorch trains the neural
network from the saved examples without a live solver connection. The reduced
scope was chosen to make a small, defensible reproduction feasible by Friday.

## What is being reproduced

The paper generates geometry-spectrum pairs by electromagnetic simulation, extracts
notch descriptors, trains a dense inverse neural network, and checks selected
predictions through additional simulations. Its original solver is CST. This
exercise implements the same overall workflow using Lumerical and a smaller,
explicitly defined geometry family and network.

The original paper uses 2000 random examples, eight ring patterns, sixteen tile
choices, up to eight notches and two network architectures. The restricted
geometry family has 8^16 possible tile combinations. This exercise enumerates
256 candidates. It is a methodological reproduction with documented changes,
not a replication of the paper's dataset, exact tile masks, accuracy or timing.
Full geometry/data and network comparison tables are in `AGENTS.md`.

## Reduced physical and learning scenario

| Component | Required scenario |
|---|---|
| Unit cell | 32 x 32 pixels, arranged as sixteen 8 x 8 tiles |
| Dimensions | 0.2 mm pixel, 6.4 mm cell, 1.5 mm FR4 substrate, 18 micrometre metal |
| Substrate | Relative permittivity 4.2 + 0.025i, checked against the fitted solver material |
| Metal | Conductive copper, 5.8e7 S/m, patterned top and continuous ground |
| Excitation | Normal incidence, x polarization, propagation toward -z |
| Boundaries | Periodic x/y and PML z, representing an infinite repeated surface |
| Band | 4-45 GHz, 1641 frequency points |
| Geometry variables | Four categorical tile IDs A, B, C, D, each from 0 to 3 |
| Features | Up to two notches, each with frequency GHz, depth dB, width GHz and presence flag |
| Network | 8 inputs, hidden widths 32 and 64 with ReLU, 16 logits interpreted as four 4-class heads |
| Learning | Cross entropy, Adam, learning rate 0.001, batch size 16 |
| Training limit | Up to 500 epochs, early stopping after 30 unimproved validation epochs |
| Split | 70% training, 15% validation, remaining test, fixed seed 2026 |
| Device | CUDA GPU for training, explicit CPU settings for software proof tests only |
| Baseline | Nearest neighbour using the same training subset and scaled features |

Tile definitions are empty, a 0.8 mm square ring with 0.2 mm line width, a 1.2 mm
square ring with 0.2 mm line width, and a 1.2 mm square ring with 0.4 mm line width.
Each tile is centred. The four tile rows are `[A,B,B,A]`, `[C,D,D,C]`,
`[C,D,D,C]`, `[A,B,B,A]`. Enumerate all 4^4 combinations once.

A notch is a contiguous interval below -10 dB in co-polar reflection power,
with interpolated threshold-crossing width. Preserve spectra that are rejected
for training. Exclude spectra with no notches, more than two notches, truncated
notches, or unresolved notches. At least 24 eligible geometries are required.
The candidate count does not guarantee sufficient qualifying physical responses.

## Required workflow

1. Read the original PDF in `paper/s41598-021-86588-2.pdf`. Use `paper/paper.txt`
   for searching and the PDF for authoritative figures, tables and equations.
   Explain the paper's problem, proposed system, methodology, metasurface,
   network and results in accessible language, then identify the changes here.
2. Run `01_generate_geometries.py` offline. Save all 256 pattern codes and
   masks in `geometry_checkpoint.npz`, checkpointing after every layout.
3. Run `02_simulate_geometries.py` via the Lumerical backend. Consume the saved
   complete geometry manifest, perform the existing preflight/convergence gates,
   and save raw powers and per-layout runtimes in `simulation_checkpoint.npz`.
   Do not extract notch descriptors in this stage.
4. Run `03_train.py` on CUDA. Prepare `training_dataset.npz` offline from raw
   spectra, extracting notch descriptors and eligibility reasons. Fit scaling
   only on train, use validation for early stopping, and reserve test records.
   Save optimizer, RNG, batch order and progress after every minibatch.
5. Run `04_test_and_export.py` to evaluate held-out samples and the baseline,
   then export exactly one NN-predicted geometry with mask, SI rectangles,
   physical specification, preview and complete Lumerical construction script.
6. Simulate that design in Lumerical and check its own numerical convergence.
   Compare notch count, frequency, depth and width separately from geometry accuracy.
7. Inspect `runs/real/analysis/ANALISE_REDES.md`, generated by
   `src/real/00_analyze_networks.py` and automatically before training. It counts both original networks and the
   reduced network, documenting the parameter inconsistency in Table 1.

Default export target is the first held-out sample, selected without choosing
the best-looking prediction. A custom target can be set through `config.py`.
Changing the physical scenario or saved dataset requires new experiment folders.
A PEC acceleration pilot is a different metal model and must remain separate
from copper data and results.

## Deliverables

- The four workflow scripts, optional analysis script and reusable modules, with constants in
  `config.py`, complete function docstrings and comments explaining assumptions.
- Crash recovery in generation and training, with atomic files, integrity checks
  and a previous-snapshot backup. Repeating the command resumes the same run.
- Test evidence for geometry, descriptors, interruption/recovery, data identity,
  split/scaling isolation, solver result contracts and one-design export.
- Held-out metrics, nearest-neighbour comparison, training curves and one design's
  physical validation results when genuine solver execution is available.
- `README.md` with installation, GPU checks, ordered execution, recovery and
  Lumerical project inspection/validation steps.
- `JUSTIFICATIVA_REDUCAO.md` with concise Portuguese bullets and comparison tables.
  Explain deadline risk from simulations and unavailable data. Do not claim the
  paper trained on trillions of layouts or that K80 superiority makes training
  impossible. The actual workstation and simulation duration are unmeasured.
- `AGENTS.md` with paper/reproduction comparison tables, this prompt context,
  and the attributed original PDF and searchable text.
- Repository creation and publication are outside the current delivery request.

## Interpretation and acceptance criteria

The reproduction must clearly distinguish implemented software from performed
physical experiments. A successful synthetic or CPU proof test establishes
software behaviour only. It does not establish CUDA execution, electromagnetic
accuracy, physical feasibility or reproduction of the paper's results.

Different geometries can have similar spectra, so exact-layout accuracy alone
is insufficient. Report per-position and complete-layout accuracy, test sample
count, baseline performance and the exported design's physical notch errors.
Label cached spectrum comparisons separately from fresh simulation results.
If the notch count differs, report the mismatch instead of misleading errors.

Use total reflection, including cross polarization, when discussing absorption
from the grounded surface. Low co-polar reflection alone does not prove absorption.
Copper skin-depth losses can require much finer mesh than a coarse pilot, and
convergence of one pilot does not validate every generated layout.

If fewer than 24 real spectra qualify, document the feasibility failure and
revise the scenario transparently. Never replace physical data with synthetic
fixtures to claim a completed electromagnetic reproduction. Training should
fail clearly when CUDA is requested but unavailable, with no silent CPU fallback.

## Current evidence and remaining work

`evidence/validation.md` records the checks performed for this revision.
Historical evidence filenames belong to the previous three-entrypoint release.
`evidence/software-proof/` contains a tagged artificial export for format checks,
not a physically validated design.

Actual CUDA training and licensed Lumerical simulation have not been performed
in this environment. On the execution workstation, generate real data, train
on GPU, simulate the exported design, and record numerical convergence and
physical errors before claiming physical reproduction results.

## References

1. Original paper: https://doi.org/10.1038/s41598-021-86588-2.
2. Ansys RF simulation methodology:
   https://optics.ansys.com/hc/en-us/articles/360042054154-Radio-frequency-simulation-methodology.
3. Ansys Python simulation-result format:
   https://optics.ansys.com/hc/en-us/articles/39744236202771-Accessing-Simulation-Results-Python-API.
4. Official PyTorch installation guidance: https://pytorch.org/get-started/locally/.
