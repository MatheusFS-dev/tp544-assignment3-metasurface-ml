# Reduced reproduction context

Project: `tp544-assignment3-metasurface-ml`. Read
`.agents/prompts/assignment_context.md` for the assignment objectives, required
workflow, deliverables and interpretation criteria. Delivery is a downloadable
ZIP, with no repository publication required.

This repository implements a small, explicitly changed version of Ghorbani et al.,
"Deep neural network-based automatic metasurface design with a wide frequency range",
Scientific Reports 11, 7102 (2021), DOI: https://doi.org/10.1038/s41598-021-86588-2.
The original, unchanged PDF is `paper/s41598-021-86588-2.pdf`; searchable full text
is `paper/paper.txt`. Consult the PDF for figures and layout. The paper is CC BY 4.0;
see `paper/NOTICE.md`. Repository code is an independent implementation.

## Problem and approach

The paper learns the inverse mapping from desired reflection notches to a patterned,
metal-backed FR4 surface. Geometry labels come from simulations, then a dense neural
network predicts a pattern without a solver call. The predicted pattern is checked
by another electromagnetic simulation. The inverse mapping can be ambiguous: several
patterns can produce similar responses. Geometry accuracy is consequently not proof
of matching the desired spectrum.

## Geometry and training-data comparison

| Property | Original paper | Reduced reproduction |
|---|---|---|
| Pattern family | 8 ring patterns, each 8 x 8 pixels | 4 defined square-ring tiles, each 8 x 8 pixels |
| Independent choices | 16 tile positions | 4 positions A, B, C, D, mirrored in both axes |
| Layout count | 8^16 = 281,474,976,710,656 possible restricted layouts | 4^4 = 256 enumerated candidates |
| Unrestricted output possibilities | 2^1024 binary pixel masks; distinct from sampled restricted input family | No free pixel prediction |
| Layout | 4 x 4 tiles, 32 x 32 pixels | Same overall size, mirrored tile arrangement |
| Pixel / cell dimensions | 0.2 mm / 6.4 mm | Same |
| Substrate | FR4, 1.5 mm, relative permittivity 4.2 + 0.025i | Same nominal values, fitted sampled-data material |
| Metal | 18 micrometres copper, ground-backed | Same thickness; conductivity 5.8e7 S/m chosen for this implementation |
| Frequency band | 4-45 GHz | Same, 1641 uniformly spaced frequency samples |
| Simulations | CST Microwave Studio; 2000 random geometry-spectrum pairs | Lumerical FDTD; all 256 candidates, with preflight and convergence gate |
| Usable training records | 2000 reported pairs | Eligible subset of 256; requires >=24 usable spectra |
| Response features | Up to 8 notches, 3 descriptors each: frequency, depth, width | Up to 2 notches, 4 descriptors each: frequency GHz, depth dB, width GHz, presence |
| Width definition | Exact extraction convention not specified sufficiently to replicate | Interpolated width of a contiguous interval below -10 dB |
| Eligibility | Paper's exact filtering is not available | Reject 0 notches, >2 notches, boundary-truncated or unresolved notches; preserve raw spectra |
| Split | 70% train / 30% test | 70% train / 15% validation / remaining test, fixed seed 2026 |
| Data availability | No public dataset download established from paper | Every generated spectrum, geometry, descriptor and eligibility reason saved locally |

The 256 count is the candidate search space, not a claim that every response is
useful or every target is attainable. The original restricted output uses 48 binary
values encoding sixteen 3-bit pattern IDs; not every arbitrary free pixel pattern
was simulated. Do not call this implementation an exact reproduction.

## Network comparison

| Component | Paper: unrestricted | Paper: restricted | Reduced reproduction |
|---|---|---|---|
| Input | 24 notch descriptors | 24 notch descriptors | 8 descriptors, scaled with training subset only |
| Hidden widths | 24, 300, 300, 300, 300 | 24, 500, 500, 500, 500 | 32, 64 |
| Hidden activation | ReLU | ReLU | ReLU |
| Dropout | After all 5 hidden layers | After first 4 hidden layers | None |
| Output | 1024 sigmoid pixel values | 48 sigmoid binary code values | 16 logits, reshaped as 4 categorical heads x 4 classes |
| Loss | Mean squared error | Mean squared error | Cross entropy |
| Optimizer / rate | Adam / 0.001 | Adam / 0.001 | Adam / 0.001 |
| Batch size | 30 | 30 | 16 |
| Epochs | Curves shown through 10,000 | Curves shown through 10,000 | Up to 500; early stopping after 30 unimproved validation epochs |
| Parâmetros pelas dimensões | 587,224 | 788,648 | 3,440 |
| Device | Tesla K80 in Colab | Tesla K80 in Colab | CUDA required by default; CPU used only for explicitly selected software tests |
| Reported accuracy | About 92% | About 91% | Measured on held-out data; no promised accuracy |
| Baseline | Not reproduced here | Not reproduced here | Nearest neighbour on the same scaled training descriptors |
| Physical assessment | Selected designs simulated again | Selected designs simulated again | Export one NN design; report cached comparison separately from fresh FDTD validation |

Paper Table 4 reports 84/81 minutes training, 0.055/0.052 seconds prediction,
and 7/9 MB for unrestricted/restricted networks. These measurements and its
accuracy definition are not directly comparable to this reduced classifier.
The exact eight tile masks, dropout probability and notch-width convention
are not fully specified for a faithful reimplementation.

## Four tile definitions and placement

| ID | Tile | Outer side | Ring width |
|---|---|---|---|
| 0 | Empty | 0 | 0 |
| 1 | Small square ring | 4 pixels = 0.8 mm | 1 pixel = 0.2 mm |
| 2 | Large square ring | 6 pixels = 1.2 mm | 1 pixel = 0.2 mm |
| 3 | Thick square ring | 6 pixels = 1.2 mm | 2 pixels = 0.4 mm |

Each ring is centred in its 8 x 8 tile. Rows are `[A,B,B,A]`, `[C,D,D,C]`,
`[C,D,D,C]`, `[A,B,B,A]`. CSV rows increase in y; previews use origin lower.
All exported dimensions use SI metres. Ground lies below the substrate; top
pattern begins at z=0. Unit cells repeat infinitely in x/y, with PML along z,
normal incidence, propagation -z and x polarization. This is not a finite array.

## Implementation and verification rules

- Configuration lives in `config.py`; numbered entry scripts accept no CLI arguments.
- Stage 1 (`01_generate_geometries.py`) saves the geometry manifest offline.
- Stage 2 (`02_simulate_geometries.py`) requires licensed Lumerical and `lumapi`,
  consumes that complete manifest and saves raw spectra without extracting notches.
- Stage 3 (`03_train.py`) prepares `training_dataset.npz` from saved raw spectra
  and trains offline. Stage 4 (`04_test_and_export.py`) exports one design offline,
  with optional fresh Lumerical validation.
- `geometry_checkpoint.npz`, `simulation_checkpoint.npz` and `last.pt` save
  progress after every geometry, simulation and minibatch, respectively.
- See `JUSTIFICATIVA_REDUCAO.md` and `docs/ANALISE_REDES.md`. Do not equate
  2,000 sampled training records with exhaustive exploration of 281 trillion layouts.
  Do not assert that the K80 is faster than an unmeasured user workstation.
- Table 1 lists 90,300 parameters for the 24 -> 300 dense layer. Its dimensions
  imply 7,500. The dimension-derived unrestricted total is 587,224, whereas the
  sum of published counts is 670,024. Report this discrepancy explicitly.
- Keep synthetic test fixtures explicitly marked. Never report their accuracy,
  spectra or exported designs as electromagnetic reproduction results.
- Save generation after every geometry and training after every minibatch,
  including weights, Adam state, order/cursor, RNG states, split and scaler.
  Keep checksums, atomic replacement and previous-snapshot backups.
- Use only one process per experiment directory. Repeat the same command after
  a crash; changes to physics/data/training identity require new run directories.
- Validate the empty reference, PEC mirror, source-band coverage, material fit,
  shutoff and mesh/time refinement before physical data collection.
- A coarse copper mesh may be inadequate for skin-depth losses. Convergence is
  a numerical check at the selected pilot, not proof of every layout's accuracy.
  Tighten copper mesh for final validation or explicitly choose PEC for a pilot;
  PEC is a different physical scenario and needs a separate dataset.
- Report layout accuracy and physical notch errors separately. Reflection depth
  uses 10*log10(power), equivalent to 20*log10(amplitude). Use total reflection,
  including cross polarization, when discussing absorption from a grounded surface.
- Run `python -m pytest -q` after functional changes. CUDA default is deliberate:
  do not silently fall back to CPU. GPU numerical equality across versions/devices
  is not guaranteed. Tests explicitly use CPU and check exact resume equality.
- Add useful Google-style docstrings to functions. No argparse, future annotations,
  hidden solver connections during training, or fabricated physical test evidence.

The deadline-oriented scope is a functional reduced workflow and one design check,
not reproducing the full paper's training set, broad target coverage or results.
