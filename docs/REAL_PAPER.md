# Paper-scale restricted workflow

This workflow implements the **restricted** architecture reported by Ghorbani et al.
It is a paper-scale approximation, not an exact reproduction. The paper used CST,
whereas this repository uses Lumerical FDTD. No physical results from this
workflow have been collected in this environment.

Each of 2,000 seeded layouts has sixteen independent 8 × 8 tile choices in a
4 × 4 arrangement. The eight masks are explicit arrays saved in the geometry
checkpoint. Their IDs define six centered square rings with `(outer side,
ring width)` in pixels: `(4,1)`, `(6,1)`, `(6,2)`, `(8,1)`, `(8,2)`, `(8,3)`.
IDs 6 and 7 are diamond rings whose Manhattan distance from the tile center
is in `[3,4]` and `[5,6]`, respectively. These approximate the visual pattern
family in Figure 1; the original exact 8 × 8 masks are not specified in the
paper. The random seed is 2026, and duplicate sampled layouts are redrawn.

Reflection notches are contiguous co-polar power intervals below −10 dB. Each
resolved notch contributes frequency in GHz, minimum depth in dB, and
interpolated −10 dB width in GHz. Up to eight triples form 24 inputs; unused
slots are zero. Empty, truncated, unresolved, and more-than-eight-notch
spectra remain saved with exclusion reasons. The paper does not fully define
its width, padding, or filtering conventions. Scaling is fitted on training
rows only.

The restricted model has hidden widths 24, 500, 500, 500, 500, ReLU, four
dropout layers, and 48 sigmoid outputs interpreted as sixteen 3-bit tile IDs.
The assumed dropout rate is 0.5. It uses MSE, Adam 0.001, batch size 30,
70% training and 30% held-out testing, and a 10,000-epoch limit without an
invented validation stopping rule. The dimension-derived parameter count is
788,648. Its reported bit and layout accuracy are defined here and cannot be
compared directly to the paper's accuracy without the original metric.

The real and dummy reduced workflows have separate manifests, spectra,
training checkpoints, and exports. Paper-scale checkpoints are confined to
`runs/real_paper/`. A `test_fixture` backend is allowed only with an explicit
software-test option and is never labeled as a physical result.
