# Inferred metasurfaces compared with dataset geometries

This report evaluates the `real_paper` results added in commit `c2db6f6`. The workflow approximates the paper's restricted network and tile family. Its solver and tile masks differ from the paper's, so its accuracy is not a direct replication result.

## Dataset and model

The imported dataset contains 302 spectra, of which 175 passed the repository's notch eligibility rules. The split has 122 training, 26 validation, and 27 test samples. The saved model completed 31 epochs with early stopping. Its exported design has **not** undergone fresh Lumerical validation.

The source spectra are labeled as externally imported Lumerical results, but their S11 trace does not specify whether it includes cross polarization. The source did not supply pixel masks. The images below render dataset tile IDs and inferred tile IDs using the repository's explicitly approximated masks.

## Geometry results

| Split | Samples | Bit accuracy | Tile accuracy | Exact 16-tile layouts | Distinct inferred layouts |
|---|---:|---:|---:|---:|---:|
| Training | 122 | 52.0% | 13.0% | 0 | 8 |
| Validation | 26 | 46.6% | 12.0% | 0 | 4 |
| Test | 27 | 52.4% | 13.2% | 0 | 6 |

The same weak geometry recovery appears in training and held-out data. One inferred layout occurs in 18 of the 27 test cases. With eight tile classes, 12.5% is the accuracy of a uniformly random tile guess. That reference does not account for class frequencies, but shows the scale of the observed 13.2% test tile accuracy.

### Training examples

Each PNG shows the dataset geometry, inference, and a tile mismatch map. A dark mismatch tile marks a different tile ID.

| Dataset row | Matching tiles | Image |
|---|---:|---|
| 95 | 1/16 | [Training row 95](training_row_95.png) |
| 225 | 4/16 | [Training row 225](training_row_225.png) |
| 147 | 6/16 | [Training row 147](training_row_147.png) |

### Held-out examples

| Dataset row | Matching tiles | Image |
|---|---:|---|
| 207 | 1/16 | [Test row 207](heldout_row_207.png) |
| 253 | 3/16 | [Test row 253](heldout_row_253.png) |
| 82 | 2/16 | [Test row 82](heldout_row_82.png) |

## Comparison with Ghorbani et al.

Ghorbani et al. report about 91% accuracy for their restricted network and show four designs whose re-simulated responses agree with their targets (Figures 5 and 6). Their accuracy definition is not established as equivalent to this repository's bit, tile, or exact-layout metrics. Their training set contains 2,000 simulated examples, while this run has 175 eligible imported spectra. The solver, exact tile masks, notch extraction, and stopping procedure also differ.

**Assessment:** This run executed the data preparation, training, and geometry export workflow, but it did not recover the dataset layouts reliably, even on training samples. Geometry disagreement alone cannot prove spectral failure because different geometries may yield similar responses. A fresh simulation of the inferred design, followed by notch-count, frequency, depth, and width comparison against its target, is still required to assess physical inverse-design success.

## Evidence

- [`dataset_metrics.json`](../data/dataset_metrics.json): provenance and eligible-sample counts.
- [`training_metrics.json`](../training/training_metrics.json): split sizes and training history.
- [`report.json`](../design/report.json): held-out geometry scores and physical-validation status.
- [`s41598-021-86588-2.pdf`](../../../../paper/s41598-021-86588-2.pdf): original paper, particularly Figures 5–6 and Table 4.
