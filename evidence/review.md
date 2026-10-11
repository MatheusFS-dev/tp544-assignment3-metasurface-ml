# Independent review

A fresh reviewer inspected the reduced pipeline, reproduced consequential issues,
and re-reviewed the fixes. Identified issues:

1. Python lumapi E results are vector datasets. Fixed extraction to use the final
   component axis; regression uses documented E array shape and nonuniform mesh.
2. Recovery rotated a corrupt primary over its good backup. Fixed restoration
   to preserve the backup; tested repeated corruption and interrupted replacement.
3. Repeated export could retain stale physical files. Invalidate previous FSP,
   validation spectrum and report before writing the current geometry.

Review regressions were observed failing before the fixes (`review-red.log`),
then passed (`review-green.log`). Export cleanup passed in `export-green.log`.
Reviewer independently ran 17 targeted tests after fixes, all passing, and found
no remaining code-review blockers. Actual Lumerical and GPU runs remain unverified.

Production API reference used for the vector-field fix:
https://optics.ansys.com/hc/en-us/articles/39744236202771-Accessing-Simulation-Results-Python-API
