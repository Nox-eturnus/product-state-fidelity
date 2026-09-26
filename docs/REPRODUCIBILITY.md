# Reproducibility and implementation guide

## Scientific modules

| Module in `code/` | Purpose |
| --- | --- |
| `product_state_estimator.py` | HiGHS linear-program score design, rational residual/range certificates, outward confidence arithmetic, and full-copy streaming |
| `product_state_geometry.py` | Partial Pauli approximation, rigorous geometry bias enclosures, deterministic selection, and streaming |
| `homogeneous_fidelity.py` | Bounded rational complex-score construction and second-moment certificates |
| `product_state_lp_prototype.py` | Small exact vertex-enumeration reference for the deliberately degraded half-signal channel |
| `finite_size_evaluation.py` | Full finite-size comparison and resource diagnostics |
| `extended_evaluation.py` | Higher-repetition comparison on a fixed correlated-mixture input |
| `intermediate_geometry_evaluation.py` | Fixed intermediate-geometry comparison |

The flat `code/` layout allows scripts to import one another directly. Data and output paths are resolved relative to each script's location, so scientific scripts do not depend on the invoking working directory. User-specified plot output directories are relative to the invoking working directory.

## Validation

First check the unchanged distribution with the standard-library-only command:

```bash
python scripts/check_integrity.py
```

Run the independent algebra and estimator checks:

```bash
python code/verify_population_recovery.py
python code/verify_product_state_boundary.py
python code/product_state_lp_prototype.py
python code/verify_product_state_estimator.py
python code/verify_product_state_geometry.py
python code/verify_homogeneous_fidelity.py
```

The first three use the Python standard library. The exact vertex-enumeration reference is intentionally limited to tiny instances and can be slower than the LP solver. The remaining checks require the numerical dependencies.

Validate the recorded experiment datasets:

```bash
python code/verify_finite_size_evaluation.py
python code/verify_extended_evaluation.py
python code/verify_intermediate_geometry.py
```

Verification overwrites the corresponding `results/*_checks.json` report. Timings, allocation measurements, environment information, and check-report bytes may change. The supplied manifest remains a checksum of the original distribution and will correctly report these deliberate changes. Do not update its hashes to conceal unexpected differences.

Finite-dimensional Born checks test algebra and implementation behavior. They do not prove asymptotic theorems. Statistical spot checks retain their original scope and do not establish empirical optimality.

## Full experiment reproduction

Run from a separate checkout to preserve supplied data:

```bash
python code/finite_size_evaluation.py --run
python code/extended_evaluation.py --run
python code/intermediate_geometry_evaluation.py
```

Run the finite-size driver first: the extended driver uses its design and recorded source provenance. The intermediate driver uses the same scoring routines with its own fixed design. Drivers write to `results/`; the intermediate driver also regenerates `data/intermediate_geometry_design.json`. Full reruns can be substantially more expensive than the read-and-replay verifiers. No dependency download or network access is needed after installation.

Random seeds and raw repetition means are stored. The main study uses its recorded 32-bit seed derivation; extended and intermediate evaluations use full 128-bit seeds and verify within-row uniqueness. Do not interchange these streams. Exact repeatability of solver-dependent score choices may depend on the pinned solver and numerical environment. Timings and memory measurements are environment-dependent.

The main result file's `design_errata` records clarifications to the fixed design, including native inversion's arbitrary-input unbiasedness and the transposed inverse-channel equation. These clarifications control interpretation of older descriptive text in the design; the experiment parameters remain fixed.

## Data and figures

| Figure | Source | Generator |
| --- | --- | --- |
| `measurement_scheme.png` | Theoretical measurement schematic | `scripts/plot_finite_size.py` |
| `finite_size_comparison.png` | 32 selected rows of `finite_size_results.json` | `scripts/plot_finite_size.py` |
| `extended_comparison.png` | Four displayed methods from the 64-row `extended_results.json` | `scripts/plot_extended.py` |
| `moment_comparison.png` | Analytic moment formulas, without simulation or fitting | `scripts/plot_moments.py` |

The comparison plots show confidence-radius calculations and empirical 90th-percentile absolute errors at `N=4096`, correlated-mixture fidelity `F=0.7`, and `n=1,2,4,6`. Native/shadow numerical bounds are distinguished from certified LP/geometry bounds. The extended evaluation uses 10,000 repetitions per row; the main finite-size figure uses 100. The moment comparison shows second moments, not necessary high-probability copy counts.

Supporting JSON files under `figures/` retain the plotted numbers. The comparison generators cross-check selected values against their supporting JSON. Both main datasets still contain all eight methods, regardless of which four are displayed.

```bash
python scripts/plot_finite_size.py --output-dir reproduced_figures
python scripts/plot_extended.py --output-dir reproduced_figures
python scripts/plot_moments.py --output-dir reproduced_figures
```

Omit `--output-dir` to replace files in `figures/`. Font and renderer versions can affect image bytes without changing the underlying numbers.

## Provenance and integrity

This distribution reorganizes the source files and replaces development-oriented labels with scientific descriptions. Recorded numerical results, seeds, trial means, confidence intervals, failures, and resource diagnostics are preserved. Reorganization is not represented as a new full experiment execution.

For the supplied result files, `source_hashes` and `frozen_source_hashes` retain original-run hashes under the current descriptive module names. `distribution_source_hashes` and `distribution_frozen_source_hashes` identify the reorganized source actually shipped here. The independent verifiers check distribution hashes when present, otherwise the runtime hashes produced by a fresh execution. Design presentation metadata was normalized; original design hashes remain in `original_design_file_sha256` and `original_design_canonical_sha256`, while active design hashes match the included files. Experiment parameters are unchanged.

`manifest.json` records SHA256 and byte counts for all distributed files except itself. The integrity checker rejects modified, missing, unsafe, or unexpectedly added files, while ignoring Python caches, local environments, and the documented `reproduced_figures/` output directory. A passing manifest check establishes file consistency, not scientific validity or authorship.

## Scope and limitations

The target is a known pure product state. Estimation guarantees apply to arbitrary i.i.d. mixed inputs under ideal local Pauli axis choice and randomization, retaining full outcome strings. A product target does not imply a product unknown input. Proxy scores multiply exact stabilizer flags using the all-copy denominator, without rare-event postselection.

The LP/common and LP/half methods have rational certificates. Inverse-common certificates include channel-enclosure coupling error. Native and canonical-shadow ranges are evaluated numerically without rigorous finite-precision confidence certification. The naive Pauli proxy is a reference with no fidelity confidence interval. These distinctions are retained in each dataset and the finite-size diagnostic report.

The homogeneous complex-score code is a bounded exact-rational prototype, and geometry selection is a sufficient strategy rather than a minimax-optimal design. The fixed-copy simulations do not establish general useful large-system accuracy. Infeasible count-only cases are retained. Population-recovery reductions and finite-dimensional numerical checks do not constitute a proof of all asymptotic statements by execution.

Runtime, preprocessing, solver effort, bit precision, score ranges, copy count `N`, and local measurement count `nN` must be interpreted separately. Allocation figures exclude native-library memory where stated. Multinomial score-law simulation has the distribution of aggregated independent copy scores but does not measure physical callback runtime. Hardware noise and probability-calibration errors need their own error budget.
