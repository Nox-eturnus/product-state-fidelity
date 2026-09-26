# Finite-size evaluation — frozen diagnostic report 

Date: 2026-09-09. Status: recorded finite-size diagnostics.  No new theory or literature claims. 

Design: version 20260909-phase5-v1 eps=0.05 (1/20) delta=1/3 seed=20260909 reps=100 N=[256, 4096] n=[1, 2, 4, 6].
Design file SHA256: `c1afc09ecb83c62775dcddf1ae8a23e926dd5904d0d10c1b1be46ddc7ba7cbb8` (frozen byte-for-byte). Canonical SHA256: `e5c90e200b6bbcadbac2ef3ddea79f9b2d255beb55a772d914323a0d8b07c304`.
Python 3.13.13 NumPy 2.5.3 SciPy 1.17.1. Sources: {"finite_size_evaluation.py": "1beb94c6bf65adf376a918457eedad0cda35bd4bc8b771f4e14f5f8d3d61b928", "homogeneous_fidelity.py": "08d1ff3fce1234402d49aaf671e24628d0bdf6baeb96e689f0b4108bbd423ca1", "product_state_estimator.py": "a83d2b7fed709829891ef83bf4fb9799d838dd5d7b12fcfb232f02e05d2eba16", "product_state_geometry.py": "c9a53045abb792fed8e61a52aa4643010b76946019a72a914f200be2005978f7", "verify_finite_size_evaluation.py": "13e1308976b378b3ee0a7b6d4483430d593383ab69bf88bb891e882bc0d6f81c"}.

Scope: diagnostic finite-size evaluation only. Certificates retain arbitrary iid mixed-input scope (R-diagonal reduction, all-copy denominator, no postselection). Test families (target product F=1, orthogonal product F=0, maximally mixed F=1/2^n, correlated mixture 0.7 target +0.3 all-opposite F=0.7, n=2 Bell Phi+ joint Born) are examples only and do NOT establish worst-case performance. Copy cost N only; nN locals, preprocessing/certification/sampling times, Python peak allocations (native HiGHS excluded), dense-law size, coefficient precision/rounding/conditioning reported separately. No total-runtime theorem. No hardware guarantee (ideal probabilities, finite RNG/control rounding outside certificates).

## Methods and observation model

Per copy product Pauli basis, one measurement per qubit, full n outcome bits retained. Across-copy adaptivity allowed in lowers; upper nonadaptive. Forbidden: entangling preprocessing, joint measurements, non-Pauli axes, repeat within copy.

- `lp_common` (theorem default, signal 1/sqrt(3), eta irrational with 80-bit enclosure): weighted axes |r_hat|/L_hat for K, deterministic stabilizers, score H=g[w]*F, all-copy denominator, production Hoeffding R=b+G*sqrt(2 ln(2/delta)/N) (Fractions exact upper, outward float), k=0 Bernoulli. Certified under ideal probabilities.
- `lp_half` (explicit rational signal 1/2, eta=1/4): same form, intentionally noisier, same exponent shape. Never silently substituted.
- `inv_common`/`inv_half`: inverses solving W^T g=e0 exactly (Fractions Gaussian elimination, transpose corrected) on same degraded channels; score H=g_inv[w]*F is exact rational. Exactly unbiased on rational W_mid channel; on true irrational channel bias <=2Gk max_eta_error, included in radius and eps-b slack via existing exact production helpers hoeffding_radius_exact/sufficient_copies_conservative (Bernoulli helpers for k=0). radius_exact is str(Fraction), display via _float_upper; exact bias/range stored with fresh W^T residual check at the appropriate (mid/half) channel. Rigorous score confidence is under ideal probabilities and is distinct from the floating multinomial Monte Carlo law (diagnostic only). k=0 reduces to [1].
- `native`: weighted local inverse product prod_j (1+L_j*s_j)/2 for noisy times Pauli indicator for S; varying native noise (no degradation), distinct from degraded common/half. Unbiased for EVERY mixed/entangled input (proof below); target-only global range from all 2^n sign patterns (identical across states, includes zero-prob observations); mathematical bias 0 for radius/counts (observed E-F is floating diagnostic only). Bounds/counts are numerical evaluations of analytic Hoeffding formulas in binary64 with ordinary ceil; no rigorous finite-precision certificate is claimed (no nextafter/padding proof). k=0 reduces to all-match indicator.
- `shadow`: canonical uniform-Pauli shadow prod_j (1+3*r_{axis}*o)/2; uniform 1/3 axes for ALL qubits including stabilizers (different from weighted/deterministic LP/native/naive); unbiased for any state via linearity; numerical evaluation of global Hoeffding formula with ordinary ceil; no rigorous finite-precision certificate is claimed; NOT claimed optimal. No optimized-unbiased comparator implemented (no Caprotti 1.15-1.5 or Cha-Lee minimax implementation/general claim).
- `naive`: deterministic nearest signed Pauli axes (dominant, tie X<Y<Z), all-matching sign-bit indicator (function of full bitstrings, not parity alone); estimates proxy Q fidelity, not P; exact per-row bias Tr(Q sigma)-F with worst-case d=||P-Q|| (full-D geometry); NO zero-bias fidelity confidence claim (Bernoulli radius for Q only as reference, not added). For all-Pauli coincides with LP k=0 (no advantage claimed).
- `geometry`: deterministic selector beta=eps/2=0.025 via select_discards before data; proxy P_D with exact operator bias d=sqrt(1-prod_D (1+a)/2); inner accepted LP common on proxy; total bias b_inner+d_hi, radius R_inner+d_hi (Fractions exact); certified under ideal probabilities; analytic float evaluation distinguished from fresh rational production certificate by recomputation path.

Binary parity alone (even/odd Hamming weight, single bit) is NOT a fidelity estimator and is excluded. All primary methods retain full local Pauli bitstrings: LP/inverse retain weight w plus all-copy flag F (not parity); native retains per-qubit sign-corrected bits; shadow retains (axis,outcome) pairs; naive retains per-qubit match bits. Parity-only would discard weight information for LP/inverse certificates.

Controlling design errata (design bytes frozen, this report controls): Controlling errata (design bytes frozen): N grid inherited original task; native per-copy product unbiased for arbitrary inputs (design product-promise-only text superseded; see report proof and Bell 4/9 vs marginal-product 1/4); inverse is (W^T)^{-1}e0 i.e. W^T g=e0 (design W^{-1} shorthand corrected); near-axis stored floats approximate named rational and define exact normalized-float target, not exact proportionality. Manifest provenance preserved; targets/cases unchanged.

Native unbiasedness proof (compact, arbitrary inputs): per qubit j fix target Bloch r_j (|r_j|=1), L_j=|r_j|_1. Sample axis i with p_i=|r_{ji}|/L_j, measure Pauli_i outcome o=+-1, sign-correct s=sign(r_{ji})o, factor Z_j=(1+L_j s)/2. For fixed axis i with A=Pauli_i, s_i=sign(r_{ji}), sum_o Z_j(axis,o)(I+oA)/2 = (I+L_j s_i A)/2 (direct o=+-1 sum). Averaging over i gives sum_i p_i(I+L_j s_i Pauli_i)/2 = (I+sum_i r_{ji}Pauli_i)/2 = (I+R_j)/2 = tau_{+,j}. Axes independent across qubits, so joint POVM-averaged operator is tensor_j tau_{+,j}=P. Hence for ANY sigma (mixed/entangled), E[H]=Tr(sigma P)=F. Different from product of marginal means: Bell Phi+ with target (2/3,1/3,2/3)^2 has joint F=4/9 while each marginal is I/2 with per-qubit overlap 1/2 and marginal product 1/4; the per-copy product score attains 4/9, the marginal-product method attains 1/4 and fails without a product promise.

Global ranges: native target-only from all 2^n sign products (1+-L_j)/2; shadow target-only from all 6^n (axis,outcome) products (1+3 r_{axis}o)/2. Computed before simulation, identical across states at same target/N/method, including zero-probability observations under any single test state. Example: n=2 Pauli_Z shadow global [-2,4] (not Bell-support [-0.5,4]); hom_near n=2 global about [-2,4] (not Bell-support [-0.503,4]). Inverse radius/counts are exact-rational via production helpers with midpoint coupling included (radius_exact str(Fraction)). Native/shadow radius/counts are numerical evaluations of analytic Hoeffding formulas with ordinary ceil; no rigorous finite-precision certificate is claimed (no nextafter/padding proof); fixed-grid high-precision Decimal checks are diagnostics only. No blanket claim that all method rows are production-certified: LP/geometry are production-certified, inverse is rigorous exact-rational, native/shadow are numerical-only, naive has no fidelity interval.

Distributional equivalence: finite per-copy score laws use honest full Born observations (weighted axes, joint Bell via 4x4 projectors for n=2, per-qubit Born for product families with shared latent for corr07, never independent marginals). Aggregated laws group by distinct score values; for EACH rep 0..99 an independent Generator seeded SHA256(master|tid_full|state|method|N|rep) mod 2^32 draws ONE Multinomial(N,p); this has same law as sum of N iid full-copy scores. Sampling AFTER design/scores frozen, separate from solver/proposal; full raw means and seed/mean digests stored for independent recomputation (no hidden vectorized rep=0 deviation). Operational run_streaming/run_geometry_streaming smoke regressions against correlated full-copy backend prove wrappers work; main 100-rep statistics use frozen multinomial diagnostics to avoid per-trial revalidation. Simulations are aggregated-law diagnostics, not measured runtime of N operational full-copy calls.

## Results (from saved finite_size_results.json)

Primary planned rows: 1360; ok: 1360; errors: 0. All planned rows accounted for, no selection after results. Fresh simulations reproducible excluding resource measurements (deterministic seeds). Strict JSON no NaN/Infinity (allow_nan=False).

### N=256 (eps=0.05, delta=1/3)

| n | target | state | lp_common RMSE / R / suff | lp_half RMSE / R | inv_half RMSE / R | native RMSE / R | shadow RMSE / R | naive RMSE / bias | geometry RMSE / R |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | hom_111 | target | 0.035/R=0.162/s=2753 | 0.037/R=0.177/s=3226 | 0.039/R=0.177/s=3226 | 0.036/R=0.102/s=1076 | 0.031/R=0.102/s=1076 | 0.210/b=-0.211 | 0.029/R=0.162/s=2753 |
| 1 | hom_111 | orth | 0.033/R=0.162/s=2753 | 0.027/R=0.177/s=3226 | 0.032/R=0.177/s=3226 | 0.024/R=0.102/s=1076 | 0.028/R=0.102/s=1076 | 0.212/b=+0.211 | 0.034/R=0.162/s=2753 |
| 1 | hom_111 | mixed | 0.056/R=0.162/s=2753 | 0.057/R=0.177/s=3226 | 0.049/R=0.177/s=3226 | 0.056/R=0.102/s=1076 | 0.053/R=0.102/s=1076 | 0.034/b=+0.000 | 0.051/R=0.162/s=2753 |
| 1 | hom_111 | corr07 | 0.060/R=0.162/s=2753 | 0.057/R=0.177/s=3226 | 0.056/R=0.177/s=3226 | 0.050/R=0.102/s=1076 | 0.047/R=0.102/s=1076 | 0.083/b=-0.085 | 0.050/R=0.162/s=2753 |
| 1 | hom_232 | target | 0.034/R=0.162/s=2753 | 0.032/R=0.177/s=3226 | 0.038/R=0.177/s=3226 | 0.032/R=0.099/s=996 | 0.033/R=0.118/s=1434 | 0.173/b=-0.167 | 0.033/R=0.162/s=2753 |
| 1 | hom_232 | orth | 0.025/R=0.162/s=2753 | 0.039/R=0.177/s=3226 | 0.036/R=0.177/s=3226 | 0.027/R=0.099/s=996 | 0.034/R=0.118/s=1434 | 0.168/b=+0.167 | 0.034/R=0.162/s=2753 |
| 1 | hom_232 | mixed | 0.052/R=0.162/s=2753 | 0.058/R=0.177/s=3226 | 0.063/R=0.177/s=3226 | 0.062/R=0.099/s=996 | 0.053/R=0.118/s=1434 | 0.030/b=+0.000 | 0.054/R=0.162/s=2753 |
| 1 | hom_232 | corr07 | 0.050/R=0.162/s=2753 | 0.066/R=0.177/s=3226 | 0.065/R=0.177/s=3226 | 0.052/R=0.099/s=996 | 0.048/R=0.118/s=1434 | 0.070/b=-0.067 | 0.053/R=0.162/s=2753 |
| 1 | hom_near | target | 0.042/R=0.162/s=2753 | 0.039/R=0.177/s=3226 | 0.041/R=0.177/s=3226 | 0.002/R=0.059/s=360 | 0.035/R=0.177/s=3226 | 0.000/b=-0.000 | 0.000/R=0.060/s=374 |
| 1 | hom_near | orth | 0.030/R=0.162/s=2753 | 0.040/R=0.177/s=3226 | 0.038/R=0.177/s=3226 | 0.001/R=0.059/s=360 | 0.031/R=0.177/s=3226 | 0.000/b=+0.000 | 0.000/R=0.060/s=374 |
| 1 | hom_near | mixed | 0.051/R=0.162/s=2753 | 0.064/R=0.177/s=3226 | 0.060/R=0.177/s=3226 | 0.031/R=0.059/s=360 | 0.057/R=0.177/s=3226 | 0.031/b=+0.000 | 0.032/R=0.060/s=374 |
| 1 | hom_near | corr07 | 0.053/R=0.162/s=2753 | 0.066/R=0.177/s=3226 | 0.068/R=0.177/s=3226 | 0.030/R=0.059/s=360 | 0.056/R=0.177/s=3226 | 0.029/b=-0.000 | 0.030/R=0.060/s=374 |
| 1 | mixed_core_tail | target | 0.032/R=0.162/s=2753 | 0.033/R=0.177/s=3226 | 0.037/R=0.177/s=3226 | 0.032/R=0.099/s=996 | 0.027/R=0.118/s=1434 | 0.170/b=-0.167 | 0.036/R=0.162/s=2753 |
| 1 | mixed_core_tail | orth | 0.028/R=0.162/s=2753 | 0.039/R=0.177/s=3226 | 0.037/R=0.177/s=3226 | 0.029/R=0.099/s=996 | 0.034/R=0.118/s=1434 | 0.168/b=+0.167 | 0.028/R=0.162/s=2753 |
| 1 | mixed_core_tail | mixed | 0.054/R=0.162/s=2753 | 0.058/R=0.177/s=3226 | 0.066/R=0.177/s=3226 | 0.051/R=0.099/s=996 | 0.053/R=0.118/s=1434 | 0.033/b=+0.000 | 0.047/R=0.162/s=2753 |
| 1 | mixed_core_tail | corr07 | 0.051/R=0.162/s=2753 | 0.066/R=0.177/s=3226 | 0.064/R=0.177/s=3226 | 0.052/R=0.099/s=996 | 0.055/R=0.118/s=1434 | 0.077/b=-0.067 | 0.053/R=0.162/s=2753 |
| 1 | pauli_Z | target | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.030/R=0.177/s=3226 | 0.000/b=+0.000 | 0.000/R=0.059/s=359 |
| 1 | pauli_Z | orth | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.032/R=0.177/s=3226 | 0.000/b=+0.000 | 0.000/R=0.059/s=359 |
| 1 | pauli_Z | mixed | 0.032/R=0.059/s=359 | 0.032/R=0.059/s=359 | 0.029/R=0.059/s=359 | 0.035/R=0.059/s=359 | 0.053/R=0.177/s=3226 | 0.030/b=+0.000 | 0.034/R=0.059/s=359 |
| 1 | pauli_Z | corr07 | 0.028/R=0.059/s=359 | 0.031/R=0.059/s=359 | 0.026/R=0.059/s=359 | 0.029/R=0.059/s=359 | 0.054/R=0.177/s=3226 | 0.027/b=+0.000 | 0.027/R=0.059/s=359 |
| 2 | hom_111 | target | 0.046/R=0.222/s=5135 | 0.059/R=0.266/s=7257 | 0.062/R=0.266/s=7257 | 0.042/R=0.140/s=2007 | 0.052/R=0.140/s=2007 | 0.378/b=-0.378 | 0.063/R=0.222/s=5135 |
| 2 | hom_111 | orth | 0.022/R=0.222/s=5135 | 0.029/R=0.266/s=7257 | 0.034/R=0.266/s=7257 | 0.023/R=0.140/s=2007 | 0.021/R=0.140/s=2007 | 0.045/b=+0.045 | 0.021/R=0.222/s=5135 |
| 2 | hom_111 | mixed | 0.063/R=0.222/s=5135 | 0.085/R=0.266/s=7257 | 0.071/R=0.266/s=7257 | 0.056/R=0.140/s=2007 | 0.066/R=0.140/s=2007 | 0.028/b=+0.000 | 0.061/R=0.222/s=5135 |
| 2 | hom_111 | corr07 | 0.068/R=0.222/s=5135 | 0.081/R=0.266/s=7257 | 0.078/R=0.266/s=7257 | 0.067/R=0.140/s=2007 | 0.061/R=0.140/s=2007 | 0.251/b=-0.251 | 0.066/R=0.222/s=5135 |
| 2 | hom_111 | bell | 0.060/R=0.222/s=5135 | 0.076/R=0.266/s=7257 | 0.078/R=0.266/s=7257 | 0.067/R=0.140/s=2007 | 0.062/R=0.140/s=2007 | 0.174/b=+0.167 | 0.058/R=0.222/s=5135 |
| 2 | hom_232 | target | 0.043/R=0.222/s=5135 | 0.072/R=0.266/s=7257 | 0.071/R=0.266/s=7257 | 0.045/R=0.131/s=1770 | 0.047/R=0.177/s=3226 | 0.301/b=-0.306 | 0.042/R=0.222/s=5135 |
| 2 | hom_232 | orth | 0.021/R=0.222/s=5135 | 0.030/R=0.266/s=7257 | 0.034/R=0.266/s=7257 | 0.016/R=0.131/s=1770 | 0.019/R=0.177/s=3226 | 0.030/b=+0.028 | 0.022/R=0.222/s=5135 |
| 2 | hom_232 | mixed | 0.058/R=0.222/s=5135 | 0.081/R=0.266/s=7257 | 0.073/R=0.266/s=7257 | 0.050/R=0.131/s=1770 | 0.065/R=0.177/s=3226 | 0.025/b=+0.000 | 0.061/R=0.222/s=5135 |
| 2 | hom_232 | corr07 | 0.062/R=0.222/s=5135 | 0.079/R=0.266/s=7257 | 0.087/R=0.266/s=7257 | 0.061/R=0.131/s=1770 | 0.068/R=0.177/s=3226 | 0.208/b=-0.206 | 0.066/R=0.222/s=5135 |
| 2 | hom_232 | bell | 0.061/R=0.222/s=5135 | 0.076/R=0.266/s=7257 | 0.075/R=0.266/s=7257 | 0.052/R=0.131/s=1770 | 0.058/R=0.177/s=3226 | 0.064/b=+0.056 | 0.058/R=0.222/s=5135 |
| 2 | hom_near | target | 0.048/R=0.222/s=5135 | 0.073/R=0.266/s=7257 | 0.067/R=0.266/s=7257 | 0.002/R=0.059/s=361 | 0.046/R=0.355/s=12901 | 0.000/b=-0.000 | 0.000/R=0.061/s=380 |
| 2 | hom_near | orth | 0.022/R=0.222/s=5135 | 0.029/R=0.266/s=7257 | 0.033/R=0.266/s=7257 | 0.000/R=0.059/s=361 | 0.022/R=0.355/s=12901 | 0.000/b=+0.000 | 0.000/R=0.061/s=380 |
| 2 | hom_near | mixed | 0.059/R=0.222/s=5135 | 0.075/R=0.266/s=7257 | 0.078/R=0.266/s=7257 | 0.028/R=0.059/s=361 | 0.058/R=0.355/s=12901 | 0.031/b=+0.000 | 0.028/R=0.061/s=380 |
| 2 | hom_near | corr07 | 0.069/R=0.222/s=5135 | 0.085/R=0.266/s=7257 | 0.080/R=0.266/s=7257 | 0.028/R=0.059/s=361 | 0.060/R=0.355/s=12901 | 0.030/b=-0.000 | 0.025/R=0.061/s=380 |
| 2 | hom_near | bell | 0.058/R=0.222/s=5135 | 0.079/R=0.266/s=7257 | 0.077/R=0.266/s=7257 | 0.031/R=0.059/s=361 | 0.063/R=0.355/s=12901 | 0.032/b=+0.000 | 0.030/R=0.061/s=380 |
| 2 | mixed_core_tail | target | 0.045/R=0.222/s=5135 | 0.060/R=0.266/s=7257 | 0.057/R=0.266/s=7257 | 0.026/R=0.099/s=998 | 0.051/R=0.266/s=7257 | 0.170/b=-0.167 | 0.031/R=0.163/s=2868 |
| 2 | mixed_core_tail | orth | 0.020/R=0.222/s=5135 | 0.037/R=0.266/s=7257 | 0.033/R=0.266/s=7257 | 0.001/R=0.099/s=998 | 0.021/R=0.266/s=7257 | 0.000/b=+0.000 | 0.000/R=0.163/s=2868 |
| 2 | mixed_core_tail | mixed | 0.068/R=0.222/s=5135 | 0.071/R=0.266/s=7257 | 0.073/R=0.266/s=7257 | 0.037/R=0.099/s=998 | 0.059/R=0.266/s=7257 | 0.027/b=+0.000 | 0.050/R=0.163/s=2868 |
| 2 | mixed_core_tail | corr07 | 0.072/R=0.222/s=5135 | 0.102/R=0.266/s=7257 | 0.077/R=0.266/s=7257 | 0.046/R=0.099/s=998 | 0.055/R=0.266/s=7257 | 0.119/b=-0.117 | 0.048/R=0.163/s=2868 |
| 2 | mixed_core_tail | bell | 0.065/R=0.222/s=5135 | 0.074/R=0.266/s=7257 | 0.068/R=0.266/s=7257 | 0.044/R=0.099/s=998 | 0.060/R=0.266/s=7257 | 0.173/b=-0.167 | 0.043/R=0.163/s=2868 |
| 2 | pauli_Z | target | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.048/R=0.355/s=12901 | 0.000/b=+0.000 | 0.000/R=0.059/s=359 |
| 2 | pauli_Z | orth | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.019/R=0.355/s=12901 | 0.000/b=+0.000 | 0.000/R=0.059/s=359 |
| 2 | pauli_Z | mixed | 0.027/R=0.059/s=359 | 0.029/R=0.059/s=359 | 0.028/R=0.059/s=359 | 0.028/R=0.059/s=359 | 0.056/R=0.355/s=12901 | 0.026/b=+0.000 | 0.027/R=0.059/s=359 |
| 2 | pauli_Z | corr07 | 0.028/R=0.059/s=359 | 0.027/R=0.059/s=359 | 0.028/R=0.059/s=359 | 0.024/R=0.059/s=359 | 0.066/R=0.355/s=12901 | 0.029/b=+0.000 | 0.029/R=0.059/s=359 |
| 2 | pauli_Z | bell | 0.032/R=0.059/s=359 | 0.028/R=0.059/s=359 | 0.028/R=0.059/s=359 | 0.032/R=0.059/s=359 | 0.065/R=0.355/s=12901 | 0.029/b=+0.000 | 0.030/R=0.059/s=359 |
| 4 | hom_111 | target | 0.085/R=0.413/s=17957 | 0.215/R=0.456/s=- | 0.130/R=0.599/s=36737 | 0.090/R=0.261/s=6986 | 0.088/R=0.261/s=6986 | 0.611/b=-0.613 | 0.073/R=0.413/s=17957 |
| 4 | hom_111 | orth | 0.011/R=0.413/s=17957 | 0.042/R=0.456/s=- | 0.022/R=0.599/s=36737 | 0.012/R=0.261/s=6986 | 0.013/R=0.261/s=6986 | 0.003/b=+0.002 | 0.013/R=0.413/s=17957 |
| 4 | hom_111 | mixed | 0.057/R=0.413/s=17957 | 0.089/R=0.456/s=- | 0.079/R=0.599/s=36737 | 0.054/R=0.261/s=6986 | 0.051/R=0.261/s=6986 | 0.015/b=+0.000 | 0.056/R=0.413/s=17957 |
| 4 | hom_111 | corr07 | 0.112/R=0.413/s=17957 | 0.216/R=0.456/s=- | 0.146/R=0.599/s=36737 | 0.119/R=0.261/s=6986 | 0.104/R=0.261/s=6986 | 0.429/b=-0.429 | 0.119/R=0.413/s=17957 |
| 4 | hom_232 | target | 0.079/R=0.413/s=17957 | 0.201/R=0.456/s=- | 0.115/R=0.599/s=36737 | 0.097/R=0.234/s=5593 | 0.067/R=0.399/s=16328 | 0.518/b=-0.518 | 0.098/R=0.413/s=17957 |
| 4 | hom_232 | orth | 0.013/R=0.413/s=17957 | 0.036/R=0.456/s=- | 0.021/R=0.599/s=36737 | 0.010/R=0.234/s=5593 | 0.014/R=0.399/s=16328 | 0.002/b=+0.001 | 0.011/R=0.413/s=17957 |
| 4 | hom_232 | mixed | 0.055/R=0.413/s=17957 | 0.090/R=0.456/s=- | 0.081/R=0.599/s=36737 | 0.052/R=0.234/s=5593 | 0.053/R=0.399/s=16328 | 0.015/b=+0.000 | 0.053/R=0.413/s=17957 |
| 4 | hom_232 | corr07 | 0.118/R=0.413/s=17957 | 0.190/R=0.456/s=- | 0.165/R=0.599/s=36737 | 0.104/R=0.234/s=5593 | 0.121/R=0.399/s=16328 | 0.365/b=-0.362 | 0.119/R=0.413/s=17957 |
| 4 | hom_near | target | 0.070/R=0.413/s=17957 | 0.183/R=0.456/s=- | 0.121/R=0.599/s=36737 | 0.003/R=0.059/s=362 | 0.083/R=1.420/s=206409 | 0.000/b=-0.000 | 0.000/R=0.061/s=389 |
| 4 | hom_near | orth | 0.012/R=0.413/s=17957 | 0.023/R=0.456/s=- | 0.027/R=0.599/s=36737 | 0.000/R=0.059/s=362 | 0.009/R=1.420/s=206409 | 0.000/b=+0.000 | 0.000/R=0.061/s=389 |
| 4 | hom_near | mixed | 0.060/R=0.413/s=17957 | 0.094/R=0.456/s=- | 0.081/R=0.599/s=36737 | 0.014/R=0.059/s=362 | 0.055/R=1.420/s=206409 | 0.015/b=+0.000 | 0.015/R=0.061/s=389 |
| 4 | hom_near | corr07 | 0.113/R=0.413/s=17957 | 0.206/R=0.456/s=- | 0.146/R=0.599/s=36737 | 0.027/R=0.059/s=362 | 0.106/R=1.420/s=206409 | 0.025/b=-0.000 | 0.030/R=0.061/s=389 |
| 4 | mixed_core_tail | target | 0.077/R=0.413/s=17957 | 0.187/R=0.456/s=- | 0.157/R=0.599/s=36737 | 0.039/R=0.132/s=1777 | 0.097/R=0.799/s=65310 | 0.306/b=-0.306 | 0.053/R=0.223/s=5442 |
| 4 | mixed_core_tail | orth | 0.012/R=0.413/s=17957 | 0.025/R=0.456/s=- | 0.023/R=0.599/s=36737 | 0.000/R=0.132/s=1777 | 0.012/R=0.799/s=65310 | 0.000/b=+0.000 | 0.000/R=0.223/s=5442 |
| 4 | mixed_core_tail | mixed | 0.057/R=0.413/s=17957 | 0.092/R=0.456/s=- | 0.076/R=0.599/s=36737 | 0.026/R=0.132/s=1777 | 0.059/R=0.799/s=65310 | 0.013/b=+0.000 | 0.030/R=0.223/s=5442 |
| 4 | mixed_core_tail | corr07 | 0.111/R=0.413/s=17957 | 0.207/R=0.456/s=- | 0.159/R=0.599/s=36737 | 0.054/R=0.132/s=1777 | 0.111/R=0.799/s=65310 | 0.218/b=-0.214 | 0.062/R=0.223/s=5442 |
| 4 | pauli_Z | target | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.084/R=1.420/s=206411 | 0.000/b=+0.000 | 0.000/R=0.059/s=359 |
| 4 | pauli_Z | orth | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.012/R=1.420/s=206411 | 0.000/b=+0.000 | 0.000/R=0.059/s=359 |
| 4 | pauli_Z | mixed | 0.014/R=0.059/s=359 | 0.014/R=0.059/s=359 | 0.015/R=0.059/s=359 | 0.015/R=0.059/s=359 | 0.054/R=1.420/s=206411 | 0.015/b=+0.000 | 0.018/R=0.059/s=359 |
| 4 | pauli_Z | corr07 | 0.029/R=0.059/s=359 | 0.028/R=0.059/s=359 | 0.025/R=0.059/s=359 | 0.028/R=0.059/s=359 | 0.122/R=1.420/s=206411 | 0.029/b=+0.000 | 0.032/R=0.059/s=359 |
| 6 | hom_111 | target | 0.199/R=0.456/s=- | 0.220/R=0.499/s=- | 0.230/R=1.348/s=185980 | 0.121/R=0.487/s=24324 | 0.130/R=0.487/s=24324 | 0.761/b=-0.759 | 0.208/R=0.456/s=- |
| 6 | hom_111 | orth | 0.116/R=0.456/s=- | 0.039/R=0.499/s=- | 0.017/R=1.348/s=185980 | 0.006/R=0.487/s=24324 | 0.008/R=0.487/s=24324 | 0.001/b=+0.000 | 0.104/R=0.456/s=- |
| 6 | hom_111 | mixed | 0.106/R=0.456/s=- | 0.095/R=0.499/s=- | 0.098/R=1.348/s=185980 | 0.053/R=0.487/s=24324 | 0.038/R=0.487/s=24324 | 0.007/b=+0.000 | 0.101/R=0.456/s=- |
| 6 | hom_111 | corr07 | 0.166/R=0.456/s=- | 0.213/R=0.499/s=- | 0.218/R=1.348/s=185980 | 0.158/R=0.487/s=24324 | 0.169/R=0.487/s=24324 | 0.531/b=-0.532 | 0.173/R=0.456/s=- |
| 6 | hom_232 | target | 0.183/R=0.456/s=- | 0.226/R=0.499/s=- | 0.211/R=1.348/s=185980 | 0.140/R=0.415/s=17677 | 0.157/R=0.898/s=82658 | 0.664/b=-0.665 | 0.189/R=0.456/s=- |
| 6 | hom_232 | orth | 0.122/R=0.456/s=- | 0.034/R=0.499/s=- | 0.015/R=1.348/s=185980 | 0.005/R=0.415/s=17677 | 0.005/R=0.898/s=82658 | 0.000/b=+0.000 | 0.121/R=0.456/s=- |
| 6 | hom_232 | mixed | 0.115/R=0.456/s=- | 0.103/R=0.499/s=- | 0.104/R=1.348/s=185980 | 0.040/R=0.415/s=17677 | 0.049/R=0.898/s=82658 | 0.009/b=+0.000 | 0.097/R=0.456/s=- |
| 6 | hom_232 | corr07 | 0.153/R=0.456/s=- | 0.221/R=0.499/s=- | 0.237/R=1.348/s=185980 | 0.162/R=0.415/s=17677 | 0.167/R=0.898/s=82658 | 0.469/b=-0.466 | 0.138/R=0.456/s=- |
| 6 | hom_near | target | 0.198/R=0.456/s=- | 0.213/R=0.499/s=- | 0.259/R=1.348/s=185980 | 0.004/R=0.060/s=364 | 0.129/R=5.679/s=3302509 | 0.000/b=-0.000 | 0.000/R=0.062/s=397 |
| 6 | hom_near | orth | 0.112/R=0.456/s=- | 0.026/R=0.499/s=- | 0.020/R=1.348/s=185980 | 0.000/R=0.060/s=364 | 0.005/R=5.679/s=3302509 | 0.000/b=+0.000 | 0.000/R=0.062/s=397 |
| 6 | hom_near | mixed | 0.098/R=0.456/s=- | 0.089/R=0.499/s=- | 0.088/R=1.348/s=185980 | 0.008/R=0.060/s=364 | 0.057/R=5.679/s=3302509 | 0.006/b=+0.000 | 0.007/R=0.062/s=397 |
| 6 | hom_near | corr07 | 0.149/R=0.456/s=- | 0.220/R=0.499/s=- | 0.233/R=1.348/s=185980 | 0.028/R=0.060/s=364 | 0.166/R=5.679/s=3302509 | 0.026/b=-0.000 | 0.026/R=0.062/s=397 |
| 6 | mixed_core_tail | target | 0.199/R=0.456/s=- | 0.231/R=0.499/s=- | 0.250/R=1.348/s=185980 | 0.058/R=0.176/s=3165 | 0.116/R=2.396/s=587781 | 0.424/b=-0.421 | 0.075/R=0.304/s=10293 |
| 6 | mixed_core_tail | orth | 0.117/R=0.456/s=- | 0.032/R=0.499/s=- | 0.018/R=1.348/s=185980 | 0.000/R=0.176/s=3165 | 0.007/R=2.396/s=587781 | 0.000/b=+0.000 | 0.000/R=0.304/s=10293 |
| 6 | mixed_core_tail | mixed | 0.100/R=0.456/s=- | 0.092/R=0.499/s=- | 0.078/R=1.348/s=185980 | 0.021/R=0.176/s=3165 | 0.042/R=2.396/s=587781 | 0.008/b=+0.000 | 0.018/R=0.304/s=10293 |
| 6 | mixed_core_tail | corr07 | 0.147/R=0.456/s=- | 0.218/R=0.499/s=- | 0.231/R=1.348/s=185980 | 0.087/R=0.176/s=3165 | 0.143/R=2.396/s=587781 | 0.298/b=-0.295 | 0.085/R=0.304/s=10293 |
| 6 | pauli_Z | target | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.110/R=5.679/s=3302572 | 0.000/b=+0.000 | 0.000/R=0.059/s=359 |
| 6 | pauli_Z | orth | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.000/R=0.059/s=359 | 0.007/R=5.679/s=3302572 | 0.000/b=+0.000 | 0.000/R=0.059/s=359 |
| 6 | pauli_Z | mixed | 0.009/R=0.059/s=359 | 0.008/R=0.059/s=359 | 0.008/R=0.059/s=359 | 0.008/R=0.059/s=359 | 0.043/R=5.679/s=3302572 | 0.007/b=+0.000 | 0.007/R=0.059/s=359 |
| 6 | pauli_Z | corr07 | 0.027/R=0.059/s=359 | 0.031/R=0.059/s=359 | 0.029/R=0.059/s=359 | 0.030/R=0.059/s=359 | 0.144/R=5.679/s=3302572 | 0.027/b=+0.000 | 0.031/R=0.059/s=359 |

### N=4096 (eps=0.05, delta=1/3)

| n | target | state | lp_common RMSE / R / suff | lp_half RMSE / R | inv_half RMSE / R | native RMSE / R | shadow RMSE / R | naive RMSE / bias | geometry RMSE / R |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | hom_111 | target | 0.006/R=0.041/s=2753 | 0.010/R=0.044/s=3226 | 0.011/R=0.044/s=3226 | 0.008/R=0.026/s=1076 | 0.007/R=0.026/s=1076 | 0.212/b=-0.211 | 0.007/R=0.041/s=2753 |
| 1 | hom_111 | orth | 0.008/R=0.041/s=2753 | 0.007/R=0.044/s=3226 | 0.011/R=0.044/s=3226 | 0.008/R=0.026/s=1076 | 0.008/R=0.026/s=1076 | 0.211/b=+0.211 | 0.008/R=0.041/s=2753 |
| 1 | hom_111 | mixed | 0.012/R=0.041/s=2753 | 0.017/R=0.044/s=3226 | 0.015/R=0.044/s=3226 | 0.015/R=0.026/s=1076 | 0.015/R=0.026/s=1076 | 0.008/b=+0.000 | 0.013/R=0.041/s=2753 |
| 1 | hom_111 | corr07 | 0.012/R=0.041/s=2753 | 0.014/R=0.044/s=3226 | 0.018/R=0.044/s=3226 | 0.013/R=0.026/s=1076 | 0.011/R=0.026/s=1076 | 0.086/b=-0.085 | 0.013/R=0.041/s=2753 |
| 1 | hom_232 | target | 0.008/R=0.041/s=2753 | 0.009/R=0.044/s=3226 | 0.009/R=0.044/s=3226 | 0.008/R=0.025/s=996 | 0.006/R=0.030/s=1434 | 0.167/b=-0.167 | 0.008/R=0.041/s=2753 |
| 1 | hom_232 | orth | 0.007/R=0.041/s=2753 | 0.009/R=0.044/s=3226 | 0.010/R=0.044/s=3226 | 0.008/R=0.025/s=996 | 0.009/R=0.030/s=1434 | 0.167/b=+0.167 | 0.006/R=0.041/s=2753 |
| 1 | hom_232 | mixed | 0.014/R=0.041/s=2753 | 0.014/R=0.044/s=3226 | 0.015/R=0.044/s=3226 | 0.013/R=0.025/s=996 | 0.012/R=0.030/s=1434 | 0.009/b=+0.000 | 0.014/R=0.041/s=2753 |
| 1 | hom_232 | corr07 | 0.014/R=0.041/s=2753 | 0.016/R=0.044/s=3226 | 0.015/R=0.044/s=3226 | 0.013/R=0.025/s=996 | 0.011/R=0.030/s=1434 | 0.068/b=-0.067 | 0.012/R=0.041/s=2753 |
| 1 | hom_near | target | 0.009/R=0.041/s=2753 | 0.009/R=0.044/s=3226 | 0.009/R=0.044/s=3226 | 0.000/R=0.015/s=360 | 0.009/R=0.044/s=3226 | 0.000/b=-0.000 | 0.000/R=0.016/s=374 |
| 1 | hom_near | orth | 0.008/R=0.041/s=2753 | 0.008/R=0.044/s=3226 | 0.010/R=0.044/s=3226 | 0.000/R=0.015/s=360 | 0.007/R=0.044/s=3226 | 0.000/b=+0.000 | 0.000/R=0.016/s=374 |
| 1 | hom_near | mixed | 0.013/R=0.041/s=2753 | 0.016/R=0.044/s=3226 | 0.016/R=0.044/s=3226 | 0.008/R=0.015/s=360 | 0.014/R=0.044/s=3226 | 0.008/b=+0.000 | 0.008/R=0.016/s=374 |
| 1 | hom_near | corr07 | 0.013/R=0.041/s=2753 | 0.015/R=0.044/s=3226 | 0.014/R=0.044/s=3226 | 0.007/R=0.015/s=360 | 0.013/R=0.044/s=3226 | 0.007/b=-0.000 | 0.007/R=0.016/s=374 |
| 1 | mixed_core_tail | target | 0.007/R=0.041/s=2753 | 0.012/R=0.044/s=3226 | 0.010/R=0.044/s=3226 | 0.005/R=0.025/s=996 | 0.007/R=0.030/s=1434 | 0.166/b=-0.167 | 0.007/R=0.041/s=2753 |
| 1 | mixed_core_tail | orth | 0.006/R=0.041/s=2753 | 0.007/R=0.044/s=3226 | 0.010/R=0.044/s=3226 | 0.006/R=0.025/s=996 | 0.007/R=0.030/s=1434 | 0.167/b=+0.167 | 0.007/R=0.041/s=2753 |
| 1 | mixed_core_tail | mixed | 0.013/R=0.041/s=2753 | 0.016/R=0.044/s=3226 | 0.015/R=0.044/s=3226 | 0.012/R=0.025/s=996 | 0.014/R=0.030/s=1434 | 0.008/b=+0.000 | 0.015/R=0.041/s=2753 |
| 1 | mixed_core_tail | corr07 | 0.014/R=0.041/s=2753 | 0.015/R=0.044/s=3226 | 0.015/R=0.044/s=3226 | 0.013/R=0.025/s=996 | 0.013/R=0.030/s=1434 | 0.068/b=-0.067 | 0.013/R=0.041/s=2753 |
| 1 | pauli_Z | target | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.008/R=0.044/s=3226 | 0.000/b=+0.000 | 0.000/R=0.015/s=359 |
| 1 | pauli_Z | orth | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.008/R=0.044/s=3226 | 0.000/b=+0.000 | 0.000/R=0.015/s=359 |
| 1 | pauli_Z | mixed | 0.008/R=0.015/s=359 | 0.007/R=0.015/s=359 | 0.009/R=0.015/s=359 | 0.007/R=0.015/s=359 | 0.014/R=0.044/s=3226 | 0.007/b=+0.000 | 0.008/R=0.015/s=359 |
| 1 | pauli_Z | corr07 | 0.008/R=0.015/s=359 | 0.008/R=0.015/s=359 | 0.007/R=0.015/s=359 | 0.007/R=0.015/s=359 | 0.013/R=0.044/s=3226 | 0.008/b=+0.000 | 0.007/R=0.015/s=359 |
| 2 | hom_111 | target | 0.014/R=0.056/s=5135 | 0.018/R=0.067/s=7257 | 0.017/R=0.067/s=7257 | 0.009/R=0.035/s=2007 | 0.014/R=0.035/s=2007 | 0.378/b=-0.378 | 0.011/R=0.056/s=5135 |
| 2 | hom_111 | orth | 0.005/R=0.056/s=5135 | 0.009/R=0.067/s=7257 | 0.008/R=0.067/s=7257 | 0.006/R=0.035/s=2007 | 0.006/R=0.035/s=2007 | 0.045/b=+0.045 | 0.005/R=0.056/s=5135 |
| 2 | hom_111 | mixed | 0.015/R=0.056/s=5135 | 0.018/R=0.067/s=7257 | 0.019/R=0.067/s=7257 | 0.015/R=0.035/s=2007 | 0.017/R=0.035/s=2007 | 0.007/b=+0.000 | 0.016/R=0.056/s=5135 |
| 2 | hom_111 | corr07 | 0.017/R=0.056/s=5135 | 0.024/R=0.067/s=7257 | 0.021/R=0.067/s=7257 | 0.016/R=0.035/s=2007 | 0.018/R=0.035/s=2007 | 0.250/b=-0.251 | 0.015/R=0.056/s=5135 |
| 2 | hom_111 | bell | 0.015/R=0.056/s=5135 | 0.021/R=0.067/s=7257 | 0.020/R=0.067/s=7257 | 0.014/R=0.035/s=2007 | 0.014/R=0.035/s=2007 | 0.167/b=+0.167 | 0.015/R=0.056/s=5135 |
| 2 | hom_232 | target | 0.013/R=0.056/s=5135 | 0.017/R=0.067/s=7257 | 0.018/R=0.067/s=7257 | 0.012/R=0.033/s=1770 | 0.012/R=0.044/s=3226 | 0.306/b=-0.306 | 0.009/R=0.056/s=5135 |
| 2 | hom_232 | orth | 0.005/R=0.056/s=5135 | 0.008/R=0.067/s=7257 | 0.007/R=0.067/s=7257 | 0.005/R=0.033/s=1770 | 0.005/R=0.044/s=3226 | 0.028/b=+0.028 | 0.006/R=0.056/s=5135 |
| 2 | hom_232 | mixed | 0.016/R=0.056/s=5135 | 0.018/R=0.067/s=7257 | 0.020/R=0.067/s=7257 | 0.014/R=0.033/s=1770 | 0.015/R=0.044/s=3226 | 0.006/b=+0.000 | 0.014/R=0.056/s=5135 |
| 2 | hom_232 | corr07 | 0.017/R=0.056/s=5135 | 0.022/R=0.067/s=7257 | 0.021/R=0.067/s=7257 | 0.017/R=0.033/s=1770 | 0.018/R=0.044/s=3226 | 0.206/b=-0.206 | 0.017/R=0.056/s=5135 |
| 2 | hom_232 | bell | 0.015/R=0.056/s=5135 | 0.019/R=0.067/s=7257 | 0.021/R=0.067/s=7257 | 0.016/R=0.033/s=1770 | 0.015/R=0.044/s=3226 | 0.056/b=+0.056 | 0.014/R=0.056/s=5135 |
| 2 | hom_near | target | 0.010/R=0.056/s=5135 | 0.013/R=0.067/s=7257 | 0.020/R=0.067/s=7257 | 0.001/R=0.015/s=361 | 0.011/R=0.089/s=12901 | 0.000/b=-0.000 | 0.000/R=0.016/s=380 |
| 2 | hom_near | orth | 0.006/R=0.056/s=5135 | 0.007/R=0.067/s=7257 | 0.008/R=0.067/s=7257 | 0.000/R=0.015/s=361 | 0.004/R=0.089/s=12901 | 0.000/b=+0.000 | 0.000/R=0.016/s=380 |
| 2 | hom_near | mixed | 0.015/R=0.056/s=5135 | 0.017/R=0.067/s=7257 | 0.018/R=0.067/s=7257 | 0.006/R=0.015/s=361 | 0.015/R=0.089/s=12901 | 0.007/b=+0.000 | 0.007/R=0.016/s=380 |
| 2 | hom_near | corr07 | 0.016/R=0.056/s=5135 | 0.021/R=0.067/s=7257 | 0.020/R=0.067/s=7257 | 0.008/R=0.015/s=361 | 0.019/R=0.089/s=12901 | 0.008/b=-0.000 | 0.008/R=0.016/s=380 |
| 2 | hom_near | bell | 0.015/R=0.056/s=5135 | 0.019/R=0.067/s=7257 | 0.019/R=0.067/s=7257 | 0.008/R=0.015/s=361 | 0.014/R=0.089/s=12901 | 0.008/b=+0.000 | 0.007/R=0.016/s=380 |
| 2 | mixed_core_tail | target | 0.014/R=0.056/s=5135 | 0.015/R=0.067/s=7257 | 0.017/R=0.067/s=7257 | 0.007/R=0.025/s=998 | 0.012/R=0.067/s=7257 | 0.167/b=-0.167 | 0.008/R=0.042/s=2868 |
| 2 | mixed_core_tail | orth | 0.005/R=0.056/s=5135 | 0.009/R=0.067/s=7257 | 0.009/R=0.067/s=7257 | 0.000/R=0.025/s=998 | 0.006/R=0.067/s=7257 | 0.000/b=+0.000 | 0.000/R=0.042/s=2868 |
| 2 | mixed_core_tail | mixed | 0.014/R=0.056/s=5135 | 0.021/R=0.067/s=7257 | 0.019/R=0.067/s=7257 | 0.011/R=0.025/s=998 | 0.016/R=0.067/s=7257 | 0.006/b=+0.000 | 0.010/R=0.042/s=2868 |
| 2 | mixed_core_tail | corr07 | 0.018/R=0.056/s=5135 | 0.025/R=0.067/s=7257 | 0.018/R=0.067/s=7257 | 0.012/R=0.025/s=998 | 0.016/R=0.067/s=7257 | 0.117/b=-0.117 | 0.012/R=0.042/s=2868 |
| 2 | mixed_core_tail | bell | 0.017/R=0.056/s=5135 | 0.019/R=0.067/s=7257 | 0.021/R=0.067/s=7257 | 0.011/R=0.025/s=998 | 0.016/R=0.067/s=7257 | 0.167/b=-0.167 | 0.010/R=0.042/s=2868 |
| 2 | pauli_Z | target | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.011/R=0.089/s=12901 | 0.000/b=+0.000 | 0.000/R=0.015/s=359 |
| 2 | pauli_Z | orth | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.006/R=0.089/s=12901 | 0.000/b=+0.000 | 0.000/R=0.015/s=359 |
| 2 | pauli_Z | mixed | 0.007/R=0.015/s=359 | 0.006/R=0.015/s=359 | 0.007/R=0.015/s=359 | 0.007/R=0.015/s=359 | 0.013/R=0.089/s=12901 | 0.007/b=+0.000 | 0.008/R=0.015/s=359 |
| 2 | pauli_Z | corr07 | 0.008/R=0.015/s=359 | 0.007/R=0.015/s=359 | 0.008/R=0.015/s=359 | 0.007/R=0.015/s=359 | 0.018/R=0.089/s=12901 | 0.007/b=+0.000 | 0.007/R=0.015/s=359 |
| 2 | pauli_Z | bell | 0.007/R=0.015/s=359 | 0.007/R=0.015/s=359 | 0.009/R=0.015/s=359 | 0.008/R=0.015/s=359 | 0.016/R=0.089/s=12901 | 0.008/b=+0.000 | 0.009/R=0.015/s=359 |
| 4 | hom_111 | target | 0.026/R=0.104/s=17957 | 0.033/R=0.150/s=36737 | 0.032/R=0.150/s=36737 | 0.019/R=0.065/s=6986 | 0.023/R=0.065/s=6986 | 0.613/b=-0.613 | 0.022/R=0.104/s=17957 |
| 4 | hom_111 | orth | 0.003/R=0.104/s=17957 | 0.006/R=0.150/s=36737 | 0.007/R=0.150/s=36737 | 0.003/R=0.065/s=6986 | 0.002/R=0.065/s=6986 | 0.002/b=+0.002 | 0.003/R=0.104/s=17957 |
| 4 | hom_111 | mixed | 0.015/R=0.104/s=17957 | 0.023/R=0.150/s=36737 | 0.025/R=0.150/s=36737 | 0.016/R=0.065/s=6986 | 0.016/R=0.065/s=6986 | 0.004/b=+0.000 | 0.015/R=0.104/s=17957 |
| 4 | hom_111 | corr07 | 0.024/R=0.104/s=17957 | 0.040/R=0.150/s=36737 | 0.037/R=0.150/s=36737 | 0.025/R=0.065/s=6986 | 0.029/R=0.065/s=6986 | 0.428/b=-0.429 | 0.027/R=0.104/s=17957 |
| 4 | hom_232 | target | 0.024/R=0.104/s=17957 | 0.032/R=0.150/s=36737 | 0.032/R=0.150/s=36737 | 0.022/R=0.058/s=5593 | 0.023/R=0.100/s=16328 | 0.517/b=-0.518 | 0.022/R=0.104/s=17957 |
| 4 | hom_232 | orth | 0.003/R=0.104/s=17957 | 0.005/R=0.150/s=36737 | 0.005/R=0.150/s=36737 | 0.002/R=0.058/s=5593 | 0.003/R=0.100/s=16328 | 0.001/b=+0.001 | 0.003/R=0.104/s=17957 |
| 4 | hom_232 | mixed | 0.016/R=0.104/s=17957 | 0.023/R=0.150/s=36737 | 0.025/R=0.150/s=36737 | 0.013/R=0.058/s=5593 | 0.014/R=0.100/s=16328 | 0.004/b=+0.000 | 0.015/R=0.104/s=17957 |
| 4 | hom_232 | corr07 | 0.027/R=0.104/s=17957 | 0.037/R=0.150/s=36737 | 0.042/R=0.150/s=36737 | 0.026/R=0.058/s=5593 | 0.024/R=0.100/s=16328 | 0.363/b=-0.362 | 0.025/R=0.104/s=17957 |
| 4 | hom_near | target | 0.023/R=0.104/s=17957 | 0.033/R=0.150/s=36737 | 0.039/R=0.150/s=36737 | 0.001/R=0.015/s=362 | 0.014/R=0.355/s=206409 | 0.000/b=-0.000 | 0.000/R=0.017/s=389 |
| 4 | hom_near | orth | 0.003/R=0.104/s=17957 | 0.005/R=0.150/s=36737 | 0.007/R=0.150/s=36737 | 0.000/R=0.015/s=362 | 0.003/R=0.355/s=206409 | 0.000/b=+0.000 | 0.000/R=0.017/s=389 |
| 4 | hom_near | mixed | 0.017/R=0.104/s=17957 | 0.025/R=0.150/s=36737 | 0.024/R=0.150/s=36737 | 0.004/R=0.015/s=362 | 0.015/R=0.355/s=206409 | 0.003/b=+0.000 | 0.004/R=0.017/s=389 |
| 4 | hom_near | corr07 | 0.026/R=0.104/s=17957 | 0.039/R=0.150/s=36737 | 0.037/R=0.150/s=36737 | 0.007/R=0.015/s=362 | 0.025/R=0.355/s=206409 | 0.007/b=-0.000 | 0.007/R=0.017/s=389 |
| 4 | mixed_core_tail | target | 0.027/R=0.104/s=17957 | 0.034/R=0.150/s=36737 | 0.032/R=0.150/s=36737 | 0.013/R=0.033/s=1777 | 0.026/R=0.200/s=65310 | 0.304/b=-0.306 | 0.012/R=0.057/s=5442 |
| 4 | mixed_core_tail | orth | 0.003/R=0.104/s=17957 | 0.006/R=0.150/s=36737 | 0.006/R=0.150/s=36737 | 0.000/R=0.033/s=1777 | 0.003/R=0.200/s=65310 | 0.000/b=+0.000 | 0.000/R=0.057/s=5442 |
| 4 | mixed_core_tail | mixed | 0.017/R=0.104/s=17957 | 0.023/R=0.150/s=36737 | 0.024/R=0.150/s=36737 | 0.007/R=0.033/s=1777 | 0.015/R=0.200/s=65310 | 0.004/b=+0.000 | 0.007/R=0.057/s=5442 |
| 4 | mixed_core_tail | corr07 | 0.029/R=0.104/s=17957 | 0.041/R=0.150/s=36737 | 0.039/R=0.150/s=36737 | 0.016/R=0.033/s=1777 | 0.029/R=0.200/s=65310 | 0.215/b=-0.214 | 0.015/R=0.057/s=5442 |
| 4 | pauli_Z | target | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.024/R=0.355/s=206411 | 0.000/b=+0.000 | 0.000/R=0.015/s=359 |
| 4 | pauli_Z | orth | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.003/R=0.355/s=206411 | 0.000/b=+0.000 | 0.000/R=0.015/s=359 |
| 4 | pauli_Z | mixed | 0.004/R=0.015/s=359 | 0.004/R=0.015/s=359 | 0.003/R=0.015/s=359 | 0.004/R=0.015/s=359 | 0.014/R=0.355/s=206411 | 0.004/b=+0.000 | 0.004/R=0.015/s=359 |
| 4 | pauli_Z | corr07 | 0.007/R=0.015/s=359 | 0.007/R=0.015/s=359 | 0.007/R=0.015/s=359 | 0.007/R=0.015/s=359 | 0.027/R=0.355/s=206411 | 0.006/b=+0.000 | 0.007/R=0.015/s=359 |
| 6 | hom_111 | target | 0.041/R=0.193/s=63523 | 0.111/R=0.257/s=- | 0.054/R=0.337/s=185980 | 0.037/R=0.122/s=24324 | 0.039/R=0.122/s=24324 | 0.760/b=-0.759 | 0.035/R=0.193/s=63523 |
| 6 | hom_111 | orth | 0.002/R=0.193/s=63523 | 0.021/R=0.257/s=- | 0.004/R=0.337/s=185980 | 0.002/R=0.122/s=24324 | 0.001/R=0.122/s=24324 | 0.000/b=+0.000 | 0.002/R=0.193/s=63523 |
| 6 | hom_111 | mixed | 0.014/R=0.193/s=63523 | 0.052/R=0.257/s=- | 0.021/R=0.337/s=185980 | 0.014/R=0.122/s=24324 | 0.014/R=0.122/s=24324 | 0.002/b=+0.000 | 0.015/R=0.193/s=63523 |
| 6 | hom_111 | corr07 | 0.040/R=0.193/s=63523 | 0.115/R=0.257/s=- | 0.069/R=0.337/s=185980 | 0.043/R=0.122/s=24324 | 0.045/R=0.122/s=24324 | 0.532/b=-0.532 | 0.047/R=0.193/s=63523 |
| 6 | hom_232 | target | 0.032/R=0.193/s=63523 | 0.110/R=0.257/s=- | 0.044/R=0.337/s=185980 | 0.040/R=0.104/s=17677 | 0.040/R=0.225/s=82658 | 0.665/b=-0.665 | 0.041/R=0.193/s=63523 |
| 6 | hom_232 | orth | 0.002/R=0.193/s=63523 | 0.026/R=0.257/s=- | 0.005/R=0.337/s=185980 | 0.001/R=0.104/s=17677 | 0.001/R=0.225/s=82658 | 0.000/b=+0.000 | 0.002/R=0.193/s=63523 |
| 6 | hom_232 | mixed | 0.015/R=0.193/s=63523 | 0.057/R=0.257/s=- | 0.023/R=0.337/s=185980 | 0.012/R=0.104/s=17677 | 0.014/R=0.225/s=82658 | 0.002/b=+0.000 | 0.013/R=0.193/s=63523 |
| 6 | hom_232 | corr07 | 0.034/R=0.193/s=63523 | 0.120/R=0.257/s=- | 0.067/R=0.337/s=185980 | 0.034/R=0.104/s=17677 | 0.045/R=0.225/s=82658 | 0.465/b=-0.466 | 0.047/R=0.193/s=63523 |
| 6 | hom_near | target | 0.039/R=0.193/s=63523 | 0.100/R=0.257/s=- | 0.065/R=0.337/s=185980 | 0.001/R=0.015/s=364 | 0.036/R=1.420/s=3302509 | 0.000/b=-0.000 | 0.000/R=0.017/s=397 |
| 6 | hom_near | orth | 0.002/R=0.193/s=63523 | 0.013/R=0.257/s=- | 0.005/R=0.337/s=185980 | 0.000/R=0.015/s=364 | 0.001/R=1.420/s=3302509 | 0.000/b=+0.000 | 0.000/R=0.017/s=397 |
| 6 | hom_near | mixed | 0.013/R=0.193/s=63523 | 0.065/R=0.257/s=- | 0.026/R=0.337/s=185980 | 0.002/R=0.015/s=364 | 0.012/R=1.420/s=3302509 | 0.002/b=+0.000 | 0.002/R=0.017/s=397 |
| 6 | hom_near | corr07 | 0.043/R=0.193/s=63523 | 0.106/R=0.257/s=- | 0.068/R=0.337/s=185980 | 0.007/R=0.015/s=364 | 0.048/R=1.420/s=3302509 | 0.007/b=-0.000 | 0.007/R=0.017/s=397 |
| 6 | mixed_core_tail | target | 0.047/R=0.193/s=63523 | 0.123/R=0.257/s=- | 0.061/R=0.337/s=185980 | 0.015/R=0.044/s=3165 | 0.039/R=0.599/s=587781 | 0.421/b=-0.421 | 0.016/R=0.078/s=10293 |
| 6 | mixed_core_tail | orth | 0.002/R=0.193/s=63523 | 0.023/R=0.257/s=- | 0.004/R=0.337/s=185980 | 0.000/R=0.044/s=3165 | 0.001/R=0.599/s=587781 | 0.000/b=+0.000 | 0.000/R=0.078/s=10293 |
| 6 | mixed_core_tail | mixed | 0.013/R=0.193/s=63523 | 0.055/R=0.257/s=- | 0.026/R=0.337/s=185980 | 0.005/R=0.044/s=3165 | 0.014/R=0.599/s=587781 | 0.002/b=+0.000 | 0.005/R=0.078/s=10293 |
| 6 | mixed_core_tail | corr07 | 0.047/R=0.193/s=63523 | 0.111/R=0.257/s=- | 0.074/R=0.337/s=185980 | 0.023/R=0.044/s=3165 | 0.037/R=0.599/s=587781 | 0.295/b=-0.295 | 0.020/R=0.078/s=10293 |
| 6 | pauli_Z | target | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.032/R=1.420/s=3302572 | 0.000/b=+0.000 | 0.000/R=0.015/s=359 |
| 6 | pauli_Z | orth | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.000/R=0.015/s=359 | 0.001/R=1.420/s=3302572 | 0.000/b=+0.000 | 0.000/R=0.015/s=359 |
| 6 | pauli_Z | mixed | 0.002/R=0.015/s=359 | 0.002/R=0.015/s=359 | 0.002/R=0.015/s=359 | 0.002/R=0.015/s=359 | 0.012/R=1.420/s=3302572 | 0.002/b=+0.000 | 0.002/R=0.015/s=359 |
| 6 | pauli_Z | corr07 | 0.008/R=0.015/s=359 | 0.006/R=0.015/s=359 | 0.006/R=0.015/s=359 | 0.007/R=0.015/s=359 | 0.048/R=1.420/s=3302572 | 0.007/b=+0.000 | 0.007/R=0.015/s=359 |

Fair fixed-(eps,delta,N) columns; surrogate b+G/sqrt(N) shown separately in results (not equated with certified accuracy). Range, certificate bias, radius, sufficient copies for FROZEN score (not minimax, not reoptimized N) per row. Certification status retained: many LP/geometry radii exceed eps at frozen N (especially k>=4, N=256); inverse/shadow/native global Hoeffding radii large (range >>1); naive has no fidelity interval; homogeneous counts astronomical (count-only, no sampling). R>eps means target accuracy not certified at frozen N; advantage requires comparator and is not inferred from R>eps alone. Observed results only with method/case qualification (e.g., Pauli k=0 constant-copy; near-axis geometry discards to k_proxy=0 with small d+R). No novelty/practical scaling claim.

### Summary comparison across all 8 methods (certified vs rigorous vs numerical, fixed eps=0.05)

Certified = LP/geometry production certificates; rigorous = inverse exact-rational Hoeffding via production helpers (ideal probabilities); numerical = native/shadow evaluations of analytic Hoeffding formulas without a rigorous finite-precision certificate; naive has no fidelity interval. Bias/coverage/exceedance are diagnostic aggregates over 100 reps with Wilson/Hoeffding uncertainty caveat (NOT a new guarantee). All raw rows retained above.

| method | kind | ok rows | R<=eps successes | mean|bias| | max|bias| | mean cover | mean exceed |
|---|---|---|---|---|---|---|---|---|
| lp_common | production-certified | 170 | 33 | 0.0084 | 0.1517 | 0.9956 | 0.1735 |
| lp_half | production-certified | 170 | 33 | 0.0302 | 0.1780 | 0.9950 | 0.2372 |
| inv_common | rigorous-exact-rational | 170 | 33 | 0.0000 | 0.0000 | 0.9962 | 0.1426 |
| inv_half | rigorous-exact-rational | 170 | 33 | 0.0000 | 0.0000 | 0.9959 | 0.1918 |
| native | numerical-Hoeffding | 170 | 69 | 0.0000 | 0.0000 | 0.9819 | 0.0932 |
| shadow | numerical-Hoeffding | 170 | 30 | 0.0000 | 0.0000 | 0.9956 | 0.1714 |
| naive | no-fidelity-interval | 170 | 0 | 0.1032 | 0.7593 | - | 0.3586 |
| geometry | production-certified | 170 | 51 | 0.0043 | 0.1517 | 0.9928 | 0.1114 |

Totals: LP certified 66, geometry certified 51, inverse analytic 66, native analytic 69, shadow analytic 30.

Resource split: per-row law vs sampling/postprocessing times recorded (law_time_sec, sampling_time_sec); shared LP cache details show actual build/solve/certification times/bytes/conditioning/precision paid once with cached lookup distinguished (lp_cache_details/stats); geometry selection/preparation cost separately (geometry_prep); inverse preparation time/precision separately (inverse_prep); total runtime/Python peak with native-memory exclusion caveat (resources).

### Resource-only LP rows (k=8,12,16, N=4096, both channels, no simulation)

- k=8 common: status=ok b=2648232667967211410784573635029859882250707142404547581689341217509716687395181706780976690590014134230238330209074333911157665214271608880315753229474855982773780021051856463402936262600559657572195649/31388255274631663797670748519136872505146838599396545480129335378509451097073131017066558474213941579819434317564978561145035650946243492531704389612284094488637695127430311011935833371789195869511745536 G=1387/256 R=0.24462509823378795 suff=None cond=106.914093880225 solve_s=0.009185000002617016 reason=b=0.0843702 >= eps=0.05; formula needs eps>b
- k=8 degraded_half: status=ok b=1612387/16777216 G=1679/256 R=0.29009855319582617 suff=None cond=357.35277812191066 solve_s=0.011682299998938106 reason=b=0.0961058 >= eps=0.05; formula needs eps>b
- k=12 common: status=ok b=1135632815674430538884116825390953600857964602552916867669323013201234372403102827096083202604498244321117395198856264418053523595798385648865813036644111086015966606597718374612180720082080713121199904320768074872956926877886426406096283466863973364680771361821465934994966526482369347961175801635140205/11121945424056193189690467900341726996221944033937509489717073245037651034392327692849762492894379346468496886132370349646958626519015802102422031036865986899175372329727273709224924432009671058614769235385931975999511270408374352403145499687982696078872326615816823547116457988600199915556936362196205568 G=1871/256 R=0.3182839987522186 suff=None cond=1022.6529084037097 solve_s=0.01480190000438597 reason=b=0.102107 >= eps=0.05; formula needs eps>b
- k=12 degraded_half: status=ok b=647643407/4294967296 G=1619/256 R=0.3378515894339834 suff=None cond=6170.559933481238 solve_s=0.015322800005378667 reason=b=0.150791 >= eps=0.05; formula needs eps>b
- k=16 common: status=ok b=2128746436892363298201753392877111367186821691934341083707759095412610993929354082493695609158535204597648588759638969772545257364967132121066498068921401492652003786315302495903597128239595727304623813266144351912539887714812566086968017335677171614153562580449184834290752017887071166088779891225759104931812021637994171740026610056606172910793181356261304776139183787756938900152765474770597512323715/15394102643522538502088538771811548265584247645287609365841246441646425048336883618107282367300653644735889894164658049308619444077465202957827878664506090172811008182154628742642756537721977600209417643455538312323800897034775864946799491948989588370963660184360936371787894460739662760653643766701584075228757640716967057663560434553632546543656129142262917779218563178961403289883809363166641587748864 G=1759/256 R=0.3415192960998252 suff=None cond=9558.139014273587 solve_s=0.017761099996278062 reason=b=0.138283 >= eps=0.05; formula needs eps>b
- k=16 degraded_half: status=ok b=47322137049/274877906944 G=1767/256 R=0.37631731019894404 suff=None cond=103610.65832918513 solve_s=0.018835000002582092 reason=b=0.172157 >= eps=0.05; formula needs eps>b

### Homogeneous rational prototype (n=1..4, demo_proposal T=1/4 grid, den=4096, beta=1/40, plan_copies eps=1/20, count-only)

- n=1: status=ok mass=23791/4096 b=83025/6619136 V=1698035043/16777216 ceil=1 m=288553 N=865659 reason=
- n=2: status=ok mass=221411/4096 b=82935/6619136 V=147068492763/16777216 ceil=1 m=24973686 N=74921058 reason=
- n=3: status=error mass=None b=None V=None ceil=None m=None N=None reason=RuntimeError: HiGHS LP failed: success=False status=2 message=The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)
- n=4: status=error mass=None b=None V=None ceil=None m=None N=None reason=RuntimeError: HiGHS LP failed: success=False status=2 message=The problem is infeasible. (HiGHS Status 8: model_status is Infeasible; primal_status is None)

These count-only rows are separate from primary fixed-N comparisons; astronomical N retained as huge-int row with no sampling attempted.

### Operational smoke

- (a) lp_common n=2 hom_232 corr07 N=16 via run_streaming: {"N": 16, "estimate": 0.61474609375, "interval": [0.0, 1.0], "n_flag1": 16, "status": "ok"}
- (b) geometry n=2 mixed_core_tail corr07 N=16 via run_geometry_streaming: {"D": [1], "N": 16, "estimate": 0.8544921875, "interval": [0.205793354098762, 1.0], "status": "ok"}
  Backend: correlated full-copy backend: single latent all-target/all-opposite draw per copy (0.7/0.3) plus per-qubit Born via born_plus_prob with requested axes; never independent marginal simulation; design/backend RNG streams separate deterministic

### Coverage note

Reported coverage is diagnostic coverage across 100 fresh trials with Wilson 95% intervals plus Hoeffding half-width sqrt(ln(2/0.05)/(2*100))~=0.136 reference; this is NOT a new guarantee. LP/geometry confidence is production-certified under ideal probabilities; inverse is rigorous exact-rational score confidence under ideal probabilities (distinct from the floating Monte Carlo law); native/shadow are numerical evaluations without a rigorous finite-precision certificate. Naive fidelity confidence unavailable (worst-case bias d). Never infer confidence from observed variance.

### Limits

- Certificates assume ideal axis/degradation/Born probabilities and fresh iid copies; finite RNG/control rounding and apparatus errors outside certificates; no hardware guarantee.
- Python peak traced allocations exclude native HiGHS memory; dense-law size/cost, coefficient precision/rounding/conditioning, N and nN recorded separately; avoid expensive per-trial revalidation via frozen laws (operational wrappers tested separately via smoke).
- Existing production implementations untouched (hashes recorded alongside evaluator/verifier hashes); results distinguish production certificates, rigorous exact-rational inverse confidence, numerical-only native/shadow diagnostics, finite scope, sample cost vs total resources. No blanket claim that all method rows are production-certified.
- Native/shadow target-normalized bounds/counts are plainly labeled numerical evaluations of analytic Hoeffding formulas without a rigorous finite-precision certificate (ordinary ceil, no nextafter/padding proof). Soft time ceilings (HiGHS 120s option, per-row 60s) are observational flags with no hard-timeout claim. Homogeneous n=3,4 infeasibility is honest fixed-grid resource outcome, retained, not hidden.
