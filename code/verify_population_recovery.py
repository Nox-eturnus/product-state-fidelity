"""Deterministic stdlib-only checks for the product-magic BSC reduction.

Target R=(X+Y+Z)/sqrt(3), tau_+=(I+R)/2, tau_-=(I-R)/2,
rho_target(n)=tau_+^{tensor n}, eta=(1-1/sqrt(3))/2.

What is checked (n=1..3, explicit complex matrices/tensor products):
  - For arbitrary (possibly entangled/coherent/mixed) sigma, the Pauli-axis-
    averaged full-bitstring distribution Q(s) equals BSC_eta applied to the
    R-diagonal distribution p_x=Tr[sigma E_x], E_x=tensor_j tau_{(-1)^{x_j}}.
  - p is an honest distribution (sums to 1, nonnegative up to tolerance) and
    p_{0^n} equals the target fidelity Tr[sigma rho_target].
  - Off-diagonal R-coherences vanish in the averaged POVM: Q(sigma) equals
    Q(sigma_diag) where sigma_diag=sum_x p_x E_x.
  - For diagonal-submodel sigma_p, every local X/Y/Z setting gives the same
    full-bitstring distribution (identical BSC channel per setting).
  - Normalization/positivity of all Born distributions; target and orthogonal
    edge states give BSC(0^n) and BSC(1^n) respectively.
  - Independent Born probabilities use explicit projectors (I+sA)/2 and
    explicit tensor products, not the BSC formula itself.
  - Inverse-channel unbiased score g(s)=prod_j (1+sqrt(3)*s_j)/2:
    classical BSC expectation equals 1{x=0^n} for every latent x, and
    target second moment equals (3/2)^n; the same first/second moments
    derived from measured averaged Born distributions Q(s) of the explicit
    R-basis projectors E_x agree (explanatory unbiased-variance comparison
    only; no high-probability sample necessity inferred).

These checks support the reduction algebra only. They do not prove any
asymptotic sample-complexity theorem and are not novelty evidence.

Run: python code/verify_population_recovery.py
Writes population_recovery_checks.json alongside this script.
"""

from __future__ import annotations

import itertools
import json
import math
import platform
from pathlib import Path

SEED = 20260906
TOL = 1e-9
NS = (1, 2, 3)
SQRT3 = math.sqrt(3.0)
ETA = (1.0 - 1.0 / SQRT3) / 2.0

# 2x2 matrices as ((a,b),(c,d)) nested tuples of complex; converted to lists.
I2 = ((1 + 0j, 0j), (0j, 1 + 0j))
X2 = ((0j, 1 + 0j), (1 + 0j, 0j))
Y2 = ((0j, -1j), (1j, 0j))
Z2 = ((1 + 0j, 0j), (0j, -1 + 0j))


def mat(m):
    return [[complex(v) for v in row] for row in m]


I_M = mat(I2)
X_M = mat(X2)
Y_M = mat(Y2)
Z_M = mat(Z2)
R_M = [[(X_M[i][j] + Y_M[i][j] + Z_M[i][j]) / SQRT3 for j in range(2)] for i in range(2)]
TAU_P = [[(I_M[i][j] + R_M[i][j]) / 2.0 for j in range(2)] for i in range(2)]
TAU_M = [[(I_M[i][j] - R_M[i][j]) / 2.0 for j in range(2)] for i in range(2)]

AXIS = {"X": X_M, "Y": Y_M, "Z": Z_M}


def kron(a, b):
    n, m = len(a), len(a[0])
    p, q = len(b), len(b[0])
    out = [[0j] * (m * q) for _ in range(n * p)]
    for i in range(n):
        for j in range(m):
            aij = a[i][j]
            if aij == 0:
                continue
            for k in range(p):
                for ell in range(q):
                    out[i * p + k][j * q + ell] = aij * b[k][ell]
    return out


def eye(dim):
    return [[1 + 0j if i == j else 0j for j in range(dim)] for i in range(dim)]


def zeros(dim):
    return [[0j] * dim for _ in range(dim)]


def madd(a, b, sa=1.0, sb=1.0):
    n = len(a)
    return [[sa * a[i][j] + sb * b[i][j] for j in range(n)] for i in range(n)]


def mscale(a, s):
    return [[v * s for v in row] for row in a]


def mmul(a, b):
    n = len(a)
    out = [[0j] * n for _ in range(n)]
    for i in range(n):
        for k in range(n):
            aik = a[i][k]
            if aik == 0:
                continue
            for j in range(n):
                out[i][j] += aik * b[k][j]
    return out


def trace(a):
    return sum(a[i][i] for i in range(len(a)))


def tensor_list(mats):
    out = [[1 + 0j]]
    for m in mats:
        out = kron(out, m)
    return out


def ket_to_dm(vec):
    n = len(vec)
    return [[vec[i] * vec[j].conjugate() for j in range(n)] for i in range(n)]


def proj_tensor(setting, outcomes):
    """Tensor projector for local axes setting (list of 'X'/'Y'/'Z') and
    outcomes s_j in (+1,-1). P = tensor_j (I + s_j A_j)/2, explicit."""
    mats = []
    for ax, s in zip(setting, outcomes):
        a = AXIS[ax]
        mats.append([[ (I_M[i][j] + s * a[i][j]) / 2.0 for j in range(2)] for i in range(2)])
    return tensor_list(mats)


def r_projector_tensor(xbits):
    """E_x = tensor_j tau_{(-1)^{x_j}}; x_j=0 -> tau_+, 1 -> tau_-."""
    return tensor_list([TAU_P if b == 0 else TAU_M for b in xbits])


def target_projector(n):
    return tensor_list([TAU_P] * n)


def bsc_weight(s_tuple, x_tuple):
    w = 1.0
    for s, x in zip(s_tuple, x_tuple):
        same = (s == (1 if x == 0 else -1))
        w *= (1.0 - ETA) if same else ETA
    return w


def g_score(s_tuple):
    """Inverse-channel unbiased score prod_j (1+sqrt(3)*s_j)/2."""
    v = 1.0
    for s in s_tuple:
        v *= (1.0 + SQRT3 * s) / 2.0
    return v


def all_settings(n):
    return list(itertools.product("XYZ", repeat=n))


def all_outcomes(n):
    return list(itertools.product((1, -1), repeat=n))


def all_x(n):
    return list(itertools.product((0, 1), repeat=n))


def lcg(seed):
    state = seed & 0x7FFFFFFF
    while True:
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        yield state / 0x7FFFFFFF


def make_states(n):
    """Deterministic family of test density matrices (explicit)."""
    dim = 2 ** n
    states = {}
    # |0...0><0...0| (computational product)
    vec0 = [0j] * dim
    vec0[0] = 1 + 0j
    states["comp_zero"] = ket_to_dm(vec0)
    # |1...1>
    vec1 = [0j] * dim
    vec1[dim - 1] = 1 + 0j
    states["comp_one"] = ket_to_dm(vec1)
    # |+>^n with |+>=(|0>+|1>)/sqrt2 via explicit Kronecker products.
    def kron_vec(a, b):
        return [x * y for x in a for y in b]
    vv = [1 + 0j]
    for _ in range(n):
        vv = kron_vec(vv, [1 / math.sqrt(2) + 0j, 1 / math.sqrt(2) + 0j])
    states["plus_product"] = ket_to_dm(vv)
    # Target product tau_+^n as density (pure product)
    # tau_+ eigenvector: R|+R> = +|+R>. Build single-qubit +R vector explicitly:
    # R = (X+Y+Z)/sqrt3; solve (R-I)v=0 deterministically via fixed formula.
    # Use normalized null vector of (R-I): second row gives relation.
    # (R-I) = [[(1/sqrt3)-1, (1-i)/sqrt3],[(1+i)/sqrt3, (-1/sqrt3)-1]]
    a11 = (1 / SQRT3) - 1
    a12 = (1 - 1j) / SQRT3
    # null vector: [-a12, a11] normalized (since row1 . v = 0)
    vr = [-a12, a11]
    norm = math.sqrt(abs(vr[0]) ** 2 + abs(vr[1]) ** 2)
    vr = [c / norm for c in vr]
    vv_t = [1 + 0j]
    for _ in range(n):
        vv_t = kron_vec(vv_t, vr)
    states["target_pure"] = ket_to_dm(vv_t)
    # Orthogonal product tau_-^n: eigenvector of R with -1; null of (R+I)
    b11 = (1 / SQRT3) + 1
    b12 = (1 - 1j) / SQRT3
    vo = [-b12, b11]
    norm2 = math.sqrt(abs(vo[0]) ** 2 + abs(vo[1]) ** 2)
    vo = [c / norm2 for c in vo]
    vv_o = [1 + 0j]
    for _ in range(n):
        vv_o = kron_vec(vv_o, vo)
    states["orthogonal_pure"] = ket_to_dm(vv_o)
    # Entangled: Bell/GHZ (|0^n>+|1^n>)/sqrt2
    vg = [0j] * dim
    vg[0] = 1 / math.sqrt(2)
    vg[dim - 1] = 1 / math.sqrt(2)
    states["ghz"] = ket_to_dm(vg)
    # W for n=3; for n<3 use (|01>+|10>)/sqrt2 analogue embedded
    if n == 3:
        vw = [0j] * 8
        for idx in (1, 2, 4):
            vw[idx] = 1 / math.sqrt(3)
        states["w"] = ket_to_dm(vw)
    elif n == 2:
        vw = [0j] * 4
        vw[1] = 1 / math.sqrt(2)
        vw[2] = 1 / math.sqrt(2)
        states["w01_10"] = ket_to_dm(vw)
    else:
        vw = [1 / math.sqrt(2) + 0j, 1j / math.sqrt(2)]
        states["phase_plus"] = ket_to_dm(vw)
    # Maximally mixed
    states["maximally_mixed"] = mscale(eye(dim), 1.0 / dim)
    # Mixture 0.6 target + 0.4 orthogonal (diagonal in R basis, separable mixture)
    states["target_orth_mix"] = madd(states["target_pure"], states["orthogonal_pure"], 0.6, 0.4)
    # Mixture 0.5 GHZ + 0.5 maximally mixed (entangled + noise)
    states["ghz_mixed"] = madd(states["ghz"], states["maximally_mixed"], 0.5, 0.5)
    # Coherent R-superposition: (|+R>^n + |-R>^n)/sqrt2 (off-diagonal in R basis)
    vs = [(vv_t[i] + vv_o[i]) / math.sqrt(2) for i in range(dim)]
    nrm = math.sqrt(sum(abs(c) ** 2 for c in vs))
    vs = [c / nrm for c in vs]
    states["r_cat"] = ket_to_dm(vs)
    return states


def born_dist(sigma, setting):
    n = len(setting)
    out = {}
    for s in all_outcomes(n):
        proj = proj_tensor(setting, s)
        v = trace(mmul(sigma, proj))
        out[s] = v
    return out


def check_n(n):
    settings = all_settings(n)
    outcomes = all_outcomes(n)
    xs = all_x(n)
    states = make_states(n)
    target = target_projector(n)
    res = {
        "n": n,
        "dim": 2 ** n,
        "num_settings": len(settings),
        "num_states": len(states),
        "max_avg_vs_bsc": 0.0,
        "max_diag_setting_spread": 0.0,
        "max_normalization_err": 0.0,
        "max_positivity_violation": 0.0,
        "max_imag_residual": 0.0,
        "max_p_sum_err": 0.0,
        "max_p_negativity": 0.0,
        "max_fidelity_match_err": 0.0,
        "max_offdiag_vanish_err": 0.0,
        "max_target_bsc_err": 0.0,
        "max_orth_bsc_err": 0.0,
        "max_unbiased_score_err": 0.0,
        "max_target_second_moment_err": 0.0,
        "max_measured_first_moment_err": 0.0,
        "max_measured_second_moment_err": 0.0,
        "state_details": {},
    }
    # Precompute R projectors once
    ex_list = [r_projector_tensor(x) for x in xs]
    for name, sigma in states.items():
        # R-diagonal distribution (complex then real)
        p_complex = [trace(mmul(sigma, e)) for e in ex_list]
        imag = max(abs(v.imag) for v in p_complex)
        p = [min(max(v.real, 0.0), 1.0) if abs(v.imag) < 1e-7 else v.real for v in p_complex]
        # Honesty before clipping: check raw real parts
        raw_real = [v.real for v in p_complex]
        p_sum_err = abs(sum(raw_real) - 1.0)
        p_neg = max(0.0, max(-v for v in raw_real))
        fid = trace(mmul(sigma, target)).real
        fid_match = abs(raw_real[xs.index(tuple([0] * n))] - fid)
        res["max_imag_residual"] = max(res["max_imag_residual"], imag)
        res["max_p_sum_err"] = max(res["max_p_sum_err"], p_sum_err)
        res["max_p_negativity"] = max(res["max_p_negativity"], p_neg)
        res["max_fidelity_match_err"] = max(res["max_fidelity_match_err"], fid_match)
        # Born per setting + averaged Q
        q_avg = {s: 0j for s in outcomes}
        worst_norm = 0.0
        worst_pos = 0.0
        for st in settings:
            bd = born_dist(sigma, st)
            ssum = sum(v.real for v in bd.values())
            worst_norm = max(worst_norm, abs(ssum - 1.0))
            for v in bd.values():
                if v.real < -worst_pos:
                    worst_pos = -v.real
                if v.real > 1.0 + worst_pos:
                    worst_pos = v.real - 1.0
                res["max_imag_residual"] = max(res["max_imag_residual"], abs(v.imag))
            for s in outcomes:
                q_avg[s] += bd[s] / len(settings)
        res["max_normalization_err"] = max(res["max_normalization_err"], worst_norm)
        # Positivity of averaged too
        for v in q_avg.values():
            if v.real < 0:
                res["max_positivity_violation"] = max(res["max_positivity_violation"], -v.real)
            if v.real > 1:
                res["max_positivity_violation"] = max(res["max_positivity_violation"], v.real - 1.0)
        res["max_positivity_violation"] = max(res["max_positivity_violation"], worst_pos)
        # BSC prediction from raw p (use raw_real, not clipped, for exactness)
        maxdiff = 0.0
        for s in outcomes:
            pred = sum(rv * bsc_weight(s, x) for rv, x in zip(raw_real, xs))
            diff = abs(q_avg[s].real - pred)
            maxdiff = max(maxdiff, diff)
            res["max_imag_residual"] = max(res["max_imag_residual"], abs(q_avg[s].imag))
        res["max_avg_vs_bsc"] = max(res["max_avg_vs_bsc"], maxdiff)
        # Off-diagonal vanishing: sigma_diag from raw p
        sigma_diag = zeros(2 ** n)
        for rv, e in zip(raw_real, ex_list):
            sigma_diag = madd(sigma_diag, e, 1.0, rv)
        q_diag_avg = {s: 0j for s in outcomes}
        for st in settings:
            bd = born_dist(sigma_diag, st)
            for s in outcomes:
                q_diag_avg[s] += bd[s] / len(settings)
        offdiff = max(abs(q_avg[s].real - q_diag_avg[s].real) for s in outcomes)
        res["max_offdiag_vanish_err"] = max(res["max_offdiag_vanish_err"], offdiff)
        res["state_details"][name] = {
            "avg_vs_bsc": maxdiff,
            "p_sum_err": p_sum_err,
            "offdiag_vanish": offdiff,
        }
    # Diagonal-submodel identical-across-settings check with deterministic p
    gen = lcg(SEED + n)
    raw = [next(gen) + 0.1 for _ in xs]
    tot = sum(raw)
    p_diag = [v / tot for v in raw]
    sigma_p = zeros(2 ** n)
    for w, e in zip(p_diag, ex_list):
        sigma_p = madd(sigma_p, e, 1.0, w)
    dists = [born_dist(sigma_p, st) for st in settings]
    spread = 0.0
    for s in outcomes:
        vals = [d[s].real for d in dists]
        spread = max(spread, max(vals) - min(vals))
    res["max_diag_setting_spread"] = spread
    # Edge states: target must give BSC(0^n), orthogonal BSC(1^n)
    for label, sigma_edge, center in (
        ("target", states["target_pure"], tuple([1] * n)),
        ("orthogonal", states["orthogonal_pure"], tuple([-1] * n)),
    ):
        qe = {s: 0j for s in outcomes}
        for st in settings:
            bd = born_dist(sigma_edge, st)
            for s in outcomes:
                qe[s] += bd[s] / len(settings)
        for s in outcomes:
            # center is in s-space (+1/-1); convert to x-bits: +1->0, -1->1
            xc = tuple(0 if v == 1 else 1 for v in center)
            pred = bsc_weight(s, xc)
            err = abs(qe[s].real - pred)
            if label == "target":
                res["max_target_bsc_err"] = max(res["max_target_bsc_err"], err)
            else:
                res["max_orth_bsc_err"] = max(res["max_orth_bsc_err"], err)
    # Inverse-channel unbiased score: classical BSC expectation and target
    # second moment, plus the same moments derived from measured averaged
    # Born distributions of the explicit R-basis projectors E_x.
    zero_x = tuple([0] * n)
    target_second = (1.5) ** n
    for x in xs:
        want_first = 1.0 if x == zero_x else 0.0
        classical_first = sum(bsc_weight(s, x) * g_score(s) for s in outcomes)
        res["max_unbiased_score_err"] = max(
            res["max_unbiased_score_err"], abs(classical_first - want_first)
        )
        # Measured first moment from explicit averaged Born distribution of E_x.
        sigma_x = r_projector_tensor(x)
        qx = {s: 0j for s in outcomes}
        for st in settings:
            bd = born_dist(sigma_x, st)
            for s in outcomes:
                qx[s] += bd[s] / len(settings)
        measured_first = sum(qx[s].real * g_score(s) for s in outcomes)
        res["max_measured_first_moment_err"] = max(
            res["max_measured_first_moment_err"], abs(measured_first - want_first)
        )
    classical_second = sum(
        bsc_weight(s, zero_x) * g_score(s) ** 2 for s in outcomes
    )
    res["max_target_second_moment_err"] = abs(classical_second - target_second)
    # Measured target second moment from explicit target Born distribution.
    q_target = {s: 0j for s in outcomes}
    for st in settings:
        bd = born_dist(states["target_pure"], st)
        for s in outcomes:
            q_target[s] += bd[s] / len(settings)
    measured_second = sum(q_target[s].real * g_score(s) ** 2 for s in outcomes)
    res["max_measured_second_moment_err"] = abs(measured_second - target_second)
    # Full diagonal-model distribution preview (all 2^n entries with x-bit labels).
    res["diag_model_p"] = p_diag
    res["diag_model_labels"] = [''.join(str(b) for b in x) for x in xs]
    return res


def main():
    per_n = [check_n(n) for n in NS]
    overall_max = max(
        max(
            r["max_avg_vs_bsc"],
            r["max_diag_setting_spread"],
            r["max_normalization_err"],
            r["max_positivity_violation"],
            r["max_imag_residual"],
            r["max_p_sum_err"],
            r["max_p_negativity"],
            r["max_fidelity_match_err"],
            r["max_offdiag_vanish_err"],
            r["max_target_bsc_err"],
            r["max_orth_bsc_err"],
            r["max_unbiased_score_err"],
            r["max_target_second_moment_err"],
            r["max_measured_first_moment_err"],
            r["max_measured_second_moment_err"],
        )
        for r in per_n
    )
    status = "PASS" if overall_max <= TOL else "FAIL"
    payload = {
        "status": status,
        "scope": "Reduction-algebra support only (n=1..3 explicit matrices); not an asymptotic-theorem or novelty check.",
        "seed": SEED,
        "tolerance": TOL,
        "python": platform.python_version(),
        "eta": ETA,
        "eta_formula": "(1-1/sqrt(3))/2",
        "mu_fixed_eta": 0.5,
        "n_values": list(NS),
        "overall_max_error": overall_max,
        "per_n": per_n,
        "notes": [
            "Born probabilities use explicit (I+sA)/2 projectors and kron tensor products.",
            "Averaged Q(s) uses uniform average over all 3^n X/Y/Z settings, axis labels discarded.",
            "BSC prediction uses p_x=Tr[sigma E_x] with E_x=tensor tau_{(-1)^x}; p_0^n is target fidelity.",
            "Diagonal-submodel spread checks identical BSC channel across all settings.",
            "Target/orthogonal edge checks compare to BSC(0^n)/BSC(1^n).",
            "Unbiased-score checks use g(s)=prod (1+sqrt(3)*s_j)/2; classical BSC and measured Born moments agree on 1{x=0} and (3/2)^n (variance comparison only).",
            "No sampling randomness; deterministic states plus LCG p for diagonal model.",
        ],
    }
    dest = (Path(__file__).resolve().parents[1] / "results" / "population_recovery_checks.json")
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
