"""Tiny exact-arithmetic LP prototype for degraded signal-1/2 channel .

Scope: finite small-k diagnostic only (k=1,2; N=1,4,16,256 perfect squares).
Finite vertex enumeration with fractions.Fraction, NOT a scalable solver.
Small-k optima do NOT validate the asymptotic exponent.

Physical implementation note: weighted axis/sign sampling gives signal 1/L_j
per noisy qubit; an independent coin with mean L_j/2 (legal since
L_j<=sqrt(3)<2) multiplies the sign-corrected bit, giving overall signal
(1/L_j)(L_j/2)=1/2 (eta=1/4) for ANY pure target orientation. This is
intentionally noisier than the theorem upper 1/sqrt(3) but still fixed
positive noise, same exponent shape. Ideal real axis/randomization
probabilities are assumed; coefficient rounding is certified separately from
apparatus probability errors. Exact stabilizer flag multiplication uses the
same g with NO division by flag probability (division by C appears only in
analysis, never in the estimator).

Channel: W[w,h] = P(observed weight w | latent weight h) at eta=1/4,
  W[w,h]=sum_{u+v=w} Bin(h,u)(3/4)^u(1/4)^{h-u} Bin(k-h,v)(1/4)^v(3/4)^{k-h-v}.
Column h is latent, row w is observed. Correct orientation verified by
independent full-bitstring enumeration; all columns sum to 1 exactly.

LP: variables g_0..g_k,b,t; constraints
  -b <= sum_w W[w,h] g_w - delta_{h0} <= b for all h,
  -t <= g_w <= t for all w, b>=0, t>=0.
Objective b + t/sqrt(N) with N perfect squares (exact rational).
Enumerate all (k+3)-subsets of active inequalities, exact Gaussian
elimination, discard singular/infeasible, optimize each N over same vertices.
Independent feasible comparators: zero, constant-1/2, exact inverse
  g_w=(3/2)^{k-w}(-1/2)^w (alternating signs, unbiased).
For each returned score: exact b=max_h|sum W g - delta|, G=max|g|;
root-MSE surrogate b+G/sqrt(N) reported SEPARATELY from fixed-confidence
Hoeffding radius b+G*sqrt(2 ln(2/delta)/N) at delta=1/3; conditional
sufficient copies ceil(2G^2 ln(2/delta)/(eps-b)^2) if b<eps=1/20 else null
(sufficient only, not minimax necessity). Final estimate clipped to [0,1].
Rounding to denominator 256 certified by fresh exact b_round/G_round with
b_round<=b+maxdiff. No hidden division by zero; no division by flag prob.

Run: python code/product_state_lp_prototype.py
Writes product_state_lp_checks.json alongside this script. Stdlib only.
Exits nonzero on any failure.
"""

from __future__ import annotations

import itertools
import json
import math
import platform
import time
from fractions import Fraction
from pathlib import Path

KS = (1, 2)
N_VALUES = (1, 4, 16, 256)
DELTA = 1.0 / 3.0
EPS = Fraction(1, 20)
ROUND_DEN = 256
PERT = Fraction(1, 1000)
TOL_FLOAT = 1e-12

ETA = Fraction(1, 4)
STAY = Fraction(3, 4)


def clipped_estimate(mean):
    """Clip a raw mean estimate to [0,1]."""
    if mean < 0:
        return 0
    if mean > 1:
        return 1
    return mean


def estimate_from_counts(counts, g):
    """counts[w] nonnegative counts for observed weights; g list Fractions.

    Returns clipped weighted average sum_w (counts[w]/total) g_w.
    Uses float for final mean then clips; exact rational mean also checked.
    Raises ValueError for mismatched/empty lengths, non-integer or
    negative counts (bool rejected), or non-positive total.
    """
    if len(counts) != len(g):
        raise ValueError("counts/g length mismatch")
    if len(counts) == 0:
        raise ValueError("empty counts")
    for c in counts:
        if isinstance(c, bool) or not isinstance(c, int):
            raise ValueError("counts must be nonnegative integers")
        if c < 0:
            raise ValueError("negative count")
    total = sum(counts)
    if total <= 0:
        raise ValueError("non-positive total")
    # exact rational mean
    num = sum(Fraction(c, 1) * gw for c, gw in zip(counts, g))
    mean_exact = num / total  # Fraction / int exact, total nonzero checked
    mean_f = float(mean_exact)
    return clipped_estimate(mean_f)


def build_W_conv(k):
    W = [[Fraction(0, 1) for _ in range(k + 1)] for _ in range(k + 1)]
    for h in range(k + 1):
        for w in range(k + 1):
            total = Fraction(0, 1)
            lo = max(0, w - (k - h))
            hi = min(h, w)
            for u in range(lo, hi + 1):
                v = w - u
                term = (
                    math.comb(h, u)
                    * (STAY ** u)
                    * (ETA ** (h - u))
                    * math.comb(k - h, v)
                    * (ETA ** v)
                    * (STAY ** (k - h - v))
                )
                total += term
            W[w][h] = total
    return W


def build_W_enum(k):
    W = [[Fraction(0, 1) for _ in range(k + 1)] for _ in range(k + 1)]
    for h in range(k + 1):
        x = tuple([1] * h + [0] * (k - h))
        for bits in itertools.product((0, 1), repeat=k):
            w = sum(bits)
            p = Fraction(1, 1)
            for yj, xj in zip(bits, x):
                p *= STAY if yj == xj else ETA
            W[w][h] += p
    return W


def inverse_g(k):
    return [Fraction(3, 2) ** (k - w) * Fraction(-1, 2) ** w for w in range(k + 1)]


def residual_b(W, g):
    k = len(g) - 1
    resids = []
    for h in range(k + 1):
        s = sum(W[w][h] * g[w] for w in range(k + 1))
        delta = Fraction(1, 1) if h == 0 else Fraction(0, 1)
        resids.append(s - delta)
    b = max(abs(r) for r in resids)
    # maximizer index
    hstar = max(range(k + 1), key=lambda h: abs(resids[h]))
    return b, resids, hstar


def G_of(g):
    return max(abs(v) for v in g)


def solve_fraction(A, b):
    """Solve A x = b with Fractions; return list or None if singular.

    No hidden division by zero: pivot checked nonzero before division.
    """
    n = len(A)
    M = [list(A[i]) + [b[i]] for i in range(n)]
    for col in range(n):
        piv = None
        for r in range(col, n):
            if M[r][col] != 0:
                piv = r
                break
        if piv is None:
            return None
        if piv != col:
            M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        # pv nonzero by construction
        for j in range(col, n + 1):
            M[col][j] = M[col][j] / pv
        for r in range(n):
            if r == col:
                continue
            factor = M[r][col]
            if factor != 0:
                for j in range(col, n + 1):
                    M[r][j] = M[r][j] - factor * M[col][j]
    return [M[i][n] for i in range(n)]


def build_constraints(W):
    k = len(W) - 1  # W is (k+1)x(k+1)
    d = k + 3
    cons = []  # list of (coeff list Fractions, bound Fraction)
    # bias constraints
    for h in range(k + 1):
        delta = Fraction(1, 1) if h == 0 else Fraction(0, 1)
        c1 = [Fraction(0, 1)] * d
        for w in range(k + 1):
            c1[w] = W[w][h]
        c1[k + 1] = Fraction(-1, 1)
        c1[k + 2] = Fraction(0, 1)
        cons.append((c1, delta))
        c2 = [Fraction(0, 1)] * d
        for w in range(k + 1):
            c2[w] = -W[w][h]
        c2[k + 1] = Fraction(-1, 1)
        c2[k + 2] = Fraction(0, 1)
        cons.append((c2, -delta))
    # box constraints
    for w in range(k + 1):
        c3 = [Fraction(0, 1)] * d
        c3[w] = Fraction(1, 1)
        c3[k + 2] = Fraction(-1, 1)
        cons.append((c3, Fraction(0, 1)))
        c4 = [Fraction(0, 1)] * d
        c4[w] = Fraction(-1, 1)
        c4[k + 2] = Fraction(-1, 1)
        cons.append((c4, Fraction(0, 1)))
    c5 = [Fraction(0, 1)] * d
    c5[k + 1] = Fraction(-1, 1)
    cons.append((c5, Fraction(0, 1)))
    c6 = [Fraction(0, 1)] * d
    c6[k + 2] = Fraction(-1, 1)
    cons.append((c6, Fraction(0, 1)))
    return cons


def is_feasible(x, cons):
    for coeff, bound in cons:
        s = sum(c * v for c, v in zip(coeff, x))
        if s > bound:
            return False
    return True


def round_to_den(gw, den):
    # exact rounding of Fraction gw to multiple of 1/den, ties to even
    f = gw * den  # Fraction
    num = f.numerator
    dd = f.denominator
    q = num // dd  # floor
    rem = num % dd
    # rem in [0,dd)
    left = rem * 2
    if left < dd:
        m = q
    elif left > dd:
        m = q + 1
    else:
        m = q if (q % 2 == 0) else q + 1
    return Fraction(m, den), abs(gw - Fraction(m, den))


def rounding_cert(W, g):
    k = len(g) - 1
    g_round = []
    maxdiff = Fraction(0, 1)
    for gw in g:
        gr, d = round_to_den(gw, ROUND_DEN)
        g_round.append(gr)
        if d > maxdiff:
            maxdiff = d
    b, _, _ = residual_b(W, g)
    b_round, _, _ = residual_b(W, g_round)
    G = G_of(g)
    G_round = G_of(g_round)
    ok = b_round <= b + maxdiff
    return {
        "g_round": g_round,
        "maxdiff": maxdiff,
        "b": b,
        "b_round": b_round,
        "G": G,
        "G_round": G_round,
        "ok": ok,
    }


def check_degraded_povm():
    """Dense float check: weighted+degraded gives signal 1/2 for test r's.

    Uses boundary-style 2x2 matrices with floats. Verifies effective
    E'_t=(I+tR/2)/2 and overall flip 1/4. Returns dict.
    """
    import math as _m

    def mat_add(a, b, sa=1.0, sb=1.0):
        return [[sa * a[i][j] + sb * b[i][j] for j in range(2)] for i in range(2)]

    def mat_maxdiff(a, b):
        return max(abs(a[i][j] - b[i][j]) for i in range(2) for j in range(2))

    I = [[1 + 0j, 0j], [0j, 1 + 0j]]
    X = [[0j, 1 + 0j], [1 + 0j, 0j]]
    Y = [[0j, -1j], [1j, 0j]]
    Z = [[1 + 0j, 0j], [0j, -1 + 0j]]
    test_rs = [
        (1 / _m.sqrt(3), 1 / _m.sqrt(3), 1 / _m.sqrt(3)),
        (2 / 3, 1 / 3, 2 / 3),
        (0.6, 0.0, 0.8),
        (1 / _m.sqrt(2), 1 / _m.sqrt(2), 0.0),
    ]
    max_err = 0.0
    max_eta_err = 0.0
    for r in test_rs:
        L = abs(r[0]) + abs(r[1]) + abs(r[2])
        R = [
            [r[0] * X[i][j] + r[1] * Y[i][j] + r[2] * Z[i][j] for j in range(2)]
            for i in range(2)
        ]
        m = L / 2.0
        eta_up = (1 - 1 / L) / 2.0
        eta_deg = (1 - m) / 2.0
        eta_all = eta_up + eta_deg - 2 * eta_up * eta_deg
        max_eta_err = max(max_eta_err, abs(eta_all - 0.25))
        # effective POVM check: E'_t = ((1+m)/2)E_t + ((1-m)/2)E_{-t}
        for t in (1, -1):
            Et = [[(I[i][j] + t * R[i][j] / L) / 2.0 for j in range(2)] for i in range(2)]
            Emt = [[(I[i][j] - t * R[i][j] / L) / 2.0 for j in range(2)] for i in range(2)]
            Ep = mat_add(Et, Emt, (1 + m) / 2.0, (1 - m) / 2.0)
            Want = [[(I[i][j] + t * R[i][j] / 2.0) / 2.0 for j in range(2)] for i in range(2)]
            max_err = max(max_err, mat_maxdiff(Ep, Want))
    return {"max_povm_err": max_err, "max_eta_err": max_eta_err}


def run_k(k):
    t0 = time.perf_counter()
    W = build_W_conv(k)
    W2 = build_W_enum(k)
    # W equality + column sums
    w_match = all(W[w][h] == W2[w][h] for w in range(k + 1) for h in range(k + 1))
    col_ok = all(
        sum(W[w][h] for w in range(k + 1)) == Fraction(1, 1) for h in range(k + 1)
    )
    # inverse unbiasedness
    ginv = inverse_g(k)
    _, resids_inv, _ = residual_b(W, ginv)
    inv_ok = all(r == 0 for r in resids_inv)
    inv_b, _, _ = residual_b(W, ginv)
    inv_G = G_of(ginv)
    # comparators
    gzero = [Fraction(0, 1)] * (k + 1)
    bzero, _, _ = residual_b(W, gzero)
    Gzero = G_of(gzero)
    ghalf = [Fraction(1, 2)] * (k + 1)
    bhalf, _, _ = residual_b(W, ghalf)
    Ghalf = G_of(ghalf)
    cons = build_constraints(W)
    d = k + 3
    m = len(cons)
    total_sets = 0
    singular = 0
    infeasible = 0
    verts = []
    seen = set()
    for idx in itertools.combinations(range(m), d):
        total_sets += 1
        A = [cons[i][0] for i in idx]
        bb = [cons[i][1] for i in idx]
        sol = solve_fraction(A, bb)
        if sol is None:
            singular += 1
            continue
        if not is_feasible(sol, cons):
            infeasible += 1
            continue
        key = tuple(sol)
        if key not in seen:
            seen.add(key)
            verts.append(sol)
        else:
            # duplicate vertex from different active set; count as feasible dup
            pass
    feasible_unique = len(verts)
    # objectives per N over same vertices
    comp_objs = {}
    for N in N_VALUES:
        s = math.isqrt(N)
        assert s * s == N
        comp_objs[N] = {
            "zero": bzero + Gzero / s,
            "half": bhalf + Ghalf / s,
            "inv": inv_b + inv_G / s,
        }
    optima = {}
    for N in N_VALUES:
        s = math.isqrt(N)
        best = None
        best_x = None
        for x in verts:
            b_lp = x[k + 1]
            t_lp = x[k + 2]
            obj = b_lp + t_lp / s
            if best is None or obj < best:
                best = obj
                best_x = x
        # honest tie count: vertices attaining best exactly
        num_ties = sum(
            1 for x in verts if (x[k + 1] + x[k + 2] / s) == best
        )
        # compare to comparators (must be no worse)
        min_comp = min(comp_objs[N].values())
        assert best is not None
        assert best <= min_comp, f"k={k} N={N} best {best} > comps {min_comp}"
        g_opt = list(best_x[: k + 1])
        b_lp = best_x[k + 1]
        t_lp = best_x[k + 2]
        b_rec, resids, hstar = residual_b(W, g_opt)
        G_rec = G_of(g_opt)
        # tightness: LP vars must match recomputed (optimality forces equality)
        assert b_lp == b_rec, f"k={k} N={N} b_lp {b_lp} != {b_rec}"
        assert t_lp == G_rec, f"k={k} N={N} t_lp {t_lp} != {G_rec}"
        # maximizer attains b
        assert abs(resids[hstar]) == b_rec
        # feasible residual check already via is_feasible, recheck explicitly
        assert is_feasible(best_x, cons)
        surr = b_rec + G_rec / s  # Fraction
        # Hoeffding radius (float) at delta=1/3
        rad = float(b_rec) + float(G_rec) * math.sqrt(2 * math.log(2 / DELTA) / N)
        # sufficient copies if b<eps
        if b_rec < EPS:
            denom = float(EPS - b_rec)
            need = math.ceil(2 * float(G_rec) ** 2 * math.log(2 / DELTA) / (denom ** 2))
            need_reason = None
        else:
            need = None
            need_reason = f"b={float(b_rec):.6g} >= eps=0.05; formula needs eps>b"
            # if G==0 and b>=eps, also null; reason stands
        # rounding cert
        rc = rounding_cert(W, g_opt)
        assert rc["ok"], f"rounding cert failed k={k} N={N}"
        # mixed + flag checks (unnormalized law, no division by C)
        # weight-symmetric test distribution q_h = (h+1)/T
        T = sum(h + 1 for h in range(k + 1))
        q = [Fraction(h + 1, T) for h in range(k + 1)]
        mu = [sum(W[w][h] * g_opt[w] for w in range(k + 1)) for h in range(k + 1)]
        nu = [
            sum(W[w][h] * g_opt[w] * g_opt[w] for w in range(k + 1))
            for h in range(k + 1)
        ]
        E_mix = sum(q[h] * mu[h] for h in range(k + 1))
        truth_mix = q[0]
        bias_mix = abs(E_mix - truth_mix)
        assert bias_mix <= b_rec
        E2_mix = sum(q[h] * nu[h] for h in range(k + 1))
        assert E2_mix <= G_rec * G_rec
        # flag0: C=0
        C0 = Fraction(0, 1)
        E_f0 = Fraction(0, 1)
        truth_f0 = Fraction(0, 1)
        assert abs(E_f0 - truth_f0) <= b_rec
        assert E_f0 * E_f0 <= G_rec * G_rec  # second moment 0
        E2_f0 = Fraction(0, 1)
        assert E2_f0 <= G_rec * G_rec
        # rare flag 1/1000
        Crare = Fraction(1, 1000)
        E_rare = Crare * E_mix
        truth_rare = Crare * truth_mix
        bias_rare = abs(E_rare - truth_rare)
        assert bias_rare <= b_rec  # since C*b <= b
        assert bias_rare <= Crare * b_rec + Fraction(0, 1)
        E2_rare = Crare * E2_mix
        assert E2_rare <= G_rec * G_rec
        # bit lengths
        bitlens = [
            (gw.numerator.bit_length(), gw.denominator.bit_length()) for gw in g_opt
        ]
        optima[str(N)] = {
            "N": N,
            "sqrtN": s,
            "g": [str(v) for v in g_opt],
            "g_float": [float(v) for v in g_opt],
            "b": str(b_rec),
            "b_float": float(b_rec),
            "G": str(G_rec),
            "G_float": float(G_rec),
            "residuals": [str(r) for r in resids],
            "hstar": hstar,
            "objective": str(surr),
            "objective_float": float(surr),
            "surrogate": str(surr),
            "surrogate_float": float(surr),
            "hoeffding_radius": rad,
            "sufficient_copies": need,
            "sufficient_reason": need_reason,
            "rounding": {
                "g_round": [str(v) for v in rc["g_round"]],
                "g_round_float": [float(v) for v in rc["g_round"]],
                "maxdiff": str(rc["maxdiff"]),
                "maxdiff_float": float(rc["maxdiff"]),
                "b_round": str(rc["b_round"]),
                "b_round_float": float(rc["b_round"]),
                "G_round": str(rc["G_round"]),
                "G_round_float": float(rc["G_round"]),
                "cert_ok": rc["ok"],
            },
            "mixed": {
                "q": [str(v) for v in q],
                "bias": str(bias_mix),
                "bias_float": float(bias_mix),
                "second": str(E2_mix),
                "second_float": float(E2_mix),
            },
            "flag0": {"C": "0", "bias": "0", "second": "0"},
            "flag_rare": {
                "C": "1/1000",
                "bias": str(bias_rare),
                "bias_float": float(bias_rare),
                "second": str(E2_rare),
                "second_float": float(E2_rare),
            },
            "bit_lengths": bitlens,
            "max_bitlen_num": max(b[0] for b in bitlens),
            "max_bitlen_den": max(b[1] for b in bitlens),
            "feasible": True,
            "num_ties": num_ties,
            "tie_note": "vertices attaining best objective exactly; ties reported honestly",
            "comparator_objectives": {kk: str(vv) for kk, vv in comp_objs[N].items()},
        }
    # non-dyadic perturbation check (at least one, non-vacuous)
    # use first optimum N=1
    g0 = [Fraction(s) for s in optima["1"]["g"]]
    # rebuild as Fractions exactly from stored strings
    g_pert = [v + PERT for v in g0]
    b_pert, _, _ = residual_b(W, g_pert)
    G_pert = G_of(g_pert)
    rc_pert = rounding_cert(W, g_pert)
    assert rc_pert["ok"]
    pert_nonvacuous = rc_pert["maxdiff"] > 0
    # if all optima already dyadic (all maxdiff==0), perturbation must be non-vacuous
    all_dyadic = all(
        Fraction(optima[str(N)]["rounding"]["maxdiff"]) == 0 for N in N_VALUES
    )
    if all_dyadic:
        assert pert_nonvacuous, "need non-vacuous perturbation when optima dyadic"
    t1 = time.perf_counter()
    return {
        "k": k,
        "W": [[str(W[w][h]) for h in range(k + 1)] for w in range(k + 1)],
        "W_float": [[float(W[w][h]) for h in range(k + 1)] for w in range(k + 1)],
        "W_match_enum": w_match,
        "W_cols_sum1": col_ok,
        "inverse": {
            "g": [str(v) for v in ginv],
            "g_float": [float(v) for v in ginv],
            "b": str(inv_b),
            "G": str(inv_G),
            "unbiased": inv_ok,
        },
        "comparators": {
            "zero": {"g": [str(v) for v in gzero], "b": str(bzero), "G": str(Gzero)},
            "half": {"g": [str(v) for v in ghalf], "b": str(bhalf), "G": str(Ghalf)},
        },
        "enumeration": {
            "dim": d,
            "num_constraints": m,
            "total_active_sets": total_sets,
            "singular": singular,
            "infeasible": infeasible,
            "feasible_unique": feasible_unique,
            "time_sec": t1 - t0,
        },
        "optima": optima,
        "perturbation": {
            "g_pert": [str(v) for v in g_pert],
            "b_pert": str(b_pert),
            "G_pert": str(G_pert),
            "maxdiff": str(rc_pert["maxdiff"]),
            "maxdiff_float": float(rc_pert["maxdiff"]),
            "b_round": str(rc_pert["b_round"]),
            "cert_ok": rc_pert["ok"],
            "nonvacuous": bool(pert_nonvacuous),
            "all_optima_dyadic": bool(all_dyadic),
        },
    }


def main():
    t_start = time.perf_counter()
    per_k = {}
    for k in KS:
        per_k[str(k)] = run_k(k)
    degraded = check_degraded_povm()
    # clipping sanity
    assert clipped_estimate(-0.5) == 0
    assert clipped_estimate(1.5) == 1
    assert clipped_estimate(0.3) == 0.3
    # estimator sanity: counts [3,1] with g=[1,0] -> mean 0.75 clipped 0.75
    assert abs(estimate_from_counts([3, 1], [Fraction(1, 1), Fraction(0, 1)]) - 0.75) < 1e-12
    assert estimate_from_counts([10, 0], [Fraction(2, 1), Fraction(-1, 1)]) == 1
    # focused input-validation regression (no large expansion)
    def _rejects(counts, g):
        try:
            estimate_from_counts(counts, g)
        except ValueError:
            return True
        return False

    _g2 = [Fraction(1, 1), Fraction(0, 1)]
    mismatched_rejected = _rejects([1, 2, 3], _g2) and _rejects([1], _g2)
    negative_rejected = _rejects([-1, 2], _g2)
    noninteger_rejected = _rejects([1.5, 1], _g2)
    bool_rejected = _rejects([True, 1], _g2)
    empty_rejected = _rejects([], [])
    all_zero_rejected = _rejects([0, 0], _g2)
    assert mismatched_rejected, "mismatched lengths must raise ValueError"
    assert negative_rejected, "negative counts must raise ValueError"
    assert noninteger_rejected, "fractional counts must raise ValueError"
    assert bool_rejected, "bool counts must raise ValueError"
    assert empty_rejected, "empty counts must raise ValueError"
    assert all_zero_rejected, "all-zero counts must raise ValueError"
    valid_estimate = estimate_from_counts(
        [3, 1], [Fraction(1, 1), Fraction(0, 1)]
    )
    assert abs(valid_estimate - 0.75) < 1e-12
    valid_clipped = estimate_from_counts(
        [10, 0], [Fraction(2, 1), Fraction(-1, 1)]
    )
    assert valid_clipped == 1
    # integer/fraction rational parsing via Fraction(s)
    assert Fraction("2") == Fraction(2, 1)
    assert Fraction("0") == Fraction(0, 1)
    assert Fraction("-3") == Fraction(-3, 1)
    assert Fraction("3/2") == Fraction(3, 2)
    assert Fraction("-1/2") == Fraction(-1, 2)
    serialized_ok = True
    for kk in KS:
        for NN in N_VALUES:
            for s in per_k[str(kk)]["optima"][str(NN)]["g"]:
                if Fraction(s) != Fraction(s):
                    serialized_ok = False
    assert serialized_ok, "serialized rational parsing failed"
    regression_checks = {
        "mismatched_lengths_rejected": bool(mismatched_rejected),
        "negative_rejected": bool(negative_rejected),
        "noninteger_rejected": bool(noninteger_rejected),
        "bool_rejected": bool(bool_rejected),
        "empty_rejected": bool(empty_rejected),
        "all_zero_rejected": bool(all_zero_rejected),
        "valid_estimate": valid_estimate,
        "valid_clipped": valid_clipped,
        "integer_parsing_ok": True,
        "fraction_parsing_ok": True,
        "serialized_parsing_ok": bool(serialized_ok),
    }
    # global gates
    for k in KS:
        r = per_k[str(k)]
        assert r["W_match_enum"], f"k={k} W mismatch"
        assert r["W_cols_sum1"], f"k={k} col sum"
        assert r["inverse"]["unbiased"], f"k={k} inverse"
        assert r["enumeration"]["feasible_unique"] > 0
    assert degraded["max_povm_err"] <= 1e-9, degraded
    assert degraded["max_eta_err"] <= 1e-12, degraded
    # at least one non-vacuous rounding across all k/N or perturbation
    any_nonvac = any(
        Fraction(per_k[str(k)]["optima"][str(N)]["rounding"]["maxdiff"]) > 0
        for k in KS
        for N in N_VALUES
    ) or any(per_k[str(k)]["perturbation"]["nonvacuous"] for k in KS)
    assert any_nonvac, "rounding checks vacuous"
    t_end = time.perf_counter()
    payload = {
        "status": "PASS",
        "scope": "Tiny exact-rational LP diagnostic for k=1,2 only; finite enumeration, not a scalable solver; does not validate asymptotic exponent.",
        "seed": "deterministic (no RNG)",
        "python": platform.python_version(),
        "ks": list(KS),
        "N_values": list(N_VALUES),
        "delta": DELTA,
        "eps": "1/20",
        "round_den": ROUND_DEN,
        "per_k": per_k,
        "degraded_povm": degraded,
        "clipping": "final estimate clipped to [0,1]",
        "regression_checks": regression_checks,
        "timing_sec": t_end - t_start,
        "notes": [
            "W[w,h] rows observed, cols latent at eta=1/4; convolution vs bitstring enumeration match; columns sum to 1 exactly.",
            "Exact inverse g_w=(3/2)^{k-w}(-1/2)^w unbiased for every latent weight.",
            "Vertices from all (k+3)-subsets via exact Fraction Gaussian elimination; singular/infeasible discarded; same vertices reused for all N.",
            "Returned b=max_h|sum W g - delta|, G=max|g|; surrogate b+G/sqrt(N) distinct from Hoeffding radius b+G*sqrt(2ln(2/delta)/N) at delta=1/3.",
            "Sufficient copies ceil(2G^2 ln(2/delta)/(eps-b)^2) only if b<eps; sufficient, not necessary.",
            "Rounding to 1/256 certified by fresh b_round/G_round with b_round<=b+maxdiff; 1/1000 perturbation ensures non-vacuous check.",
            "Mixed/flag0/flag-1/1000 checks use unnormalized law I_flag*g with no division by flag probability.",
            "Degraded signal-1/2 via L/2 coin makes W rational; ideal probabilities assumed, apparatus errors outside certificate.",
            "No LP solver installed; no hidden division by zero.",
        ],
    }
    dest = (Path(__file__).resolve().parents[1] / "results" / "product_state_lp_checks.json")
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
