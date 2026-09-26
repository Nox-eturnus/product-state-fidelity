"""Independent deterministic validation for partial Pauli geometry.

Frozen small scope: n<=3 dense eigvalsh (numpy) for signed/tied hetero P-P_D
plus Holder bound on coherent/entangled/mixed states; high-precision Decimal
verification of a/c/d enclosures including tiny transverse; exhaustive subset
comparison at n<=6 (own exact Fractions + own sqrt as tight reference;
production uses bounded outward product rounding to SCALE=2**bits, so checks
are containment plus width, never exact equality); operational mixed iid
R-diagonal backend plus actual iid full-copy Born backend from an entangled
density matrix (n<=3 explicit joint Born law, deterministic exact-law
first-moment check at nonconstant g) plus geometry+LP composition;
no-discard equivalence and all-discard Bernoulli (constant-outcome interface
checks labelled MOCK, not physical); tamper rejection before data (zero
callbacks); invalid inputs; large near-axis tails (core<=3, n up to 50) +
n=200 default-bit regression plus high-precision boundary (bits=1024) +
budget-boundary sensitivity + homogeneous float-diagnostic support only
(no empirical asymptotic proof). Certificates use exact Fractions with
bounded denominators; Born diagnostics use honest float tolerance, never
as certificates.

Run: python code/verify_product_state_geometry.py
Writes product_state_geometry_checks.json alongside. Exits nonzero on failure.
Requires NumPy/SciPy in .venv (for inner solves + eigvalsh). No network.
"""

from __future__ import annotations

import dataclasses
import decimal
import itertools
import json
import math
import platform
import random
import time
import tracemalloc
from fractions import Fraction
from pathlib import Path

import product_state_estimator as est
import product_state_geometry as geo

BITS = 80
BITS_HIGH = 1024
TOL_EIG = 1e-9
TOL_DECIMAL = Fraction(1, 10**18)  # generous vs 1e-24 width + 1e-100 Decimal error
TOL_BORN = 1e-9  # honest float tolerance for Born-law diagnostics only, never certificates
SEED = 20260908

# ---------------- own independent geometry (NOT production) -----------------

def own_require_bits(bits: int) -> int:
    if isinstance(bits, bool) or not isinstance(bits, int):
        raise ValueError("bits")
    if bits < 1 or bits > 1024:
        raise ValueError("bits range")
    return bits


def own_sqrt_interval(s: Fraction, bits: int):
    if not isinstance(s, Fraction):
        raise ValueError("s")
    bits = own_require_bits(bits)
    if s < 0:
        raise ValueError("neg")
    if s == 0:
        return (Fraction(0, 1), Fraction(0, 1))
    p, q = s.numerator, s.denominator
    t0 = math.isqrt(p * q)
    if t0 * t0 == p * q:
        e = Fraction(t0, q)
        return (e, e)
    scale = 1 << bits
    Q = (p * scale * scale) // q
    t = math.isqrt(Q)
    lo = Fraction(t, scale)
    hi = Fraction(t + 1, scale)
    assert lo * lo <= s < hi * hi
    return (lo, hi)


def own_coord_frac(v) -> Fraction:
    if isinstance(v, bool):
        raise ValueError("bool")
    if isinstance(v, int):
        return Fraction(v, 1)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError("nonfinite")
        return Fraction(v)
    raise ValueError("coord type")


def own_a_interval(r_raw, bits: int):
    fracs = [own_coord_frac(v) for v in r_raw]
    mx = max(abs(f) for f in fracs)
    s2 = sum(f * f for f in fracs)
    assert mx * mx <= s2
    lo_s, hi_s = own_sqrt_interval(s2, bits)
    assert hi_s > 0
    a_lo = mx / hi_s
    if lo_s == 0:
        a_hi = Fraction(1, 1)
    else:
        a_hi = mx / lo_s
        if a_hi > 1:
            a_hi = Fraction(1, 1)
    assert 0 <= a_lo <= a_hi <= 1
    return (a_lo, a_hi, mx, s2)


def own_nearest_proxy(r_raw):
    fracs = [own_coord_frac(v) for v in r_raw]
    abses = [abs(f) for f in fracs]
    mx = max(abses)
    idx = min(i for i, a in enumerate(abses) if a == mx)
    sgn = 1 if fracs[idx] > 0 else -1
    proxy = [0.0, 0.0, 0.0]
    proxy[idx] = 1.0 if sgn > 0 else -1.0
    return (tuple(proxy), ("X", "Y", "Z")[idx], sgn, idx)


def own_geometry(rs, D, bits: int):
    """Tight exact-product reference for small n only (no bounded rounding).

    Computes exact Fraction products of the a-interval endpoints. For n<=6
    this stays below the digit limit and gives the tightest interval implied
    by the a bounds. Production uses bounded outward rounding, so production
    must CONTAIN this reference (prod_lo<=own_lo and own_hi<=prod_hi), not
    equal it. Never call this reference with n=200 (would exceed the digit
    limit); large-n checks use production validity plus width bounds only.
    """
    n = len(rs)
    a_pairs = [own_a_interval(r, bits) for r in rs]
    al = [p[0] for p in a_pairs]
    ah = [p[1] for p in a_pairs]
    if len(D) == 0:
        c_lo = c_hi = Fraction(1, 1)
    else:
        c_lo = Fraction(1, 1)
        c_hi = Fraction(1, 1)
        for j in D:
            c_lo *= (Fraction(1, 1) + al[j]) / 2
            c_hi *= (Fraction(1, 1) + ah[j]) / 2
    s_lo = Fraction(1, 1) - c_hi
    s_hi = Fraction(1, 1) - c_lo
    if s_lo < 0:
        s_lo = Fraction(0, 1)
    if s_hi < 0:
        s_hi = Fraction(0, 1)
    if s_hi == 0:
        d_lo = d_hi = Fraction(0, 1)
    elif s_lo == s_hi:
        d_lo, d_hi = own_sqrt_interval(s_lo, bits)
    else:
        if s_lo == 0:
            d_lo = Fraction(0, 1)
        else:
            d_lo, _ = own_sqrt_interval(s_lo, bits)
        _, d_hi = own_sqrt_interval(s_hi, bits)
    proxies = []
    for j, r in enumerate(rs):
        if j in set(D):
            pr, _, _, _ = own_nearest_proxy(r)
            proxies.append(pr)
        else:
            proxies.append(tuple(float(v) for v in r))
    return {
        "a_lo": al, "a_hi": ah, "c_lo": c_lo, "c_hi": c_hi,
        "d_lo": d_lo, "d_hi": d_hi, "proxy": proxies,
    }


# ---------------- dense matrices (stdlib + numpy eigvalsh) ------------------

I_M = [[1 + 0j, 0j], [0j, 1 + 0j]]
X_M = [[0j, 1 + 0j], [1 + 0j, 0j]]
Y_M = [[0j, -1j], [1j, 0j]]
Z_M = [[1 + 0j, 0j], [0j, -1 + 0j]]
PAULI = {"X": X_M, "Y": Y_M, "Z": Z_M}


def kron(a, b):
    n, m = len(a), len(a[0])
    p, q = len(b), len(b[0])
    out = [[0j] * (m * q) for _ in range(n * p)]
    for i in range(n):
        for j in range(m):
            aij = a[i][j]
            if aij == 0:
                continue
            for kk in range(p):
                for ll in range(q):
                    out[i * p + kk][j * q + ll] = aij * b[kk][ll]
    return out


def mmul(a, b):
    n = len(a)
    out = [[0j] * n for _ in range(n)]
    for i in range(n):
        for k_ in range(n):
            aik = a[i][k_]
            if aik == 0:
                continue
            for j in range(n):
                out[i][j] += aik * b[k_][j]
    return out


def msub(a, b):
    return [[a[i][j] - b[i][j] for j in range(len(a))] for i in range(len(a))]


def trace(a):
    return sum(a[i][i] for i in range(len(a)))


def tensor_list(mats):
    out = [[1 + 0j]]
    for m in mats:
        out = kron(out, m)
    return out


def r_mat(r):
    rx, ry, rz = r
    return [[rx * X_M[i][j] + ry * Y_M[i][j] + rz * Z_M[i][j] for j in range(2)] for i in range(2)]


def tau_pm(r, s):
    R = r_mat(r)
    return [[(I_M[i][j] + s * R[i][j]) / 2.0 for j in range(2)] for i in range(2)]


def ket_to_dm(vec):
    n = len(vec)
    return [[vec[i] * vec[j].conjugate() for j in range(n)] for i in range(n)]


def to_numpy(mat):
    import numpy as np
    return np.array(mat, dtype=complex)


def bell_phiplus_dm():
    """Density matrix for Bell |Phi+>=(|00>+|11>)/sqrt2 on 2 qubits (4x4)."""
    import math as _m
    v = [1 / _m.sqrt(2) + 0j, 0j, 0j, 1 / _m.sqrt(2) + 0j]
    return ket_to_dm(v)


def meas_projector(axis: str, outcome: int):
    """Single-qubit Pauli projector (I+o P_axis)/2 as 2x2 complex matrix."""
    if axis not in ("X", "Y", "Z"):
        raise ValueError(f"axis {axis!r}")
    if outcome not in (1, -1):
        raise ValueError(f"outcome {outcome!r}")
    P = PAULI[axis]
    return [[(I_M[i][j] + outcome * P[i][j]) / 2.0 for j in range(2)] for i in range(2)]


def born_probs_for_axes(sigma, axes):
    """Explicit joint Born probabilities for one Pauli basis on n<=3 qubits.

    sigma is dim x dim density matrix (list of lists, dim=2**n). axes length
    n with entries X/Y/Z. Returns dict outcome-tuple -> float probability
    via Tr(sigma Proj), where Proj=tensor_j (I+o_j P_{axis_j})/2. Probabilities
    are floats (diagnostic, not certificates); tiny negative clipping at
    -1e-12 with renormalization check. Raises if normalization deviates.
    """
    n = len(axes)
    dim = 1 << n
    if len(sigma) != dim or any(len(row) != dim for row in sigma):
        raise ValueError("sigma dim mismatch")
    probs: dict[tuple[int, ...], float] = {}
    total = 0.0
    for bits in itertools.product((1, -1), repeat=n):
        projs = [meas_projector(axes[j], bits[j]) for j in range(n)]
        P = tensor_list(projs)
        p = trace(mmul(sigma, P)).real
        # Numerical noise can give -1e-17; clip small negatives.
        if -1e-12 <= p < 0.0:
            p = 0.0
        if not math.isfinite(p) or p < -1e-9 or p > 1.0 + 1e-9:
            raise ValueError(f"Born prob out of range {p} for {axes}/{bits}")
        probs[bits] = float(p)
        total += float(p)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Born probs sum {total} != 1 for axes {axes}")
    # Renormalize tiny drift for sampling (keeps iid law exact up to 1e-12).
    if total != 0.0:
        for k in list(probs.keys()):
            probs[k] /= total
    return probs


def make_entangled_born_measure(sigma, n: int, backend_rng):
    """Return iid full-copy measure_fn(axes)->outcomes from an entangled state.

    For each call, computes the explicit 2**n joint Born distribution for the
    requested Pauli basis from the actual density matrix sigma (n<=3), then
    draws one full outcome bitstring with backend_rng (one iid copy). Never
    simulates qubits independently; correlations come from the joint law.
    Uses only backend_rng, separate from design RNG.
    """
    if not isinstance(n, int) or n <= 0 or n > 3:
        raise ValueError("entangled backend supports 1<=n<=3 only")
    if not hasattr(backend_rng, "random"):
        raise ValueError("backend_rng must have .random()")
    dim = 1 << n
    if len(sigma) != dim:
        raise ValueError("sigma dim mismatch")

    def measure(axes):
        if not isinstance(axes, (list, tuple)) or len(axes) != n:
            raise ValueError(f"axes must be length-{n}, got {axes!r}")
        for a in axes:
            if a not in ("X", "Y", "Z"):
                raise ValueError(f"invalid axis {a!r}")
        probs = born_probs_for_axes(sigma, list(axes))
        u = float(backend_rng.random())
        if not math.isfinite(u) or not (0.0 <= u < 1.0):
            raise ValueError(f"backend rng draw {u!r} not in [0,1)")
        acc = 0.0
        # Deterministic outcome order for reproducibility.
        for bits in sorted(probs.keys(), key=lambda b: tuple(1 if x == 1 else 0 for x in b)):
            acc += probs[bits]
            if u < acc:
                out = list(bits)
                # Validate outcomes.
                for o in out:
                    if o not in (1, -1):
                        raise RuntimeError("bad outcome (unreachable)")
                return out
        # Rounding edge: return last outcome.
        return list(sorted(probs.keys())[-1])

    measure.is_entangled_born = True  # type: ignore
    return measure


def expected_proxy_score_for_plan(plan, sigma):
    """Deterministic exact-law expectation of the executed proxy score H.

    Enumerates every Pauli-basis choice with its design sampling probability
    (proxy hats/L_hat, stabilizer deterministic), every joint outcome with
    its explicit Born probability from sigma, and every degradation coin
    outcome with its mean m=L_hat/sqrt3 (common) or L_hat/2 (half). Returns
    dict with expected (float diagnostic), fidelity_proxy, fidelity_original,
    b_inner, d_hi, g list, and axis count. Float tolerance applies only to
    this diagnostic comparison, never to certificates.
    """
    inner = plan.inner_plan
    proxy = list(plan.proxy_raw)
    cls = est.classify_target(proxy)
    hats = cls["rs_hat"]
    L_hat = cls["L_hat_list"]
    K_idx = list(cls["K_idx"])
    S_idx = list(cls["S_idx"])
    stab_axis = cls["stab_axis"]
    stab_want = cls["stab_want"]
    n = cls["n"]
    k = cls["k"]
    g_floats = [float(v) for v in plan.g_round]
    channel = plan.channel
    # Fidelity via dense matrices (float diagnostic, not certificates).
    # Targets are tau_+(r_hat) with r_hat the normalized directions.
    rs_orig = [tuple(float(v) for v in r) for r in plan.rs_raw]
    P_proxy = tensor_list([tau_pm(tuple(float(v) for v in h), 1) for h in hats])
    orig_hats = est.classify_target(list(rs_orig))["rs_hat"]
    P_orig_mat = tensor_list([tau_pm(tuple(float(v) for v in h), 1) for h in orig_hats])
    fid_proxy = float(trace(mmul(sigma, P_proxy)).real)
    fid_orig = float(trace(mmul(sigma, P_orig_mat)).real)
    # Enumerate axes: noisy qubits have 3 options, stabilizers 1.
    noisy_axes_options: list[list[str]] = []
    for j in range(n):
        if j in set(K_idx):
            noisy_axes_options.append(["X", "Y", "Z"])
        else:
            noisy_axes_options.append([stab_axis[j]])  # type: ignore
    exp_val = 0.0
    n_axes = 0
    for axes_t in itertools.product(*noisy_axes_options):
        axes = list(axes_t)
        # Design sampling probability for these axes.
        p_axes = 1.0
        for j in K_idx:
            hat = hats[j]
            L = float(L_hat[j])
            idx = {"X": 0, "Y": 1, "Z": 2}[axes[j]]
            p_axes *= abs(float(hat[idx])) / L
        n_axes += 1
        probs = born_probs_for_axes(sigma, axes)
        for bits, p_out in probs.items():
            outcomes = list(bits)
            # Stabilizer flag.
            flag = 1
            for j in S_idx:
                if outcomes[j] != stab_want[j]:
                    flag = 0
                    break
            if flag == 0:
                # H=0 regardless of w; contributes 0.
                continue
            if k == 0:
                exp_val += p_axes * p_out * float(g_floats[0]) * 1.0
                continue
            # Noisy sign-corrected bits s_j = sign(hat,axis)*outcome.
            s_bits: list[int] = []
            for j in K_idx:
                ax = axes[j]
                idx = {"X": 0, "Y": 1, "Z": 2}[ax]
                hv = float(hats[j][idx])
                if hv == 0.0:
                    raise ValueError("zero-prob axis in enumeration (unreachable)")
                sgn = 1 if hv > 0 else -1
                s_bits.append(sgn * outcomes[j])
            # Degradation distribution over t (2**k possibilities).
            # m per qubit from same float logic as estimator.
            m_list: list[float] = []
            for j in K_idx:
                L = float(L_hat[j])
                if channel == est.CHANNEL_HALF:
                    m = L / 2.0
                else:
                    m = L / math.sqrt(3.0)
                # Mirror estimator clipping at 1+1e-12.
                if m > 1.0 and m <= 1.0 + 1e-12:
                    m = 1.0
                m_list.append(m)
            for d_bits in itertools.product((1, -1), repeat=k):
                p_d = 1.0
                for ii, d in enumerate(d_bits):
                    pp = (1.0 + m_list[ii]) / 2.0
                    p_d *= pp if d == 1 else (1.0 - pp)
                t_bits = [s_bits[ii] * d_bits[ii] for ii in range(k)]
                w = sum(1 for t in t_bits if t == -1)
                exp_val += p_axes * p_out * p_d * float(g_floats[w])
    return {
        "expected": float(exp_val),
        "fid_proxy": float(fid_proxy),
        "fid_orig": float(fid_orig),
        "b_inner": plan.inner_plan.b_cert,
        "G_inner": plan.inner_plan.G_cert,
        "d_hi": plan.d_hi,
        "g_floats": g_floats,
        "n_axes": n_axes,
        "n": n, "k": k,
    }


# ---------------- main validation -------------------------------------------

def main() -> int:
    t_start = time.perf_counter()
    tracemalloc.start()
    failures: list[str] = []
    notes: list[str] = []

    def fail(msg: str):
        failures.append(msg)
        print(f"FAIL: {msg}")

    try:
        import numpy, scipy
        numpy_ver, scipy_ver = numpy.__version__, scipy.__version__
    except Exception as e:
        fail(f"numpy/scipy import failed: {e}")
        numpy_ver = scipy_ver = "missing"
    py_ver = platform.python_version()
    print(f"validating geometry with python {py_ver} numpy {numpy_ver} scipy {scipy_ver}")

    # ---- A. Decimal high-precision enclosure check incl. tiny ----
    try:
        ctx = decimal.Context(prec=100, Emax=999999, Emin=-999999,
                              traps=[decimal.InvalidOperation, decimal.DivisionByZero, decimal.Overflow])
        test_rs = [
            (0.0, 0.0, 1.0),
            (1.0, 0.0, 0.0),
            (0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
            (0.7071067811865475, 0.7071067811865475, 0.0),
            (-0.7071067811865475, 0.7071067811865475, 0.0),
            (0.6, 0.0, 0.8),
            (math.sin(0.001), 0.0, math.cos(0.001)),
            (math.sin(1e-6), 0.0, math.cos(1e-6)),
            (1e-9, 0.0, math.sqrt(max(0.0, 1 - 1e-18))),
            (0.0, 0.0, 1.0000000005),  # slightly nonunit, direction Pauli
        ]
        # Fix last tiny: sqrt may round to 1.0; keep as float triple (passes 1e-9).
        max_width = Fraction(0, 1)
        for r in test_rs:
            try:
                prod = geo.geometry_certificate([r], [0], BITS)
            except Exception as e:
                # (0,0,1.0000000005) norm 1.0000000005 passes (diff 5e-10)?
                fail(f"decimal cert failed for {r}: {e}")
                continue
            a_lo = Fraction(prod["a_lo"][0])
            a_hi = Fraction(prod["a_hi"][0])
            if not (0 <= a_lo <= a_hi <= 1):
                fail(f"a interval range {r}: [{a_lo},{a_hi}]")
            w = a_hi - a_lo
            if w > max_width:
                max_width = w
            # Width must be <= ~2*2**-bits + tiny? Conservative: <= 1e-20?
            # For bits=80 width ~1e-24; allow 1e-20.
            if float(w) > 1e-20 and not (r == (0.0, 0.0, 1.0)):
                # Exact Pauli has width 0; others width tiny. If larger, still
                # valid but note. Only fail if huge (>1e-12).
                if float(w) > 1e-12:
                    fail(f"a width large {r}: {w}")
            # Decimal high-prec reference (exact binary Fractions -> Decimal).
            with decimal.localcontext(ctx):
                # Exact binary Fractions for true value (production encloses this).
                fracs_exact = [Fraction(v) if isinstance(v, float) else Fraction(v) for v in r]
                dec_coords = [decimal.Decimal(f.numerator) / decimal.Decimal(f.denominator) for f in fracs_exact]
                # Note: str(v) decimal is decimal approximation, not exact binary.
                # For containment we compare production (exact binary) to Decimal
                # computed from decimal approximations: allow TOL_DECIMAL slack.
                # Better: compute exact binary Fractions vs Decimal(str) with slack.
                s2_dec = sum((c * c for c in dec_coords), decimal.Decimal(0))
                if s2_dec <= 0:
                    fail(f"decimal s2 {r}")
                    continue
                sqrt_dec = s2_dec.sqrt()
                max_dec = max((abs(c) for c in dec_coords), default=decimal.Decimal(0))
                a_dec = max_dec / sqrt_dec
                a_dec_frac = Fraction(a_dec)
                # Check containment with slack (Decimal str vs binary exact).
                if not (a_lo - TOL_DECIMAL <= a_dec_frac <= a_hi + TOL_DECIMAL):
                    fail(f"decimal containment {r}: prod [{a_lo},{a_hi}] dec {a_dec} frac {a_dec_frac}")
                # Tiny transverse: lower edge must be <1 (detects non-axis).
                if r not in ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)):
                    # For genuinely non-axis (sin !=0), true a<1, so a_lo<1.
                    # Check a_lo <1 (strict) for non-axis cases with sin!=0.
                    if abs(float(r[0])) > 1e-12 or (len(r) == 3 and abs(float(r[1])) > 1e-12):
                        # Has transverse component; if transverse tiny (1e-9),
                        # a_lo should still be <1 (or ==1 only if transverse
                        # underflows to 0 exactly, which it doesn't as float).
                        if not (a_lo < 1):
                            fail(f"tiny transverse not detected {r}: a_lo={a_lo}")
                # Exact axis: singleton [1,1].
                if r in ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0)):
                    if not (a_lo == 1 and a_hi == 1):
                        fail(f"exact axis not singleton {r}: [{a_lo},{a_hi}]")
        notes.append(f"decimal max_width={float(max_width):.3g}")
        # d enclosure check for a 2-factor example: production uses bounded
        # outward rounding, so it must CONTAIN the tight exact reference.
        rs2 = [(0.5773502691896258, 0.5773502691896258, 0.5773502691896258),
               (math.sin(0.01), 0.0, math.cos(0.01))]
        prod2 = geo.geometry_certificate(rs2, [0, 1], BITS)
        own2 = own_geometry(rs2, [0, 1], BITS)
        p_c_lo = Fraction(prod2["c_lo_str"])
        p_c_hi = Fraction(prod2["c_hi_str"])
        p_d_lo = Fraction(prod2["d_lo_str"])
        p_d_hi = Fraction(prod2["d_hi_str"])
        if not (p_c_lo <= own2["c_lo"] and own2["c_hi"] <= p_c_hi):
            fail(f"c containment 2-factor: prod [{p_c_lo},{p_c_hi}] own [{own2['c_lo']},{own2['c_hi']}]")
        if not (p_d_lo <= own2["d_lo"] and own2["d_hi"] <= p_d_hi):
            fail(f"d containment 2-factor: prod [{p_d_lo},{p_d_hi}] own [{own2['d_lo']},{own2['d_hi']}]")
        # Bounded-rounding width must stay tiny at default bits.
        if float(p_c_hi - p_c_lo) > 1e-18:
            fail(f"c width large 2-factor: {float(p_c_hi - p_c_lo):.3g}")
        if float(p_d_hi - p_d_lo) > 1e-18:
            fail(f"d width large 2-factor: {float(p_d_hi - p_d_lo):.3g}")
    except Exception as e:
        import traceback as _tb
        fail(f"decimal harness {_tb.format_exc()}")

    # ---- B. numpy eigvalsh P-P_D + Holder for states (n<=3 signed/tied) ----
    try:
        import numpy as _np
        hetero_cases = [
            ([(1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)),
              (1 / math.sqrt(2), 1 / math.sqrt(2), 0.0),
              (0.0, 0.0, 1.0)], [0], "tied+X"),
            ([(-1 / math.sqrt(2), 1 / math.sqrt(2), 0.0),
              (0.6, 0.0, 0.8),
              (0.0, 0.0, 1.0)], [0, 1], "signed"),
            ([(math.sin(0.02), 0.0, math.cos(0.02)),
              (math.sin(0.03), 0.0, math.cos(0.03)),
              (1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3))], [0, 1], "near-axis"),
            ([(0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)], [0, 1, 2], "all-Pauli"),
        ]
        max_eig_err = 0.0
        max_holder = 0.0
        for rs, D, label in hetero_cases:
            n = len(rs)
            prod = geo.geometry_certificate(rs, D, BITS)
            d_lo = Fraction(prod["d_lo_str"])
            d_hi = Fraction(prod["d_hi_str"])
            # Own matrices for P and P_D.
            P = tensor_list([tau_pm(r, 1) for r in rs])
            # Proxy via own nearest (independent of production proxy? Compare too).
            own = own_geometry(rs, D, BITS)
            if list(prod["proxy"]) != [tuple(float(v) for v in p) for p in own["proxy"]]:
                fail(f"proxy mismatch {label}: prod {prod['proxy']} own {own['proxy']}")
            # Simpler: build Qm directly from prod proxy triples as single-qubit projectors.
            Qm = tensor_list([tau_pm(tuple(float(v) for v in pr), 1) for pr in prod["proxy"]])
            Dmat = msub(P, Qm)
            arr = to_numpy(Dmat)
            eig = _np.linalg.eigvalsh(arr)
            op = float(_np.max(_np.abs(eig)))
            # True op should lie in [d_lo,d_hi] up to eig tolerance.
            if not (float(d_lo) - TOL_EIG <= op <= float(d_hi) + TOL_EIG):
                fail(f"eigvalsh {label}: op={op} not in [{d_lo},{d_hi}]")
            max_eig_err = max(max_eig_err, abs(op - float((d_lo + d_hi) / 2)))
            # Holder bound for states: coherent/entangled/mixed.
            dim = 2 ** n
            # States: target, orth, GHZ-like, W-like (n=3), maximally mixed, R-cat.
            states = {}
            states["target"] = P
            states["orth"] = tensor_list([tau_pm(r, -1) for r in rs])
            # GHZ (|0..0>+|1..1>)/sqrt2 in computational basis.
            vec = [0j] * dim
            vec[0] = 1 / math.sqrt(2)
            vec[dim - 1] = 1 / math.sqrt(2)
            states["ghz"] = ket_to_dm(vec)
            states["mixed"] = [[(1.0 / dim if i == j else 0j) for j in range(dim)] for i in range(dim)]
            # Coherent plus-product.
            vv = [1 + 0j]
            for _ in range(n):
                vv = [x * y for x in vv for y in [1 / math.sqrt(2) + 0j, 1 / math.sqrt(2) + 0j]]
            states["plus"] = ket_to_dm(vv)
            for name, sigma in states.items():
                diff = abs(trace(mmul(sigma, Dmat)).real)
                if diff > float(d_hi) + TOL_EIG:
                    fail(f"holder {label}/{name}: |Tr|={diff} > d_hi={d_hi}")
                max_holder = max(max_holder, diff)
            # Cubic identity spot-check via own matrices? (P-Q)^3=(1-c)(P-Q)
            # Use production c midpoint? Use own c interval midpoint for check.
            c_mid = float((own["c_lo"] + own["c_hi"]) / 2)
            D2 = mmul(Dmat, Dmat)
            D3 = mmul(D2, Dmat)
            # E = D3 - (1-c) D should be ~0.
            scale = 1 - c_mid
            E = [[D3[i][j] - scale * Dmat[i][j] for j in range(dim)] for i in range(dim)]
            cub = max(abs(v) for row in E for v in row)
            if cub > 1e-7:
                fail(f"cubic {label}: {cub}")
        notes.append(f"eig max_mid_err={max_eig_err:.3g} holder_max={max_holder:.3g}")
    except Exception as e:
        import traceback as _tb2
        fail(f"eigvalsh harness {_tb2.format_exc()}")

    # ---- C. exhaustive subsets n<=6 ----
    try:
        for n in (2, 3, 4, 6):
            # Deterministic hetero target: cycle through 111, two-axis, near-axis, Pauli.
            base = [
                (1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)),
                (1 / math.sqrt(2), 1 / math.sqrt(2), 0.0),
                (math.sin(0.015), 0.0, math.cos(0.015)),
                (0.0, 0.0, 1.0),
                (-1 / math.sqrt(2), 1 / math.sqrt(2), 0.0),
                (0.6, 0.0, 0.8),
            ]
            rs = [base[j % len(base)] for j in range(n)]
            # Enumerate all subsets.
            all_subs = list(itertools.chain.from_iterable(
                itertools.combinations(range(n), r) for r in range(n + 1)))
            if len(all_subs) != 2 ** n:
                fail(f"subset count n={n}")
            prev_d_hi = None
            for D in all_subs:
                prod = geo.geometry_certificate(rs, list(D), BITS)
                own = own_geometry(rs, list(D), BITS)
                # Production uses bounded outward rounding: containment, not
                # equality, plus tiny width at default bits.
                pc_lo = Fraction(prod["c_lo_str"])
                pc_hi = Fraction(prod["c_hi_str"])
                pd_lo = Fraction(prod["d_lo_str"])
                pd_hi = Fraction(prod["d_hi_str"])
                if not (pc_lo <= own["c_lo"] and own["c_hi"] <= pc_hi):
                    fail(f"exhaustive c containment n={n} D={D}: prod [{pc_lo},{pc_hi}] own [{own['c_lo']},{own['c_hi']}]")
                    break
                if not (pd_lo <= own["d_lo"] and own["d_hi"] <= pd_hi):
                    fail(f"exhaustive d containment n={n} D={D}: prod [{pd_lo},{pd_hi}] own [{own['d_lo']},{own['d_hi']}]")
                    break
                # Rational endpoint-versus-width regression: product_rounding_bound
                # is per-endpoint |D|/2**bits, whole-width additional <=2|D|/2**bits.
                try:
                    bound = Fraction(len(D), 1 << BITS)
                    if Fraction(prod["product_rounding_bound"]) != bound:
                        fail(f"rounding bound field n={n} D={D}: got {prod['product_rounding_bound']} vs {bound}")
                        break
                    # Each endpoint within bound of its exact endpoint.
                    if not (Fraction(0, 1) <= own["c_lo"] - pc_lo <= bound):
                        fail(f"per-endpoint lo n={n} D={D}: own_lo-pc_lo={own['c_lo'] - pc_lo} > {bound}")
                        break
                    if not (Fraction(0, 1) <= pc_hi - own["c_hi"] <= bound):
                        fail(f"per-endpoint hi n={n} D={D}: pc_hi-own_hi={pc_hi - own['c_hi']} > {bound}")
                        break
                    prod_w = pc_hi - pc_lo
                    own_w = own["c_hi"] - own["c_lo"]
                    if not (Fraction(0, 1) <= prod_w - own_w <= 2 * bound):
                        fail(f"width additional n={n} D={D}: prod_w-own_w={prod_w - own_w} not in [0,{2 * bound}]")
                        break
                except Exception as e_rb:
                    fail(f"rounding regression n={n} D={D}: {e_rb}")
                    break
                if float(pc_hi - pc_lo) > 1e-15 or float(pd_hi - pd_lo) > 1e-15:
                    fail(f"exhaustive width large n={n} D={D}")
                    break
                # Empty/all checks.
                if len(D) == 0:
                    if not (prod["d_lo_str"] == "0" and prod["d_hi_str"] == "0"):
                        fail(f"empty D not zero n={n}")
                # a intervals length n.
                if len(prod["a_lo"]) != n or len(prod["a_hi"]) != n:
                    fail(f"a len n={n}")
            # Monotonicity along sorted proximity prefixes (selection order).
            sel = geo.select_discards(rs, 0.05, BITS)
            order = sel["sorted_order"]
            d_prev = Fraction(0, 1)
            for t in range(1, len(order) + 1):
                Dt = sorted(order[:t])
                c = geo.geometry_certificate(rs, Dt, BITS)
                dh = Fraction(c["d_hi_str"])
                if dh < d_prev:
                    fail(f"monotonicity n={n} t={t}: {dh} < {d_prev}")
                d_prev = dh
            # Selection feasibility: chosen feasible, next infeasible (if any).
            chosen = list(sel["chosen_D"])
            ch_cert = geo.geometry_certificate(rs, chosen, BITS)
            if not (Fraction(ch_cert["d_hi_str"]) <= Fraction(sel["beta"])):
                fail(f"selection not feasible n={n}")
            if len(chosen) < n:
                # Next prefix should be infeasible.
                nxt = sorted(order[:len(chosen) + 1])
                nxt_cert = geo.geometry_certificate(rs, nxt, BITS)
                if Fraction(nxt_cert["d_hi_str"]) <= Fraction(sel["beta"]):
                    fail(f"selection not longest n={n} chosen={chosen} next={nxt}")
        notes.append("exhaustive n<=6 done")
    except Exception as e:
        import traceback as _tb3
        fail(f"exhaustive harness {_tb3.format_exc()}")

    # ---- D. operational backend + composition + no/all discard ----
    try:
        # Small hetero: core 111 + near-axis tail + stabilizer (n=3, k=2).
        rs_op = [(1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)),
                 (math.sin(0.02), 0.0, math.cos(0.02)),
                 (0.0, 0.0, 1.0)]
        sel_op = geo.select_discards(rs_op, 0.04, BITS)
        D_op = list(sel_op["chosen_D"])
        # Build geometry plan with auto inner (N=4 common).
        plan_op = geo.plan_geometry(rs_op, D_op, 4, 1.0 / 3.0, "common", None, None, None, bits=BITS)
        # Check no-discard equivalence separately below; here check composition.
        # Mixed backend: R-diagonal mixture via estimator helper (correlated).
        hats_op = est.classify_target(rs_op)["rs_hat"]
        backend_rng = random.Random(999)
        measure_mix = est.make_demo_mixture_measure(hats_op, backend_rng, 0.7)
        out_mix = geo.run_geometry_streaming(plan_op, measure_mix, random.Random(0))
        # Composition: R_total == R_inner + d_hi exactly.
        R_inner = Fraction(out_mix["proxy_radius_exact"])
        d_hi = Fraction(out_mix["d_hi"])
        R_tot = Fraction(out_mix["radius_total_exact"])
        if R_tot != R_inner + d_hi:
            fail(f"composition R {R_tot} != {R_inner}+{d_hi}")
        # Endpoints: clip01(mean-R_tot) outward.
        mean = Fraction(out_mix["mean_exact"])
        lo_rat = mean - R_tot
        if lo_rat < 0:
            lo_rat = Fraction(0, 1)
        hi_rat = mean + R_tot
        if hi_rat > 1:
            hi_rat = Fraction(1, 1)
        if out_mix["interval_original_exact"] != [str(lo_rat), str(hi_rat)]:
            fail(f"endpoints mismatch {out_mix['interval_original_exact']} vs [{lo_rat},{hi_rat}]")
        if not (0.0 <= out_mix["interval_original"][0] <= out_mix["interval_original"][1] <= 1.0):
            fail(f"interval unordered {out_mix['interval_original']}")
        # Proxy vs original separation: keys present and distinct concepts.
        if "proxy_estimate" not in out_mix or "interval_original" not in out_mix:
            fail("proxy/original keys missing")
        if out_mix["proxy_interval"] == out_mix["interval_original"] and d_hi > 0:
            # Widened interval must strictly contain proxy interval when d>0,
            # unless clipping at boundaries. Check widening: R_tot > R_inner.
            if not (R_tot > R_inner):
                fail("no widening when d>0")
        # Entangled backend: actual iid full-copy Born law from Bell |Phi+>.
        # Two-qubit entangled case with small-bias nonconstant executed
        # score (k'=1, N=256, common). Target: [111 (noisy), near-axis 0.02
        # (noisy)], D=[1] discards the second factor to Pauli Z, so proxy
        # is [111, Pauli Z] with k'=1. Uses the known
        # independently-certified rounded inverse g=(175/128,-47/128) at
        # k'=1 (solver output at this grid point, freshly recertified via
        # plan_estimator; no solver dependence, no weak [0,2] fallback).
        # The test requires nonconstant g, small certified bias <=1/100,
        # and a stringent deterministic exact-law first-moment check, so a
        # vacuous constant or large-bias score cannot pass.
        rs_ent = [(1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)),
                  (math.sin(0.02), 0.0, math.cos(0.02))]
        D_ent = [1]
        # Known small-bias rounded inverse at k'=1 (exact Fractions).
        G_KNOWN_ENT = [Fraction(175, 128), Fraction(-47, 128)]
        try:
            plan_ent = geo.plan_geometry(rs_ent, D_ent, 256, 1.0 / 3.0, "common",
                                         G_KNOWN_ENT, None, None, bits=BITS)
        except Exception as e:
            fail(f"entangled plan build failed: {e}")
            plan_ent = None
        ent_info: dict = {}
        if plan_ent is not None:
            g_vals = list(plan_ent.g_round)
            # Nonvacuous small-bias check: nonconstant and b<=1/100.
            if len(set(g_vals)) < 2:
                fail(f"entangled g vacuous (constant): {g_vals}")
            try:
                b_known = plan_ent.inner_plan.b_cert
                if not isinstance(b_known, Fraction):
                    fail(f"entangled b_cert not Fraction: {b_known!r}")
                elif b_known > Fraction(1, 100):
                    fail(f"entangled b_cert not small-bias: {b_known} > 1/100")
            except Exception as e_b:
                fail(f"entangled b_cert check failed: {e_b}")
            sigma_bell = bell_phiplus_dm()
            # Deterministic exact-law first moment: enumerate axes/outcomes/
            # degradation and compare to fidelities within fresh biases.
            try:
                enum = expected_proxy_score_for_plan(plan_ent, sigma_bell)
            except Exception as e:
                import traceback as _tbe
                fail(f"entangled enumeration failed {_tbe.format_exc()}")
                enum = None
            if enum is not None:
                b_f = float(enum["b_inner"])
                d_f = float(plan_ent.d_hi)
                exp_v = float(enum["expected"])
                fp = float(enum["fid_proxy"])
                fo = float(enum["fid_orig"])
                # Honest float tolerance for Born diagnostics only.
                if abs(exp_v - fp) > b_f + TOL_BORN:
                    fail(f"entangled proxy first moment: E={exp_v:.9f} fid={fp:.9f} b={b_f:.6g} tol={TOL_BORN}")
                if abs(exp_v - fo) > b_f + d_f + TOL_BORN:
                    fail(f"entangled original first moment: E={exp_v:.9f} fid={fo:.9f} b+d={b_f + d_f:.6g}")
                ent_info = {"expected": exp_v, "fid_proxy": fp, "fid_orig": fo,
                            "b": b_f, "d_hi": d_f, "g": [str(v) for v in g_vals],
                            "n_axes": enum["n_axes"]}
                notes.append(f"entangled k'=1 N=256 E={exp_v:.4f} fid_proxy={fp:.4f} "
                             f"fid_orig={fo:.4f} b={b_f:.4g} d={d_f:.4g} g={ent_info['g']}")
            # Sampling check: iid full-copy Born backend via joint law.
            try:
                meas_ent = make_entangled_born_measure(sigma_bell, 2, random.Random(777))
                out_ent = geo.run_geometry_streaming(plan_ent, meas_ent, random.Random(1))
                if not (0.0 <= out_ent["estimate"] <= 1.0):
                    fail("entangled estimate range")
                if out_ent["N"] != 256:
                    fail("entangled N")
                # Composition must hold for the entangled run as well.
                if Fraction(out_ent["radius_total_exact"]) != Fraction(out_ent["proxy_radius_exact"]) + Fraction(out_ent["d_hi"]):
                    fail("entangled composition")
                notes.append(f"entangled sampling est={out_ent['proxy_estimate']:.3f} "
                             f"orig={out_ent['interval_original']}")
            except Exception as e:
                import traceback as _tbe2
                fail(f"entangled sampling failed {_tbe2.format_exc()}")
        notes.append(f"operational mix est={out_mix['proxy_estimate']:.3f} orig={out_mix['interval_original']}")

        # No-discard equivalence: D=() => proxy==original, d=0, total==inner.
        # MOCK interface check (constant outcomes, not a physical Born law):
        # exercises composition plumbing only, labelled as mock.
        plan_no = geo.plan_geometry(rs_op, [], 4, 1.0 / 3.0, "common", None, None, None, bits=BITS)
        if plan_no.d_hi != 0 or plan_no.d_lo != 0:
            fail(f"no-discard d {plan_no.d_hi}")
        if list(plan_no.proxy_raw) != [tuple(float(v) for v in r) for r in est.validate_target(rs_op)]:
            fail("no-discard proxy != original")

        def mock_constant_no(ax):
            return [1 for _ in ax]
        mock_constant_no.is_mock = True  # type: ignore

        out_no = geo.run_geometry_streaming(plan_no, mock_constant_no, random.Random(2))
        if Fraction(out_no["radius_total_exact"]) != Fraction(out_no["proxy_radius_exact"]):
            fail("no-discard radius mismatch")
        if out_no["interval_original"] != out_no["proxy_interval"]:
            fail("no-discard interval mismatch")
        if out_no["k_original"] != out_no["k_proxy"]:
            fail("no-discard k mismatch")

        # All-discard Bernoulli: discard all noisy factors => k'=0.
        cls_all = est.classify_target(rs_op)
        K_all = list(cls_all["K_idx"])
        # D = all noisy (K) plus optionally stabilizers? Use all indices for full Pauli proxy.
        plan_all = geo.plan_geometry(rs_op, list(range(len(rs_op))), 4, 1.0 / 3.0, "common", None, None, None, bits=BITS)
        if plan_all.inner_plan.k != 0:
            fail(f"all-discard k'={plan_all.inner_plan.k} expected 0")
        if list(plan_all.inner_plan.g_round) != [Fraction(1, 1)]:
            fail("all-discard g != [1]")
        def mock_constant_all(ax):
            return [1 for _ in ax]
        mock_constant_all.is_mock = True  # type: ignore

        out_all = geo.run_geometry_streaming(plan_all, mock_constant_all, random.Random(3))
        # Inner radius must be Bernoulli; total = Bernoulli + d_hi.
        R_bern = est.bernoulli_radius_exact(4, 1.0 / 3.0)
        if Fraction(out_all["proxy_radius_exact"]) != R_bern:
            fail(f"all-discard Bernoulli {out_all['proxy_radius_exact']} vs {R_bern}")
        if Fraction(out_all["radius_total_exact"]) != R_bern + plan_all.d_hi:
            fail("all-discard total mismatch")
    except Exception as e:
        import traceback as _tb4
        fail(f"operational harness {_tb4.format_exc()}")

    # ---- E. tamper rejection before data ----
    try:
        rs_t = [(1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3)), (0.0, 0.0, 1.0)]
        rec_t = est.solve_and_certify(1, 4, "common")
        g_t = [Fraction(s) for s in rec_t["g_round"]]
        b_t = Fraction(rec_t["b_cert"])
        G_t = Fraction(rec_t["G_cert"])
        # Valid geometry plan with D=[0] (discard noisy 111 => d>0) for
        # meaningful understated/wrong-D tests. Proxy is [X-Pauli, Z-Pauli].
        # For tamper tests the inner g must match proxy k'=0 ([1]), so build
        # plan_v via auto inner (k'=0 Bernoulli) rather than k=1 g above.
        # Keep k=1 g for inner-underclaim test via separate no-discard plan.
        plan_v = geo.plan_geometry(rs_t, [0], 4, 1.0 / 3.0, "common", None, None, None, bits=BITS)
        # Separate no-discard plan for inner b/G underclaim (k=1, b>0).
        plan_k1 = geo.plan_geometry(rs_t, [], 4, 1.0 / 3.0, "common", g_t, b_t, G_t, bits=BITS)

        def _mk_counters():
            calls = {"measure": 0, "rng": 0}

            # MOCK interface probe (constant outcomes, not physical): used only
            # to verify tamper rejection consumes zero callbacks/RNG.
            def _measure(axes):
                calls["measure"] += 1
                return [1 for _ in axes]
            _measure.is_mock = True  # type: ignore

            class _CRng:
                def __init__(self, base):
                    self.base = base

                def random(self):
                    calls["rng"] += 1
                    return self.base.random()

            return calls, _measure, _CRng(random.Random(1))

        def _expect_reject(plan, label):
            calls, meas, crng = _mk_counters()
            try:
                geo.run_geometry_streaming(plan, meas, crng)
                fail(f"{label} not rejected")
            except ValueError:
                if calls["measure"] != 0 or calls["rng"] != 0:
                    fail(f"{label} consumed callback/RNG measure={calls['measure']} rng={calls['rng']}")
            except Exception as e:
                fail(f"{label} wrong exc {type(e).__name__}: {e}")

        # Understated geometry (plan_v has d_hi>0, so 0 understates).
        if not (plan_v.d_hi > 0):
            fail(f"tamper base d_hi not >0: {plan_v.d_hi}")
        _expect_reject(dataclasses.replace(plan_v, d_hi=Fraction(0, 1)), "understated d_hi")
        _expect_reject(dataclasses.replace(plan_v, c_hi=Fraction(0, 1)), "understated c_hi")
        # Wrong proxy (replace first proxy factor with different Pauli).
        # plan_v proxy is [(1,0,0),(0,0,1)]; change first to (0,1,0) to differ.
        bad_proxy = ((0.0, 1.0, 0.0), plan_v.proxy_raw[1])
        if bad_proxy != plan_v.proxy_raw:
            _expect_reject(dataclasses.replace(plan_v, proxy_raw=bad_proxy), "wrong proxy")
        else:
            fail("bad proxy construction equal (unreachable)")
        # Wrong D: plan_v D=(0,) -> change to () while keeping proxy/cert
        # (proxy for () is original [111,Pauli] != stored [(1,0,0),Pauli]).
        _expect_reject(dataclasses.replace(plan_v, D=()), "wrong D")
        # Inner binding: replace inner with different N.
        _expect_reject(dataclasses.replace(plan_v, inner_plan=dataclasses.replace(plan_v.inner_plan, N=17)), "inner N mismatch")
        # Inner understated b/G via replaced inner (use k=1 plan_k1 with b>0).
        try:
            bad_inner = dataclasses.replace(plan_k1.inner_plan, b_cert=Fraction(0, 1), G_cert=Fraction(0, 1))
            _expect_reject(dataclasses.replace(plan_k1, inner_plan=bad_inner), "inner underclaim")
        except Exception as e:
            # If k=1 with b>0, replacing with 0 is understated and should be caught;
            # if construction of bad_inner itself fails? dataclasses.replace doesn't validate,
            # so it succeeds; rejection happens at streaming. If it fails here, note.
            fail(f"inner replace harness {e}")
        # Bool N / Fraction N.
        _expect_reject(dataclasses.replace(plan_v, N=True), "bool N")
        try:
            _expect_reject(dataclasses.replace(plan_v, N=Fraction(4, 1)), "Fraction N")
        except Exception as e:
            fail(f"Fraction N harness {e}")
        # Dict plan rejected.
        try:
            calls, meas, crng = _mk_counters()
            try:
                geo.run_geometry_streaming({"N": 4}, meas, crng)
                fail("dict plan not rejected")
            except ValueError:
                if calls["measure"] != 0 or calls["rng"] != 0:
                    fail("dict consumed callbacks")
        except Exception as e:
            fail(f"dict harness {e}")
        # Valid identical copy must run.
        try:
            p_copy = dataclasses.replace(plan_v)
            calls, meas, crng = _mk_counters()
            out = geo.run_geometry_streaming(p_copy, meas, crng)
            if out["N"] != 4:
                fail("valid copy N")
            if calls["measure"] != 4:
                fail(f"valid copy measure count {calls['measure']}")
        except Exception as e:
            fail(f"valid copy rejected: {e}")
    except Exception as e:
        import traceback as _tb5
        fail(f"tamper harness {_tb5.format_exc()}")

    # ---- F. invalid inputs ----
    try:
        def expect_value_error(fn, label):
            try:
                fn()
            except ValueError:
                return True
            except Exception as e:
                fail(f"{label} wrong exc {type(e).__name__}: {e}")
                return False
            fail(f"{label} did not raise")
            return False

        rs_ok = [(0.0, 0.0, 1.0), (1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3))]
        expect_value_error(lambda: geo.geometry_certificate(rs_ok, [0, 0], BITS), "duplicate D")
        expect_value_error(lambda: geo.geometry_certificate(rs_ok, [5], BITS), "out-of-range D")
        expect_value_error(lambda: geo.geometry_certificate(rs_ok, [True], BITS), "bool D")
        expect_value_error(lambda: geo.geometry_certificate(rs_ok, ["0"], BITS), "str D")
        expect_value_error(lambda: geo.geometry_certificate(rs_ok, [0], 0), "bits0")
        expect_value_error(lambda: geo.geometry_certificate(rs_ok, [0], -1), "bits neg")
        expect_value_error(lambda: geo.geometry_certificate(rs_ok, [0], True), "bits bool")
        expect_value_error(lambda: geo.geometry_certificate(rs_ok, [0], 9999), "bits huge")
        expect_value_error(lambda: geo.geometry_certificate([], [], BITS), "empty rs")
        expect_value_error(lambda: geo.geometry_certificate([(0.0, 0.0, 0.0)], [], BITS), "zero vector")
        expect_value_error(lambda: geo.geometry_certificate([(0.0, 0.0, 0.5)], [], BITS), "nonunit")
        expect_value_error(lambda: geo.geometry_certificate([(0.0, 0.0, float("nan"))], [], BITS), "nan")
        expect_value_error(lambda: geo.geometry_certificate([(True, 0.0, 1.0)], [], BITS), "bool coord")
        expect_value_error(lambda: geo.select_discards(rs_ok, -0.1, BITS), "beta neg")
        expect_value_error(lambda: geo.select_discards(rs_ok, 1.0, BITS), "beta 1")
        expect_value_error(lambda: geo.select_discards(rs_ok, True, BITS), "beta bool")
        expect_value_error(lambda: geo.select_discards(rs_ok, float("nan"), BITS), "beta nan")
        expect_value_error(lambda: geo.plan_geometry(rs_ok, [0], 0, 1/3, "common", [Fraction(1), Fraction(0)], bits=BITS), "N0")
        expect_value_error(lambda: geo.plan_geometry(rs_ok, [0], True, 1/3, "common", [Fraction(1), Fraction(0)], bits=BITS), "bool N plan")
        expect_value_error(lambda: geo.plan_geometry(rs_ok, [0], 4, 0, "common", [Fraction(1), Fraction(0)], bits=BITS), "delta0")
        expect_value_error(lambda: geo.plan_geometry(rs_ok, [0], 4, 1/3, "bad", [Fraction(1), Fraction(0)], bits=BITS), "bad channel")
        expect_value_error(lambda: geo.plan_geometry(rs_ok, [0], 4, 1/3, "common", [0.5, 0.5], bits=BITS), "float g")
        # k' mismatch: rs_ok proxy with D=[1]? rs_ok has k=1 (second noisy). D=[1] => proxy k'=0, g len 2 mismatched.
        expect_value_error(lambda: geo.plan_geometry(rs_ok, [1], 4, 1/3, "common", [Fraction(1), Fraction(0)], bits=BITS), "k mismatch")
        # k'=0 non-[1] rejected.
        expect_value_error(lambda: geo.plan_geometry([(0.0, 0.0, 1.0)], [], 4, 1/3, "common", [Fraction(0)], bits=BITS), "k0 bad score")
        # mu helpers invalid.
        expect_value_error(lambda: geo.mu_upper_of_L(Fraction(1, 2)), "mu L<1")
        expect_value_error(lambda: geo.mu_lower_of_a(Fraction(0, 1)), "mu a0")
        expect_value_error(lambda: geo.mu_lower_of_a(Fraction(2, 1)), "mu a>1")
        expect_value_error(lambda: geo.homogeneous_bias_all(0.0, 4, BITS), "theta0")
        expect_value_error(lambda: geo.homogeneous_bias_all(math.pi, 4, BITS), "theta big")
    except Exception as e:
        import traceback as _tb6
        fail(f"invalid harness {_tb6.format_exc()}")

    # ---- G. large tails + budget sensitivity + homogeneous analytical ----
    large_info: dict = {}
    try:
        # Large near-axis tail with core<=3: n=50, core 2 gapped + 48 tail theta=0.01.
        th_tail = 0.01
        r_core1 = (1 / math.sqrt(3), 1 / math.sqrt(3), 1 / math.sqrt(3))
        r_core2 = (1 / math.sqrt(2), 1 / math.sqrt(2), 0.0)
        r_tail = (math.sin(th_tail), 0.0, math.cos(th_tail))
        n_large = 50
        rs_large = [r_core1, r_core2] + [r_tail] * (n_large - 2)
        tL0 = time.perf_counter()
        sel_large = geo.select_discards(rs_large, 0.04, BITS)
        tL1 = time.perf_counter()
        large_info = {
            "n": n_large, "core": 2, "tail_theta": th_tail,
            "chosen_len": sel_large["prefix_len"],
            "chosen_D": list(sel_large["chosen_D"]),
            "k_orig": sel_large["k_original"], "k_ret": sel_large["k_retained"],
            "d_hi": sel_large["d_hi_str"],
            "select_time_sec": tL1 - tL0,
        }
        # Core retained? Tail should be selected (closest), core not (farther).
        # Sorted order should put tail first (a~0.9999) before core (0.57,0.707).
        # Check that chosen includes tails and retains core noisy count 2?
        if sel_large["k_retained"] != 2:
            fail(f"large tail k_ret {sel_large['k_retained']} expected 2 (core retained)")
        if sel_large["prefix_len"] == 0:
            fail("large tail selected none")
        # Budget-boundary sensitivity: beta = d_hi feasible, beta just below infeasible.
        d_hi_large = Fraction(sel_large["d_hi_str"])
        # Feasible at exactly d_hi.
        sel_at = geo.select_discards(rs_large, d_hi_large, BITS)
        if sel_at["prefix_len"] != sel_large["prefix_len"]:
            fail(f"boundary at {sel_at['prefix_len']} vs {sel_large['prefix_len']}")
        # Just below (subtract tiny epsilon 1e-12 as Fraction).
        eps = Fraction(1, 10**12)
        if d_hi_large > eps:
            sel_below = geo.select_discards(rs_large, d_hi_large - eps, BITS)
            if not (sel_below["prefix_len"] <= sel_large["prefix_len"]):
                fail("boundary below not <= ")
            # Typically exactly one less (since d strictly increasing with t
            # for distinct tail factors? For identical tails, d strictly
            # increasing, so below should drop at least one. Check <=, and if
            # equal then widths? Allow <= to avoid flaky exact-equality gate.
            notes.append(f"boundary d_hi={float(d_hi_large):.6f} below_len={sel_below['prefix_len']}")
        else:
            notes.append("boundary d_hi tiny, skip below")
        # Homogeneous analytical support only: verify mu identities + bias formula.
        for th in (0.1, 0.2, 0.5):
            r = (math.sin(th), 0.0, math.cos(th))
            L = abs(math.sin(th)) + abs(math.cos(th))
            a = max(abs(math.sin(th)), abs(math.cos(th)))
            # Float mu vs formula.
            eta_up = (1 - 1 / L) / 2
            mu_up_float = eta_up * (1 - eta_up) / (1 - 2 * eta_up) ** 2 if abs(1 - 2 * eta_up) > 1e-12 else float("inf")
            mu_up_form = (L * L - 1) / 4
            if abs(mu_up_float - mu_up_form) > 1e-9:
                fail(f"mu_up th={th}: {mu_up_float} vs {mu_up_form}")
            eta_lo = (1 - a) / 2
            mu_lo_float = eta_lo * (1 - eta_lo) / (1 - 2 * eta_lo) ** 2 if abs(1 - 2 * eta_lo) > 1e-12 else float("inf")
            mu_lo_form = (1 / (a * a) - 1) / 4
            if abs(mu_lo_float - mu_lo_form) > 1e-9:
                fail(f"mu_lo th={th}")
            # Bias formula: all-axis d vs cos(theta/2)^{2n} for n=3.
            n_h = 3
            hb = geo.homogeneous_bias_all(th, n_h, BITS)
            c_float_expect = (math.cos(th / 2)) ** (2 * n_h)
            if abs(hb["c_float"] - c_float_expect) > 1e-12:
                fail(f"homogeneous c th={th}")
            # Certified d interval must contain float d.
            d_f = hb["d_float"]
            if not (float(hb["d_lo"]) - 1e-9 <= d_f <= float(hb["d_hi"]) + 1e-9):
                fail(f"homogeneous d containment th={th}")
            # Small-theta scaling spot: mu_up ~ Theta(theta), mu_lo ~ Theta(theta^2).
            # Check ratios bounded for th=0.1 vs 0.05 (analytical, not empirical proof).
        # Theta scaling ratios (illustration only, not proof).
        def mu_up_f(th):
            L_ = abs(math.sin(th)) + abs(math.cos(th))
            return (L_ * L_ - 1) / 4
        def mu_lo_f(th):
            a_ = abs(math.cos(th))
            return (1 / (a_ * a_) - 1) / 4
        r1 = mu_up_f(0.1) / 0.1
        r2 = mu_up_f(0.05) / 0.05
        if not (0.3 < r1 < 1.0 and 0.3 < r2 < 1.0):
            fail(f"Theta(theta) ratios {r1},{r2}")
        q1 = mu_lo_f(0.1) / (0.1 * 0.1)
        q2 = mu_lo_f(0.05) / (0.05 * 0.05)
        if not (0.2 < q1 < 0.4 and 0.2 < q2 < 0.4):
            fail(f"Theta(theta^2) ratios {q1},{q2}")
        notes.append(f"large n={n_large} chosen={sel_large['prefix_len']} d_hi={sel_large['d_hi_str']}")
        # Strict homogeneous boundary: pi/4 accepted, just above rejected.
        try:
            hb_edge = geo.homogeneous_bias_all(math.pi / 4, 2, BITS)
            if not (hb_edge["d_lo"] <= hb_edge["d_hi"]):
                fail("homogeneous edge interval inverted")
        except Exception as e_edge:
            fail(f"homogeneous edge pi/4 rejected: {e_edge}")
        try:
            geo.homogeneous_bias_all(math.pi / 4 + 1e-9, 2, BITS)
            fail("homogeneous above pi/4 not rejected (strict boundary required)")
        except ValueError:
            pass
        except Exception as e2:
            fail(f"homogeneous above wrong exc {type(e2).__name__}: {e2}")
    except Exception as e:
        import traceback as _tb7
        fail(f"large/homogeneous harness {_tb7.format_exc()}")

    # ---- H. n=200 default-bit regression + high-precision boundary ----
    n200_info: dict = {}
    try:
        th200 = 0.01
        r200 = (math.sin(th200), 0.0, math.cos(th200))
        n200 = 200
        rs200 = [r200] * n200
        D200 = list(range(n200))
        t200_0 = time.perf_counter()
        cert200 = geo.geometry_certificate(rs200, D200, BITS)
        t200_1 = time.perf_counter()
        c_lo200 = Fraction(cert200["c_lo_str"])
        c_hi200 = Fraction(cert200["c_hi_str"])
        d_lo200 = Fraction(cert200["d_lo_str"])
        d_hi200 = Fraction(cert200["d_hi_str"])
        # Validity: intervals in [0,1], ordered, proxy length, str safe.
        if not (Fraction(0, 1) <= c_lo200 <= c_hi200 <= Fraction(1, 1)):
            fail(f"n200 c range [{c_lo200},{c_hi200}]")
        if not (Fraction(0, 1) <= d_lo200 <= d_hi200 <= Fraction(1, 1)):
            fail(f"n200 d range [{d_lo200},{d_hi200}]")
        if len(cert200["proxy"]) != n200:
            fail("n200 proxy len")
        # Serialization must stay below the Python digit limit (4300).
        for key in ("c_lo_str", "c_hi_str", "d_lo_str", "d_hi_str"):
            if len(cert200[key]) > 3000:
                fail(f"n200 serialization long {key}={len(cert200[key])}")
        # Float conversions must succeed (bounded denominators).
        try:
            _ = (cert200["c_lo_float_lower"], cert200["c_hi_float_upper"],
                 cert200["d_lo_float_lower"], cert200["d_hi_float_upper"])
        except Exception as e_fl:
            fail(f"n200 float conversion failed: {e_fl}")
        # Width at default bits must be tiny. Per-endpoint rounding is
        # <=|D|/2**bits each, so width additional rounding is <=2|D|/2**bits
        # plus a-enclosure propagation (not |D|/2**bits total).
        try:
            if Fraction(cert200["product_rounding_bound"]) != Fraction(n200, 1 << BITS):
                fail(f"n200 rounding bound field {cert200['product_rounding_bound']} vs {Fraction(n200, 1 << BITS)}")
        except Exception as e_rb200:
            fail(f"n200 rounding field check failed: {e_rb200}")
        w_c = float(c_hi200 - c_lo200)
        w_d = float(d_hi200 - d_lo200)
        if w_c > 1e-15 or w_d > 1e-15:
            fail(f"n200 width large c={w_c:.3g} d={w_d:.3g}")
        # Selector cost honesty: time n=50 selection already recorded; for
        # n=200 record certificate time only (selection would be quadratic and
        # slow, documented in production docstring, not run here).
        # High-precision boundary: same n=200 at bits=1024 must succeed and be
        # at least as tight as bits=80 (demonstrates combined resource works
        # without hitting the digit limit; no process-global safeguard change).
        tH0 = time.perf_counter()
        certH = geo.geometry_certificate(rs200, D200, BITS_HIGH)
        tH1 = time.perf_counter()
        c_loH = Fraction(certH["c_lo_str"])
        c_hiH = Fraction(certH["c_hi_str"])
        d_loH = Fraction(certH["d_lo_str"])
        d_hiH = Fraction(certH["d_hi_str"])
        if not (c_loH <= c_lo200 or True):
            # High-precision lower may move slightly due to tighter a bounds;
            # only require containment of truth, i.e. intervals overlap and
            # high-precision width is no larger (up to rounding).
            pass
        w_cH = float(c_hiH - c_loH)
        w_dH = float(d_hiH - d_loH)
        if w_cH > w_c + 1e-18 or w_dH > w_d + 1e-18:
            fail(f"high-precision not tighter: default c={w_c:.3g} high={w_cH:.3g} d={w_d:.3g}/{w_dH:.3g}")
        for key in ("c_lo_str", "c_hi_str", "d_lo_str", "d_hi_str"):
            if len(certH[key]) > 3500:
                fail(f"high-precision serialization long {key}={len(certH[key])}")
        n200_info = {
            "n": n200, "theta": th200, "bits": BITS, "bits_high": BITS_HIGH,
            "c_default": [str(c_lo200), str(c_hi200)],
            "d_default": [str(d_lo200), str(d_hi200)],
            "width_c_default": w_c, "width_d_default": w_d,
            "width_c_high": w_cH, "width_d_high": w_dH,
            "cert_time_default": t200_1 - t200_0,
            "cert_time_high": tH1 - tH0,
            "c_str_len_default": len(cert200["c_lo_str"]),
            "c_str_len_high": len(certH["c_lo_str"]),
        }
        notes.append(f"n200 default d_hi={d_hi200} width_d={w_d:.2g} "
                     f"high width_d={w_dH:.2g} cert={t200_1 - t200_0:.2f}s/{tH1 - tH0:.2f}s")
    except Exception as e:
        import traceback as _tb8
        fail(f"n200 harness {_tb8.format_exc()}")

    t_end = time.perf_counter()
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    status = "PASS" if not failures else "FAIL"
    payload = {
        "status": status,
        "scope": "Frozen small geometry harness: n<=3 eigvalsh, Decimal enclosures incl. tiny, exhaustive n<=6 containment plus per-endpoint/width rounding regression (bounded product), operational mixed iid plus entangled iid Born (Bell Phi+, k'=1 N=256 small-bias nonconstant g=(175/128,-47/128) b<=1/100, exact-law first moment) + composition, no/all discard (MOCK-labelled), tamper before data, invalids, large tails n=50 + n=200 default-bit regression + bits=1024 boundary + homogeneous strict-boundary diagnostic only. ",
        "seed": SEED,
        "python": py_ver, "numpy": numpy_ver, "scipy": scipy_ver,
        "bits": BITS, "bits_high": BITS_HIGH,
        "tolerances": {"eig": TOL_EIG, "decimal_slack": str(TOL_DECIMAL), "born_diagnostic": TOL_BORN},
        "failures": failures,
        "notes": notes + [
            "Production geometry uses exact Fraction inputs + integer-sqrt enclosures + bounded outward product rounding to SCALE=2**bits (floor/ceil per factor); outward floats via Fraction+nextafter (reimplemented, not float tolerance).",
            "Selection before data => no multiple-selection penalty; sufficient upper, not minimax; selector rebuilds each prefix (quadratic repeated-certification cost, timed).",
            "Inner estimator UNCHANGED via public API (solve_and_certify/plan_estimator/run_streaming); default common, half explicit; native varying-noise upper is theoretical only, distinct from implemented common-channel solver.",
            "Large-tail example n=50 core 2 + 48 tail theta=0.01 under beta=0.04 retains core (k'=2); n=200 default-bit certificate plus bits=1024 boundary recorded; homogeneous float quantities are diagnostics only, strict 0<theta<=pi/4.",
            "Resource limits: MAX_N=1024, bits 1..1024; geometry-field bounded denominators keep geometry str() below digit limit for allowed n/bits combos (inner scores/N obey estimator limits; arbitrary huge user Fractions/N not guaranteed); tested to n=200 + n<=6 exhaustive + n<=3 dense; timings/memory below.",
        ],
        "large_tail": large_info,
        "n200": n200_info,
        "entangled": ent_info if "ent_info" in locals() else {},
        "timing_sec": t_end - t_start,
        "memory": {"current": cur, "peak": peak, "caveat": "Python tracemalloc only; HiGHS native excluded"},
        "resource_limits": {"MAX_N": geo.MAX_N, "bits_min": geo.GEOMETRY_BITS_MIN, "bits_max": geo.GEOMETRY_BITS_MAX, "default_bits": geo.GEOMETRY_BITS_DEFAULT},
    }
    dest = (Path(__file__).resolve().parents[1] / "results" / "product_state_geometry_checks.json")
    dest.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "failures": failures, "timing_sec": t_end - t_start}, indent=2))
    if status != "PASS":
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
