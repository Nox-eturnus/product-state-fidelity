"""Bounded certified scalable estimator for product-state fidelity.

Uses SciPy HiGHS linear programming with rational certification.
Validated at k<=6 against the exact vertex-enumeration reference in
product_state_lp_prototype.py. See finite_size_evaluation.py for benchmarks.

Target convention (pure directions, explicit):
  Each input triple ``r`` is a finite binary-rational direction. The
  represented pure target factor is the mathematically normalized direction
  ``r_hat = r / ||r||_2`` exactly (real division, generally irrational
  coordinates). Ideal axis-choice, degradation-coin, and Born probabilities
  in the certificate refer to these exact ``r_hat`` directions. Numerical
  sampling values are computed consistently from normalized quantities
  (``p_i = |r_hat_i| / ||r_hat||_1``, ``L_hat = ||r||_1 / ||r||_2``,
  ``m = L_hat/sqrt(3)`` or ``L_hat/2`` as float approximations). Finite
  float rounding of those probabilities and finite RNG/control errors remain
  OUTSIDE the certificate; no hardware guarantee is claimed. Inputs must
  satisfy ``| ||r||_2 - 1 | <= 1e-9`` (float norm, tolerance only gates which
  directions are accepted, not purity itself); zero vectors rejected. The
  represented normalized target ``rs_hat`` is stored explicitly in the plan;
  stored/displayed ``rs_hat`` floats approximate exact ``r/||r||``.
  Pauli classification uses original exact signed axes on the RAW inputs
  (two coords exactly ``0.0``, third exactly ``+1.0``/``-1.0``); all other
  accepted vectors (including ``(0,0,1.0000000005)`` whose direction is Pauli
  but raw is not exact) are treated as noisy (safe side, still valid, only
  less efficient). Near-axis genuinely non-axis directions are never declared
  stabilizers.

Model (heterogeneous, per docs/REPRODUCIBILITY.md):
  target = tensor_j tau^j_+ with tau^j_+ = (I + R^j)/2, R^j = r_hat^j.(X,Y,Z),
  each r_hat^j unit (pure). k = #non-Pauli factors, n-k exactly Pauli.
  Unknown arbitrary i.i.d. mixed sigma (worst case). Per copy: product Pauli
  basis, one measurement per qubit, full n outcome bits. Across-copy and
  within-copy adaptivity allowed in lower bounds; upper uses nonadaptive
  weighted sampling plus exact stabilizer measurement.

Two explicit channels (no silent downgrade):
  - "common" (DEFAULT, theorem): degrade each noisy sign-corrected bit
    (signal 1/L_hat_j) with independent coin mean L_hat_j/sqrt(3) to common
    weakest signal 1/sqrt(3), i.e. BSC eta_common = (1-1/sqrt(3))/2
    (irrational). Ideal real axis/randomization probabilities assumed;
    hardware/calibration errors excluded from certificate.
  - "degraded_half" (explicit rational): coin mean L_hat_j/2 gives signal 1/2,
    eta=1/4 rational, for exact prototype comparison. Intentionally noisier,
    same exponent shape, exactly rational W. Must be requested by name;
    never silently substituted for "common".

LP (PSW Eq8 Hamming-weight form): variables g_0..g_k (free), b>=0, G>=0;
  constraints -b <= sum_w W[w,h] g_w - delta_{h0} <= b for all h, i.e.
  ``|W^T g - e0|_inf <= b`` where ``W[w,h]`` rows observed weight w, cols
  latent weight h, ``e0 = (1,0,...,0)``; ``-G <= g_w <= G``; objective
  ``b + G/sqrt(N)``. Solved in float64 with scipy.optimize.linprog
  (method='highs'). Float feasibility is NOT a certificate and exact
  optimality is NOT claimed; independent exact certificates are computed
  from the actually executed rounded rationals (see private low-level
  helpers ``_certify_*``). k=0 uses exact Pauli-indicator path (g=[1],
  b=0, G=1), no solver; any other k=0 score is rejected.

Certificates (fresh from executed rationals at selected channel/k, never
  caller b/G, never re-rounding user Fractions):
  - half: exact Fraction W at eta=1/4; raw binary-float-as-Fraction
    certificate, rounded denominator-256 certificate, max rounding
    difference, fresh G. ``b_round <= b_raw + maxdiff`` (columns sum to 1).
  - common: rigorous rational enclosure of sqrt(3) with >=80 bits via
    integer square root (SCALE=2**bits, s=isqrt(3*SCALE^2),
    lo=s/SCALE, hi=(s+1)/SCALE, so lo<=sqrt3<hi, width 2**-bits).
    eta=(1-sqrt3/3)/2 = 1/2-sqrt3/6, so eta in [1/2-hi/6, 1/2-lo/6];
    eta_mid=(eta_lo+eta_hi)/2 rational,
    max_eta_error=(eta_hi-eta_lo)/2 = 2**-bits/12 with this enclosure.
    Exact Fraction W_mid at midpoint; raw/rounded certificates at W_mid;
    coupling 2*G*k*max_eta_error certifies bias for true eta. Proof: k
    independent BSCs TV<=k|eta-eta'| (union/coupling), L1=2TV, Hamming
    coarse-graining cannot increase L1, so |E_true-E_mid|<=G*L1<=2Gk|..|.
    Coefficient-rounding and channel-enclosure errors kept distinct.
  ``plan_estimator`` freshly recomputes b/G from the supplied Fractions at
  the selected channel/k and rejects understated caller claims and
  wrong-channel reuse; zero-score and non-[1] k=0 scores rejected.
  Low-level helpers taking explicit W/enclosure are private (underscore)
  and do NOT certify channel identity by themselves.

Confidence (rigorous outward, no arbitrary padding):
  Exact Fraction rational ops throughout; ``ln(2/delta)`` upper via Decimal
  correctly-rounded division+ln at documented precision (prec=50) then
  ``next_plus`` on monotone upper argument, converted exactly to Fraction;
  final ``sqrt`` upper via integer-sqrt rational enclosure
  (``p/q`` -> ``(isqrt(p*q)+1)/q``, exact when perfect square); float
  display is outward via ``Fraction(float)`` comparison plus
  ``math.nextafter`` (lower downward, upper upward). Radius rational upper
  ``R = b + G*sqrt(2*ln/N)`` (``G=0`` short-circuits to ``b``, avoiding
  ``0*inf``), Bernoulli ``sqrt(ln/(2N))`` similarly; interval endpoints
  from exact mean ``m`` as ``clip01(m-R)``, ``clip01(m+R)`` rationally then
  outward (ordered in [0,1] even when raw interval is disjoint, e.g. mean
  outside [0,1] due to ``G>1``). Both rational exact and float display are
  returned/stored. Sufficient copies use exact ceil of upper rational
  ``2G^2 ln/(eps-b)^2`` (Bernoulli ``ln/(2(eps-b)^2)``), exact ``eps-b``
  Fraction comparison, explicit ``G=0`` (need 0), subnormal-delta, large-N,
  and near-eps handling; ``ValueError`` for documented resource bounds
  before overflow (float conversion to inf handled as conservative inf
  where valid, never NaN). Surrogate ``b+G/sqrt(N)`` exact Fraction for
  perfect-square N, otherwise rigorous upper ``b+G*SCALE/isqrt(N*SCALE^2)``
  with ``is_exact=False`` label plus symbolic ``"b+G/sqrt(N)"``.
  Independent tests verify with higher precision/rational inequalities.

Frozen pre-data design (immutable):
  ``EstimatorPlan`` is a frozen dataclass of tuples/Fractions (deep copies
  before any callback). ``N`` positive int (not bool, any ordinary positive
  int such as 3,17,359 supported, not only squares), ``0<delta<1``,
  ``channel`` (default "common"), score ``g`` fixed before ``N`` fresh
  i.i.d. copies. ``run_streaming`` revalidates the complete plan at entry
  BEFORE any callback/RNG via one canonical ``plan_estimator``
  reconstruction (fresh b/G exactly once, never per-copy, timed as
  ``revalidate_time_sec``), rejecting underclaimed certs and inconsistent
  derived n/k/rs_hat/K/S/L/axes/signs/channel/cert metadata and using only
  the canonical snapshot; it never re-solves/rebuilds per copy (O(n+k)
  streaming). Mutating the original ``rs``/``g`` lists or
  attempting ``plan.N=...`` cannot alter collection. Hoeffding radius
  separate from surrogate; k=0 Bernoulli; clip estimate to [0,1]; ordered
  interval covers truth under stated i.i.d.+ideal assumptions. No
  data-dependent stopping/reuse. Strict validation (finite, dims,
  bool-vs-int with original types checked before casts, axes X/Y/Z with
  stabilizer-axis match, outcomes +-1 ints,
  counts, empty, k mismatch, zero-prob noisy axes rejected, RNG draws must
  satisfy 0<=u<1 with 1.0 rejected, W shape/finiteness/stochasticity checked
  before any k=0 shortcut).

Operational: sample each copy's noisy axes with ``|r_hat_i|/L_hat``, measure
  once each via supplied full-copy callback returning correlated outcomes
  (never simulate qubits independently), sign-correct, independently
  degrade, compute ``H=g[w]*F``, accumulate with ALL N denominator.
  Streaming O(n+k) memory aside from LP; exact Fraction accumulation.
  Reproducible RNG injection (random.Random). Full local-Pauli model.

Practical example (see demo and docs/REPRODUCIBILITY.md):
  ``rs=[(1/sqrt3,)*3, (0,0,1)]`` (k=1); ``rec=solve_and_certify(1,16)``
  (common default); ``plan=plan_estimator(rs,16,1/3,"common",g)``
  (fresh b/G, immutable); ``run_streaming(plan, measure_fn, rng)`` gives
  ``estimate`` + ordered ``interval`` covering fidelity. Limits: ideal
  probabilities only, finite-N Hoeffding (delta failure), no scaling claim.

Resources reported separately: N, nN, channel/cert/solve/streaming times,
  tracemalloc Python memory with native-allocation caveat, dense byte
  counts, channel cond, coefficient bit lengths. No practical-size or
  polynomial-time solver guarantee inferred from k<=6.

Run demo: python code/product_state_estimator.py --demo
Requires NumPy/SciPy in .venv (see requirements-estimator.txt).
Exits nonzero on any failure.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import decimal
import math
import platform
import random
import sys
import time
import tracemalloc
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

# --- constants ------------------------------------------------------------

CHANNEL_COMMON = "common"
CHANNEL_HALF = "degraded_half"
ALLOWED_CHANNELS = (CHANNEL_COMMON, CHANNEL_HALF)
DEFAULT_CHANNEL = CHANNEL_COMMON

ROUND_DEN = 256
ENCLOSURE_BITS = 80
SURROGATE_BITS = 80
LN_PREC = 50
EPS0 = Fraction(1, 20)
DELTA_DEFAULT = 1.0 / 3.0
PURITY_TOL = 1e-9
HIGHS_METHOD = "highs"
HIGHS_OPTIONS = {"presolve": True}

PAULI_AXES = ("X", "Y", "Z")


# --- small validation helpers ---------------------------------------------

def _is_bool(x) -> bool:
    return isinstance(x, bool)


def _require_int_not_bool(name: str, v) -> int:
    if _is_bool(v) or not isinstance(v, int):
        raise ValueError(f"{name} must be int (bool rejected), got {type(v).__name__}={v!r}")
    return v


def _require_positive_int(name: str, v) -> int:
    v = _require_int_not_bool(name, v)
    if v <= 0:
        raise ValueError(f"{name} must be positive, got {v}")
    return v


def _require_nonneg_int(name: str, v) -> int:
    v = _require_int_not_bool(name, v)
    if v < 0:
        raise ValueError(f"{name} must be nonnegative, got {v}")
    return v


def _require_finite_float(name: str, v) -> float:
    if _is_bool(v) or not isinstance(v, (int, float)):
        raise ValueError(f"{name} must be finite number, got {type(v).__name__}={v!r}")
    f = float(v)
    if not math.isfinite(f):
        raise ValueError(f"{name} must be finite, got {v!r}")
    return f


def _require_delta(delta) -> float:
    if _is_bool(delta) or not isinstance(delta, (int, float)):
        raise ValueError(f"delta must be number in (0,1), got {delta!r}")
    f = float(delta)
    if not math.isfinite(f) or not (0.0 < f < 1.0):
        raise ValueError(f"delta must satisfy 0<delta<1, got {delta!r}")
    # Reject subnormal-underflow to zero already covered; subnormal positives
    # in (0,1) are accepted (rigorous ln handles them, see _ln_upper_frac).
    return f


def _require_eps(eps) -> float:
    if _is_bool(eps) or not isinstance(eps, (int, float)):
        raise ValueError(f"eps must be number >0, got {eps!r}")
    f = float(eps)
    if not math.isfinite(f) or not (f > 0.0):
        raise ValueError(f"eps must be finite >0, got {eps!r}")
    return f


def _require_channel(ch: str) -> str:
    if not isinstance(ch, str) or ch not in ALLOWED_CHANNELS:
        raise ValueError(f"channel must be one of {ALLOWED_CHANNELS}, got {ch!r}")
    return ch


def clipped_estimate(mean) -> float:
    """Clip raw mean to [0,1]. Rejects nonfinite/bool. Deterministic."""
    if _is_bool(mean) or not isinstance(mean, (int, float, Fraction)):
        raise ValueError(f"mean must be number, got {mean!r}")
    f = float(mean)
    if not math.isfinite(f):
        raise ValueError(f"mean must be finite, got {mean!r}")
    if f < 0.0:
        return 0.0
    if f > 1.0:
        return 1.0
    return f


def estimate_from_counts(counts, g):
    """Clipped weighted average sum_w (counts[w]/total) g_w.

    counts: nonempty list of nonnegative ints (bool rejected), total>0.
    g: same length, each finite number (int/float/Fraction, bool rejected).
    Exact rational mean via Fractions (floats interpreted exactly as binary),
    then float + clip. Raises ValueError on mismatch/empty/negative/bool/
    non-integer/nonfinite/non-positive-total.
    """
    if not isinstance(counts, (list, tuple)) or not isinstance(g, (list, tuple)):
        raise ValueError("counts and g must be lists/tuples")
    if len(counts) != len(g):
        raise ValueError(f"counts/g length mismatch {len(counts)} vs {len(g)}")
    if len(counts) == 0:
        raise ValueError("empty counts")
    for c in counts:
        if _is_bool(c) or not isinstance(c, int):
            raise ValueError(f"counts must be nonnegative ints (bool rejected), got {c!r}")
        if c < 0:
            raise ValueError(f"negative count {c}")
    total = sum(counts)
    if total <= 0:
        raise ValueError("non-positive total")
    # validate g finite
    g_fracs = []
    for v in g:
        if _is_bool(v) or not isinstance(v, (int, float, Fraction)):
            raise ValueError(f"g entries must be finite numbers, got {v!r}")
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"g entry nonfinite {v!r}")
        if isinstance(v, Fraction):
            if v.denominator == 0:
                raise ValueError("zero denominator")
            g_fracs.append(v)
        elif isinstance(v, int):
            g_fracs.append(Fraction(v, 1))
        else:
            g_fracs.append(Fraction(v))  # exact binary
    num = sum(Fraction(c, 1) * gv for c, gv in zip(counts, g_fracs))
    mean_exact = num / total
    return clipped_estimate(float(mean_exact))


# --- target handling (pure directions, explicit) ---------------------------

def _validate_bloch_vector(r) -> tuple[float, float, float]:
    """Validate one raw direction triple, return raw floats.

    Each coord must be finite int/float (bool rejected). The triple must be
    nonzero and satisfy ``| ||r||_2 - 1 | <= PURITY_TOL`` (float norm). The
    tolerance only gates which directions are accepted; purity itself comes
    from interpreting the triple as the direction ``r/||r||_2`` exactly
    (see ``_normalized_hat``). Zero vectors and far-from-unit vectors are
    rejected. ``-0.0`` is treated as ``0.0`` (``==``).
    """
    if not isinstance(r, (list, tuple)) or len(r) != 3:
        raise ValueError(f"Bloch vector must be length-3 sequence, got {r!r}")
    out = []
    for v in r:
        if _is_bool(v) or not isinstance(v, (int, float)):
            raise ValueError(f"Bloch coords must be numbers (bool rejected), got {v!r}")
        f = float(v)
        if not math.isfinite(f):
            raise ValueError(f"Bloch coord nonfinite {v!r}")
        out.append(f)
    n2 = out[0] ** 2 + out[1] ** 2 + out[2] ** 2
    if not math.isfinite(n2) or n2 <= 0.0:
        raise ValueError(f"Bloch vector must be nonzero direction, got {r!r}")
    norm = math.sqrt(n2)
    if not math.isfinite(norm):
        raise ValueError(f"Bloch norm nonfinite from {r!r}")
    if abs(norm - 1.0) > PURITY_TOL:
        raise ValueError(
            f"Bloch direction must be near-unit (|r|_2=1 within {PURITY_TOL}); "
            f"got norm {norm} from {r!r}. Represented target would be r/||r||, "
            f"but far-from-unit inputs are rejected to avoid silent rescaling."
        )
    return (out[0], out[1], out[2])


def _normalized_hat(r_raw: tuple[float, float, float]) -> tuple[float, float, float]:
    """Float approximation to the exact represented direction r/||r||_2.

    The ideal certificate refers to the exact real ``r_hat`` (binary-rational
    ``r`` divided by its exact Euclidean norm). This returns float
    ``r_i/norm`` with ``norm=sqrt(sum r_i^2)`` as the numerical proxy used
    for sampling/degradation. Finite rounding of these proxies is OUTSIDE the
    certificate (documented, no hardware guarantee).
    """
    n2 = r_raw[0] ** 2 + r_raw[1] ** 2 + r_raw[2] ** 2
    norm = math.sqrt(n2)
    # norm>0 and near 1 by validation; guard anyway.
    if not math.isfinite(norm) or norm == 0.0:
        raise ValueError(f"cannot normalize {r_raw!r}")
    return (r_raw[0] / norm, r_raw[1] / norm, r_raw[2] / norm)


def _L_hat_of_raw(r_raw: tuple[float, float, float]) -> float:
    """Normalized ``L1`` ``||r_hat||_1 = ||r||_1/||r||_2`` as float proxy."""
    hat = _normalized_hat(r_raw)
    return abs(hat[0]) + abs(hat[1]) + abs(hat[2])


def validate_target(rs) -> list[tuple[float, float, float]]:
    """Validate heterogeneous pure-direction target list.

    Returns list of RAW 3-tuples floats (as supplied, validated). The
    represented pure target is ``rs_hat = [r/||r|| for r in rs]`` exactly;
    see ``normalized_target`` and ``EstimatorPlan.rs_hat``. Rejects empty,
    wrong dims, bool, nonfinite, zero, far-from-unit. Tolerance ``PURITY_TOL``
    documented; ideal probabilities refer to normalized directions.
    """
    if not isinstance(rs, (list, tuple)):
        raise ValueError(f"target must be list of Bloch vectors, got {type(rs).__name__}")
    if len(rs) == 0:
        raise ValueError("empty target (n=0 not allowed; need >=1 qubit)")
    return [_validate_bloch_vector(r) for r in rs]


def normalized_target(rs) -> list[tuple[float, float, float]]:
    """Return float proxies for the represented pure directions ``r/||r||``.

    Validates via ``validate_target`` first. Each output is
    ``_normalized_hat(raw)``. Ideal probabilities refer to the exact real
    normalized directions; these floats are the numerical proxies (rounding
    outside certificate).
    """
    raws = validate_target(rs)
    return [_normalized_hat(r) for r in raws]


def is_pauli_exact(r) -> tuple[bool, str | None, int | None]:
    """Exact Pauli-eigenstate test on RAW inputs: ONLY signed coordinate axes.

    Returns (is_pauli, axis, want). axis in X/Y/Z, want in {+1,-1} is the
    target outcome for that axis. Exact float equality (two coords exactly
    ``0.0``, third exactly ``+1.0``/``-1.0``); near-axis floats (including
    ``(0,0,1.0000000005)``) are NOT accepted as Pauli and are treated as
    noisy (safe side, still valid, only less efficient). ``r`` must already
    be a validated raw 3-tuple. Note ``-0.0 == 0.0`` is treated as zero.
    """
    rx, ry, rz = float(r[0]), float(r[1]), float(r[2])
    coords = (rx, ry, rz)
    for idx, ax in enumerate(PAULI_AXES):
        v = coords[idx]
        others = [coords[j] for j in range(3) if j != idx]
        if v == 1.0 and others[0] == 0.0 and others[1] == 0.0:
            return (True, ax, 1)
        if v == -1.0 and others[0] == 0.0 and others[1] == 0.0:
            return (True, ax, -1)
    return (False, None, None)


def L_of(r) -> float:
    """Raw ``L1`` ``|r|_1`` (legacy helper; prefer ``_L_hat_of_raw`` for sampling).

    Kept for diagnostics/comparison. Sampling and degradation use normalized
    ``L_hat`` consistently (see ``classify_target`` ``L_hat_list``).
    """
    return abs(float(r[0])) + abs(float(r[1])) + abs(float(r[2]))


def classify_target(rs) -> dict:
    """Split validated target into noisy K and stabilizer S.

    Returns dict with ``rs`` (raw list), ``rs_hat`` (normalized float proxies,
    explicit represented directions), ``n``, ``k``, ``K_idx``, ``S_idx``,
    ``L_list`` (raw, legacy), ``L_hat_list`` (normalized, used for sampling),
    ``stab_axis``, ``stab_want``. Classification uses raw exact axes only.
    """
    vrs = validate_target(rs)
    hats = [_normalized_hat(r) for r in vrs]
    n = len(vrs)
    K_idx: list[int] = []
    S_idx: list[int] = []
    L_list: list[float] = []
    L_hat_list: list[float] = []
    stab_axis: list[str | None] = []
    stab_want: list[int | None] = []
    for j, (r, h) in enumerate(zip(vrs, hats)):
        L_list.append(L_of(r))
        L_hat_list.append(abs(h[0]) + abs(h[1]) + abs(h[2]))
        is_p, ax, want = is_pauli_exact(r)
        if is_p:
            S_idx.append(j)
            stab_axis.append(ax)
            stab_want.append(want)
        else:
            K_idx.append(j)
            stab_axis.append(None)
            stab_want.append(None)
    return {
        "rs": vrs, "rs_hat": hats, "n": n, "k": len(K_idx), "K_idx": K_idx,
        "S_idx": S_idx, "L_list": L_list, "L_hat_list": L_hat_list,
        "stab_axis": stab_axis, "stab_want": stab_want,
    }


# --- rigorous enclosure ---------------------------------------------------

def sqrt3_enclosure(bits: int = ENCLOSURE_BITS) -> tuple[Fraction, Fraction]:
    """Rigorous [lo,hi] enclosing sqrt(3) with width 2**-bits.

    bits must be positive int (not bool), >=80 recommended for production.
    SCALE=2**bits, s=isqrt(3*SCALE^2), lo=s/SCALE, hi=(s+1)/SCALE.
    Proof: s<=sqrt(3*SCALE^2)<s+1 => s/SCALE<=sqrt3<(s+1)/SCALE.
    Width hi-lo = 1/SCALE = 2**-bits. Verified by integer inequality below.
    """
    if _is_bool(bits) or not isinstance(bits, int) or bits <= 0:
        raise ValueError(f"bits must be positive int, got {bits!r}")
    scale = 1 << bits
    s = math.isqrt(3 * scale * scale)
    # Integer proof of enclosure (isqrt definition); raises if violated.
    if not (s * s <= 3 * scale * scale < (s + 1) * (s + 1)):
        raise RuntimeError("isqrt enclosure violated (unreachable)")
    lo = Fraction(s, scale)
    hi = Fraction(s + 1, scale)
    # Exact rational check lo*lo <= 3 < hi*hi (strict upper).
    if not (lo * lo <= 3 < hi * hi):
        raise RuntimeError("sqrt3 rational enclosure check failed")
    return (lo, hi)


def eta_interval_common(bits: int = ENCLOSURE_BITS) -> dict:
    """Enclose eta_common=(1-1/sqrt3)/2 = 1/2 - sqrt3/6.

    Returns dict with sqrt3_lo/hi, eta_lo/hi/mid/max_eta_error as Fractions
    plus bits. eta_lo=1/2-hi3/6, eta_hi=1/2-lo3/6,
    eta_mid=(eta_lo+eta_hi)/2, max_eta_error=(eta_hi-eta_lo)/2.
    With the sqrt3 enclosure above (width 2**-bits),
    eta width = (hi3-lo3)/6 = 2**-bits/6, so
    max_eta_error = 2**-bits/12 (at 80 bits ~=6.9e-26). Bounds
    |eta_true-eta_mid| <= max_eta_error. sqrt3 vs eta endpoints kept
    distinct (see keys).
    """
    lo3, hi3 = sqrt3_enclosure(bits)
    eta_lo = Fraction(1, 2) - hi3 / 6
    eta_hi = Fraction(1, 2) - lo3 / 6
    if not (eta_lo < eta_hi):
        raise RuntimeError("eta interval empty (unreachable)")
    mid = (eta_lo + eta_hi) / 2
    max_err = (eta_hi - eta_lo) / 2
    # Documented value check: max_err == 1/(12*2**bits) exactly.
    if max_err != Fraction(1, 12 * (1 << bits)):
        raise RuntimeError(f"max_eta_error != 2^-bits/12: {max_err}")
    return {
        "sqrt3_lo": lo3, "sqrt3_hi": hi3, "bits": bits,
        "eta_lo": eta_lo, "eta_hi": eta_hi, "eta_mid": mid,
        "max_eta_error": max_err,
    }


# --- rigorous confidence arithmetic (no arbitrary padding) ------------------
# Design: exact Fraction ops; ln upper via Decimal correctly-rounded ops at
# prec=LN_PREC then next_plus on monotone upper argument; sqrt upper via
# integer-sqrt rational enclosure; float display outward via exact
# Fraction(float) comparison plus math.nextafter. All bounds are upper
# (conservative) by construction; proofs in docstrings.

def _fraction_to_decimal_upper(frac: Fraction, prec: int = LN_PREC) -> decimal.Decimal:
    """Smallest-documented Decimal >= positive Fraction ``frac``.

    Uses correctly-rounded Decimal division at ``prec`` then ``next_plus``.
    Since division is correctly rounded, true quotient is within 0.5 ulp of
    the rounded value, so one ulp up is a strict upper (for finite positive
    values within Emax). ``prec`` positive int. Raises on non-positive/non-
    finite inputs or Decimal overflow.
    """
    if not isinstance(frac, Fraction):
        raise ValueError("frac must be Fraction")
    if frac <= 0:
        raise ValueError(f"frac must be positive, got {frac}")
    if _is_bool(prec) or not isinstance(prec, int) or prec <= 0:
        raise ValueError(f"prec must be positive int, got {prec!r}")
    num, den = frac.numerator, frac.denominator
    ctx = decimal.Context(prec=prec, Emax=999999, Emin=-999999,
                          traps=[decimal.InvalidOperation, decimal.DivisionByZero,
                                 decimal.Overflow])
    with decimal.localcontext(ctx):
        d = decimal.Decimal(num) / decimal.Decimal(den)
        if not d.is_finite():
            raise ValueError(f"Decimal division nonfinite for {frac}")
        up = d.next_plus()
        if not up.is_finite():
            raise ValueError("Decimal upper overflowed to Infinity")
        # Monotone guard: up must be >= d (next_plus increases for positive).
        if up < d:
            raise RuntimeError("Decimal next_plus decreased (unreachable)")
        return up


def _ln_upper_frac(arg: Fraction, prec: int = LN_PREC) -> Fraction:
    """Rigorous Fraction upper bound for ``ln(arg)`` with ``arg > 1``.

    Steps: Decimal upper ``a_up >= arg`` via ``_fraction_to_decimal_upper``
    (monotone: ``ln(a_up) >= ln(arg)``), then correctly-rounded
    ``Decimal.ln`` at ``prec`` plus ``next_plus`` gives ``Decimal >=
    ln(a_up) >= ln(arg)``, converted exactly to Fraction (decimal is base-10
    exact). ``prec`` documented (default 50, ~1e-50 relative). Relies on
    Decimal division/ln being correctly rounded per Python docs; one ulp
    dominates that rounding, so the result is a strict upper. Raises
    ValueError for ``arg <= 1`` (use only for ``2/delta > 2``).
    """
    if not isinstance(arg, Fraction):
        raise ValueError("arg must be Fraction")
    if arg <= 1:
        raise ValueError(f"ln upper requires arg>1, got {arg}")
    if _is_bool(prec) or not isinstance(prec, int) or prec <= 0:
        raise ValueError(f"prec must be positive int, got {prec!r}")
    a_up = _fraction_to_decimal_upper(arg, prec)
    ctx = decimal.Context(prec=prec, Emax=999999, Emin=-999999,
                          traps=[decimal.InvalidOperation, decimal.DivisionByZero,
                                 decimal.Overflow])
    with decimal.localcontext(ctx):
        # a_up > 1, ln defined and positive.
        if a_up <= 1:
            raise RuntimeError("Decimal upper <=1 (unreachable for arg>1)")
        ell = a_up.ln()
        if not ell.is_finite():
            raise ValueError("Decimal ln nonfinite")
        up = ell.next_plus()
        if not up.is_finite():
            raise ValueError("Decimal ln upper overflowed")
        if up < ell:
            raise RuntimeError("ln next_plus decreased (unreachable)")
        # Exact base-10 to Fraction conversion.
        return Fraction(up)


def _sqrt_upper_frac(s: Fraction) -> Fraction:
    """Rigorous Fraction upper bound for ``sqrt(s)`` with ``s >= 0``.

    Write ``s = p/q`` with ``p >= 0, q > 0`` (Fraction normalized). Then
    ``sqrt(s) = sqrt(p*q)/q``. Let ``t = isqrt(p*q)``; then
    ``t <= sqrt(p*q) < t+1``, so ``t/q <= sqrt(s) < (t+1)/q``. Return exact
    ``t/q`` when ``t*t == p*q`` (perfect square), else strict upper
    ``(t+1)/q``. Pure integer arithmetic, no float. Raises for ``s < 0``.
    """
    if not isinstance(s, Fraction):
        raise ValueError("s must be Fraction")
    if s < 0:
        raise ValueError(f"sqrt of negative {s}")
    if s == 0:
        return Fraction(0, 1)
    p, q = s.numerator, s.denominator
    # s>0 => p>0, q>0 (Fraction normalized denominator positive).
    if p <= 0 or q <= 0:
        raise RuntimeError("unexpected Fraction sign (unreachable)")
    t = math.isqrt(p * q)
    if t * t == p * q:
        return Fraction(t, q)
    return Fraction(t + 1, q)


def _float_upper(q: Fraction) -> float:
    """Smallest binary64 float ``>= q`` (``inf`` if ``q`` exceeds max float).

    Compares via exact ``Fraction(float)`` and steps with
    ``math.nextafter``. At most a few steps (conversion error <1 ulp), with
    a bounded loop. Never returns NaN for finite ``q``. For ``q`` larger
    than max float, returns ``inf`` (conservative upper). For very negative
    ``q``, returns the tightest float still ``>= q`` (may be ``-max``).
    """
    if not isinstance(q, Fraction):
        raise ValueError("q must be Fraction")
    try:
        f = float(q)
    except OverflowError:
        # Huge magnitude beyond float range.
        if q > 0:
            return math.inf
        # q very negative: smallest float >= q is -max (since -inf < q).
        return -sys.float_info.max
    if math.isnan(f):
        raise RuntimeError("float conversion gave NaN (unreachable for finite Fraction)")
    if math.isinf(f):
        if f > 0:
            return math.inf
        # -inf < any finite q; tightest upper is -max.
        return -sys.float_info.max
    # Finite f: adjust outward.
    # Fast path: exact comparison.
    fq = Fraction(f)
    if fq >= q:
        # Tighten downward while still >= q (at most few iterations).
        for _ in range(10):
            f_down = math.nextafter(f, -math.inf)
            if math.isinf(f_down):
                break
            try:
                fq_down = Fraction(f_down)
            except OverflowError:
                break
            if fq_down >= q:
                f = f_down
            else:
                break
        return f
    # fq < q: step up until >= q.
    for _ in range(10):
        f_up = math.nextafter(f, math.inf)
        if math.isinf(f_up):
            return math.inf
        try:
            if Fraction(f_up) >= q:
                return f_up
        except OverflowError:
            return math.inf
        f = f_up
    # Fallback (should not happen in ≤10 steps for <1ulp error): keep stepping.
    for _ in range(1000):
        f_up = math.nextafter(f, math.inf)
        if math.isinf(f_up):
            return math.inf
        if Fraction(f_up) >= q:
            return f_up
        f = f_up
    raise RuntimeError("float upper did not converge (unreachable)")


def _float_lower(q: Fraction) -> float:
    """Largest binary64 float ``<= q`` (``-inf`` if ``q`` below min float).

    Mirror of ``_float_upper``. For ``q`` smaller than ``-max``, returns
    ``-inf`` (conservative lower). Never NaN.
    """
    if not isinstance(q, Fraction):
        raise ValueError("q must be Fraction")
    try:
        f = float(q)
    except OverflowError:
        if q < 0:
            return -math.inf
        return sys.float_info.max
    if math.isnan(f):
        raise RuntimeError("float conversion gave NaN (unreachable)")
    if math.isinf(f):
        if f < 0:
            return -math.inf
        return sys.float_info.max
    fq = Fraction(f)
    if fq <= q:
        for _ in range(10):
            f_up = math.nextafter(f, math.inf)
            if math.isinf(f_up):
                break
            try:
                if Fraction(f_up) <= q:
                    f = f_up
                else:
                    break
            except OverflowError:
                break
        return f
    for _ in range(10):
        f_down = math.nextafter(f, -math.inf)
        if math.isinf(f_down):
            return -math.inf
        try:
            if Fraction(f_down) <= q:
                return f_down
        except OverflowError:
            return -math.inf
        f = f_down
    for _ in range(1000):
        f_down = math.nextafter(f, -math.inf)
        if math.isinf(f_down):
            return -math.inf
        if Fraction(f_down) <= q:
            return f_down
        f = f_down
    raise RuntimeError("float lower did not converge (unreachable)")


def _clip01_frac(x: Fraction) -> Fraction:
    """Clip Fraction ``x`` to [0,1] rationally (monotone, exact)."""
    if x < 0:
        return Fraction(0, 1)
    if x > 1:
        return Fraction(1, 1)
    return x


def _hoeffding_radius_frac(b: Fraction, G: Fraction, N: int, delta: float) -> Fraction:
    """Exact rational UPPER for ``b + G*sqrt(2 ln(2/delta)/N)``.

    Validates ``b,G >= 0`` Fractions, ``N`` positive int (not bool),
    ``0<delta<1``. If ``G == 0`` returns ``b`` exactly without computing
    ``ln``/``sqrt`` (avoids ``0*inf`` and is exact). Otherwise computes
    ``arg = 2/delta`` exactly (``Fraction(delta)`` binary-exact),
    ``ln_up >= ln(arg)``, ``S = 2*ln_up/N``, ``sq_up >= sqrt(S)``,
    returns ``b + G*sq_up``. All Fraction ops exact; ``ln``/``sqrt`` steps
    are proven uppers, so the sum is a proven upper.
    """
    N = _require_positive_int("N", N)
    delta_f = _require_delta(delta)
    if not isinstance(b, Fraction) or not isinstance(G, Fraction):
        raise ValueError("b,G must be Fractions")
    if b < 0 or G < 0:
        raise ValueError("b,G must be nonnegative")
    if G == 0:
        return b
    delta_frac = Fraction(delta_f)
    # delta in (0,1) => arg = 2/delta > 2 > 1.
    arg = Fraction(2, 1) / delta_frac
    ln_up = _ln_upper_frac(arg, LN_PREC)
    S = Fraction(2, 1) * ln_up / N
    sq_up = _sqrt_upper_frac(S)
    return b + G * sq_up


def _bernoulli_radius_frac(N: int, delta: float) -> Fraction:
    """Exact rational UPPER for ``sqrt(ln(2/delta)/(2N))`` (k=0, [0,1]).

    Validates ``N`` positive int, ``0<delta<1``. Computes ``arg=2/delta``,
    ``ln_up``, ``S=ln_up/(2N)``, ``sqrt`` upper. Pure rational + proven
    enclosures.
    """
    N = _require_positive_int("N", N)
    delta_f = _require_delta(delta)
    delta_frac = Fraction(delta_f)
    arg = Fraction(2, 1) / delta_frac
    ln_up = _ln_upper_frac(arg, LN_PREC)
    S = ln_up / (2 * N)
    return _sqrt_upper_frac(S)


def _surrogate_upper_frac(b: Fraction, G: Fraction, N: int,
                          bits: int = SURROGATE_BITS) -> tuple[Fraction, bool]:
    """Rigorous upper for surrogate ``b + G/sqrt(N)``.

    Returns ``(upper, is_exact)``. If ``N`` is a perfect square
    (``s=isqrt(N), s*s==N``) or ``G==0``, returns exact ``b+G/s`` (or ``b``)
    with ``True``. Otherwise encloses ``sqrt(N)`` with ``SCALE=2**bits``:
    ``s0=isqrt(N*SCALE^2)``, ``lo=s0/SCALE <= sqrt(N)``, so
    ``1/sqrt(N) <= SCALE/s0`` and ``upper = b+G*SCALE/s0`` with ``False``.
    ``bits`` positive int. Pure integer/Fraction arithmetic.
    """
    N = _require_positive_int("N", N)
    if not isinstance(b, Fraction) or not isinstance(G, Fraction):
        raise ValueError("b,G must be Fractions")
    if b < 0 or G < 0:
        raise ValueError("b,G must be nonnegative")
    if _is_bool(bits) or not isinstance(bits, int) or bits <= 0:
        raise ValueError(f"bits must be positive int, got {bits!r}")
    if G == 0:
        return (b, True)
    s = math.isqrt(N)
    if s * s == N:
        return (b + G / s, True)
    scale = 1 << bits
    s0 = math.isqrt(N * scale * scale)
    # s0 >= scale > 0 for N>=1; lo = s0/scale <= sqrt(N).
    if s0 <= 0:
        raise RuntimeError("surrogate enclosure s0<=0 (unreachable for N>=1)")
    # Verify lo*lo <= N < hi*hi with hi=(s0+1)/scale (integer check).
    if not (s0 * s0 <= N * scale * scale < (s0 + 1) * (s0 + 1)):
        raise RuntimeError("surrogate isqrt check failed")
    inv_up = Fraction(scale, s0)
    return (b + G * inv_up, False)


# --- Hamming-weight channel (polynomial size, no enumeration) -------------

def _check_k(k) -> int:
    if _is_bool(k) or not isinstance(k, int):
        raise ValueError(f"k must be int, got {k!r}")
    if k < 0:
        raise ValueError(f"k must be >=0, got {k}")
    return k


def build_channel_half(k: int):
    """Exact + float Hamming-weight BSC channel at eta=1/4 (signal 1/2).

    Returns (W_frac, W_float). W[w][h] rows observed w, cols latent h.
    W_frac: (k+1)x(k+1) Fractions via convolution (polynomial size).
    W_float: numpy float64 array same shape (exactly representable for
    k<=6 since denominator 4^k dyadic). Columns sum to 1 exactly (frac)
    and within 1e-15 (float). No bitstring enumeration in production.
    """
    k = _check_k(k)
    import numpy as np

    eta = Fraction(1, 4)
    stay = Fraction(3, 4)
    W_frac = [[Fraction(0, 1) for _ in range(k + 1)] for _ in range(k + 1)]
    for h in range(k + 1):
        for w in range(k + 1):
            tot = Fraction(0, 1)
            lo = max(0, w - (k - h))
            hi = min(h, w)
            for u in range(lo, hi + 1):
                v = w - u
                tot += (math.comb(h, u) * (stay ** u) * (eta ** (h - u))
                        * math.comb(k - h, v) * (eta ** v) * (stay ** (k - h - v)))
            W_frac[w][h] = tot
    # float version via same convolution in float64
    eta_f = 0.25
    stay_f = 0.75
    W_float = np.zeros((k + 1, k + 1), dtype=np.float64)
    for h in range(k + 1):
        for w in range(k + 1):
            tot = 0.0
            lo = max(0, w - (k - h))
            hi = min(h, w)
            for u in range(lo, hi + 1):
                v = w - u
                tot += (math.comb(h, u) * (stay_f ** u) * (eta_f ** (h - u))
                        * math.comb(k - h, v) * (eta_f ** v) * (stay_f ** (k - h - v)))
            W_float[w, h] = tot
    # validate finite
    if not bool(__import__("numpy").isfinite(W_float).all()):
        raise ValueError("nonfinite float channel")
    return (W_frac, W_float)


def build_channel_common_float(k: int):
    """Float64 Hamming-weight channel at eta=(1-1/sqrt3)/2 (signal 1/sqrt3).

    Uses float64 math.sqrt(3). For certification see build_channel_mid.
    Returns numpy float64 (k+1)x(k+1). No enumeration.
    """
    k = _check_k(k)
    import numpy as np

    eta_f = (1.0 - 1.0 / math.sqrt(3.0)) / 2.0
    stay_f = 1.0 - eta_f
    if not (math.isfinite(eta_f) and 0.0 < eta_f < 0.5):
        raise ValueError(f"float eta out of range {eta_f}")
    W = np.zeros((k + 1, k + 1), dtype=np.float64)
    for h in range(k + 1):
        for w in range(k + 1):
            tot = 0.0
            lo = max(0, w - (k - h))
            hi = min(h, w)
            for u in range(lo, hi + 1):
                v = w - u
                tot += (math.comb(h, u) * (stay_f ** u) * (eta_f ** (h - u))
                        * math.comb(k - h, v) * (eta_f ** v) * (stay_f ** (k - h - v)))
            W[w, h] = tot
    import numpy as _np
    if not bool(_np.isfinite(W).all()):
        raise ValueError("nonfinite float channel")
    return W


def build_channel_mid(k: int, eta_mid: Fraction):
    """Exact Fraction channel at rational midpoint eta_mid. No enumeration."""
    k = _check_k(k)
    if not isinstance(eta_mid, Fraction):
        raise ValueError(f"eta_mid must be Fraction, got {type(eta_mid).__name__}")
    if not (Fraction(0, 1) < eta_mid < Fraction(1, 2)):
        raise ValueError(f"eta_mid out of (0,1/2): {eta_mid}")
    stay = Fraction(1, 1) - eta_mid
    W = [[Fraction(0, 1) for _ in range(k + 1)] for _ in range(k + 1)]
    for h in range(k + 1):
        for w in range(k + 1):
            tot = Fraction(0, 1)
            lo = max(0, w - (k - h))
            hi = min(h, w)
            for u in range(lo, hi + 1):
                v = w - u
                tot += (math.comb(h, u) * (stay ** u) * (eta_mid ** (h - u))
                        * math.comb(k - h, v) * (eta_mid ** v) * (stay ** (k - h - v)))
            W[w][h] = tot
    return W


# --- LP assembly + HiGHS solve --------------------------------------------

def _validate_W_float(W_float):
    """Validate float Hamming channel stochasticity (square/finite/range/cols).

    Requires 2-D square ``(k+1)x(k+1)`` float64, all finite, entries in
    ``[-1e-12, 1+1e-12]``, each column sum within ``1e-9`` of 1. Returns
    ``k``. Raises ValueError otherwise (including malformed 1x1/1xn/NaN).
    Called BEFORE any k=0 shortcut so malformed inputs always fail.
    """
    import numpy as np

    W_arr = np.asarray(W_float, dtype=np.float64)
    if W_arr.ndim != 2:
        raise ValueError(f"W must be 2-D, got ndim {W_arr.ndim}")
    if W_arr.shape[0] != W_arr.shape[1]:
        raise ValueError(f"W must be square (k+1)x(k+1), got shape {W_arr.shape}")
    k1 = W_arr.shape[0]
    if k1 == 0:
        raise ValueError("empty W")
    k = k1 - 1
    if k < 0:
        raise ValueError("empty W")
    if not bool(np.isfinite(W_arr).all()):
        raise ValueError("W contains nonfinite (NaN/inf) entries")
    if bool((W_arr < -1e-12).any()) or bool((W_arr > 1.0 + 1e-12).any()):
        raise ValueError("W entries out of [0,1] (tolerance 1e-12)")
    col_sums = W_arr.sum(axis=0)
    if not bool(np.isfinite(col_sums).all()):
        raise ValueError("W column sums nonfinite")
    if bool((abs(col_sums - 1.0) > 1e-9).any()):
        raise ValueError(f"W columns must sum to 1 (stochastic), got {col_sums}")
    return k


def assemble_lp(W_float, N: int):
    """Assemble HiGHS LP for given float channel and copy count.

    Variables x=[g_0..g_k, b, G], d=k+3. c=[0..0,1,1/sqrt(N)].
    A_ub/b_ub encode ``|W^T g - e0|_inf <= b`` and ``|g|_inf <= G`` with
    ``W[w,h]`` rows observed, cols latent, ``e0=(1,0,...,0)``.
    W_float must be finite stochastic (k+1)x(k+1); N any positive int.
    Returns (c, A_ub, b_ub, bounds) as numpy arrays/list.
    """
    import numpy as np

    N = _require_positive_int("N", N)
    k = _validate_W_float(W_float)
    W_float = np.asarray(W_float, dtype=np.float64)
    d = k + 3
    sN = math.sqrt(float(N))
    c = np.zeros(d, dtype=np.float64)
    c[k + 1] = 1.0
    c[k + 2] = 1.0 / sN
    if not bool(np.isfinite(c).all()):
        raise ValueError("nonfinite objective")
    rows = []
    rhs = []
    # bias constraints: Wg - b <= delta ; -Wg - b <= -delta
    for h in range(k + 1):
        delta = 1.0 if h == 0 else 0.0
        r1 = np.zeros(d, dtype=np.float64)
        r1[0:k + 1] = W_float[:, h]
        r1[k + 1] = -1.0
        rows.append(r1)
        rhs.append(delta)
        r2 = np.zeros(d, dtype=np.float64)
        r2[0:k + 1] = -W_float[:, h]
        r2[k + 1] = -1.0
        rows.append(r2)
        rhs.append(-delta)
    # box: g_w - G <=0 ; -g_w - G <=0
    for w in range(k + 1):
        r3 = np.zeros(d, dtype=np.float64)
        r3[w] = 1.0
        r3[k + 2] = -1.0
        rows.append(r3)
        rhs.append(0.0)
        r4 = np.zeros(d, dtype=np.float64)
        r4[w] = -1.0
        r4[k + 2] = -1.0
        rows.append(r4)
        rhs.append(0.0)
    A_ub = np.vstack(rows).astype(np.float64) if rows else np.zeros((0, d))
    b_ub = np.array(rhs, dtype=np.float64)
    if not (bool(np.isfinite(A_ub).all()) and bool(np.isfinite(b_ub).all())):
        raise ValueError("nonfinite LP data")
    bounds = [(None, None)] * (k + 1) + [(0.0, None), (0.0, None)]
    return (c, A_ub, b_ub, bounds)


def solve_lp(W_float, N: int, *, options: dict | None = None) -> dict:
    """Solve LP with SciPy HiGHS (float64). Fail clearly on failure/nonfinite.

    Validates W shape/finiteness/stochasticity BEFORE any k=0 shortcut, so
    malformed 1x1/1xn/NaN inputs fail instead of taking the exact path.
    Returns dict with g_float(list float), b_solver,G_solver floats,
    objective_solver float, status,nit,message,versions,options,times,
    residuals. Float feasibility is NOT a certificate; exact optimality NOT
    claimed. k=0 with valid [[1]] takes the exact path (no solver).
    """
    import numpy as np

    t0 = time.perf_counter()
    N = _require_positive_int("N", N)
    # Validate first (rejects malformed even for k=0).
    k = _validate_W_float(W_float)
    W_float = np.asarray(W_float, dtype=np.float64)
    if k == 0:
        # Exact Pauli path requires valid 1x1 stochastic [[1]] (within tol).
        if abs(float(W_float[0, 0]) - 1.0) > 1e-9:
            raise ValueError(f"k=0 channel must be [[1]], got {W_float!r}")
        t1 = time.perf_counter()
        try:
            import scipy
            scipy_ver = scipy.__version__
        except Exception:
            scipy_ver = "missing"
        # Objective 1/sqrt(N) as float (display only; exact surrogate elsewhere).
        return {
            "k": 0, "N": N, "method": "exact_k0_no_solver",
            "g_float": [1.0], "b_solver": 0.0, "G_solver": 1.0,
            "objective_solver": 1.0 / math.sqrt(float(N)),
            "success": True, "status": 0, "nit": 0, "message": "exact k=0 indicator",
            "numpy_version": np.__version__, "scipy_version": scipy_ver,
            "python_version": platform.python_version(),
            "options": {}, "solve_time_sec": t1 - t0,
            "max_constraint_violation": 0.0,
        }
    if options is None:
        options = dict(HIGHS_OPTIONS)
    # versions
    try:
        import scipy
        import scipy.optimize
        scipy_ver = scipy.__version__
        numpy_ver = np.__version__
    except Exception as e:
        raise RuntimeError(f"SciPy/NumPy required for LP solve: {e}") from e
    c, A_ub, b_ub, bounds = assemble_lp(W_float, N)
    # extra nonfinite guard
    if not (bool(np.isfinite(c).all()) and bool(np.isfinite(A_ub).all()) and bool(np.isfinite(b_ub).all())):
        raise ValueError("nonfinite LP data (refusing to call solver)")
    import scipy.optimize as _opt
    try:
        res = _opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                           method=HIGHS_METHOD, options=options)
    except Exception as e:
        raise RuntimeError(f"linprog raised {type(e).__name__}: {e}") from e
    t1 = time.perf_counter()
    # record; fail clearly if not optimal
    success = bool(getattr(res, "success", False))
    status = int(getattr(res, "status", -1))
    nit = int(getattr(res, "nit", -1)) if getattr(res, "nit", None) is not None else -1
    message = str(getattr(res, "message", ""))
    if not success or status != 0 or getattr(res, "x", None) is None:
        raise RuntimeError(f"HiGHS solve failed: success={success} status={status} message={message}")
    x = np.asarray(res.x, dtype=np.float64)
    if x.shape != (k + 3,) or not bool(np.isfinite(x).all()):
        raise RuntimeError(f"HiGHS returned nonfinite/wrong-shape x: {x!r}")
    g = [float(v) for v in x[:k + 1]]
    if not all(math.isfinite(v) for v in g):
        raise RuntimeError("nonfinite g from solver")
    b_sol = float(x[k + 1])
    G_sol = float(x[k + 2])
    if not (math.isfinite(b_sol) and math.isfinite(G_sol)):
        raise RuntimeError("nonfinite b/G from solver")
    if b_sol < -1e-7 or G_sol < -1e-7:
        raise RuntimeError(f"solver b/G negative beyond tolerance: b={b_sol} G={G_sol}")
    # residual: max(A_ub x - b_ub) should be <= tol; report (not certificate)
    viol = float(np.max(A_ub @ x - b_ub)) if A_ub.size else 0.0
    fun = float(getattr(res, "fun", float(np.dot(c, x))))
    return {
        "k": k, "N": N, "method": HIGHS_METHOD,
        "g_float": g, "b_solver": b_sol, "G_solver": G_sol,
        "objective_solver": fun,
        "success": success, "status": status, "nit": nit, "message": message,
        "numpy_version": numpy_ver, "scipy_version": scipy_ver,
        "python_version": platform.python_version(),
        "options": dict(options), "solve_time_sec": t1 - t0,
        "max_constraint_violation": viol,
    }


# --- rounding + certification ----------------------------------------------

def _round_one_to_den(gw: Fraction, den: int) -> tuple[Fraction, Fraction]:
    if den <= 0:
        raise ValueError("den must be positive")
    f = gw * den
    num = f.numerator
    dd = f.denominator
    q = num // dd
    rem = num % dd
    left = rem * 2
    if left < dd:
        m = q
    elif left > dd:
        m = q + 1
    else:
        m = q if (q % 2 == 0) else q + 1
    gr = Fraction(m, den)
    return (gr, abs(gw - gr))


def round_coefficients(g_float_list, den: int = ROUND_DEN):
    """Round solver floats (exact binary) to denominator den (ties to even).

    Returns (g_round list Fractions, maxdiff Fraction). Validates finite.
    """
    if _is_bool(den) or not isinstance(den, int) or den <= 0:
        raise ValueError(f"den must be positive int, got {den!r}")
    if not isinstance(g_float_list, (list, tuple)) or len(g_float_list) == 0:
        raise ValueError("g_float_list must be nonempty list")
    g_raw = []
    for v in g_float_list:
        if _is_bool(v) or not isinstance(v, (int, float, Fraction)):
            raise ValueError(f"g entries must be numbers, got {v!r}")
        if isinstance(v, float) and not math.isfinite(v):
            raise ValueError(f"g entry nonfinite {v!r}")
        if isinstance(v, Fraction):
            g_raw.append(v)
        elif isinstance(v, int):
            g_raw.append(Fraction(v, 1))
        else:
            g_raw.append(Fraction(v))  # exact binary-float-as-Fraction
    g_round = []
    maxdiff = Fraction(0, 1)
    for gw in g_raw:
        gr, d = _round_one_to_den(gw, den)
        g_round.append(gr)
        if d > maxdiff:
            maxdiff = d
    return (g_raw, g_round, maxdiff)


def _residuals_exact(W_frac, g_list) -> tuple[Fraction, list[Fraction], int]:
    k = len(g_list) - 1
    if len(W_frac) != k + 1 or any(len(row) != k + 1 for row in W_frac):
        raise ValueError("W/g dimension mismatch")
    resids = []
    for h in range(k + 1):
        s = sum(W_frac[w][h] * g_list[w] for w in range(k + 1))
        delta = Fraction(1, 1) if h == 0 else Fraction(0, 1)
        resids.append(s - delta)
    b = max(abs(r) for r in resids)
    hstar = max(range(k + 1), key=lambda h: abs(resids[h]))
    return (b, resids, hstar)


def _G_of_frac(g_list) -> Fraction:
    return max(abs(v) for v in g_list)


# --- low-level certification helpers (private, no channel identity) --------
# These take an EXPLICIT W/enclosure supplied by the caller and therefore do
# NOT by themselves certify that W is the claimed channel's matrix. They are
# private low-level primitives; public construction (plan_estimator,
# solve_and_certify) always builds the channel freshly for the selected
# channel/k and validates. Do not call these with arbitrary matrices and
# label the result as this channel's certified output.

def _certify_half(k: int, W_frac, g_float_list, den: int = ROUND_DEN) -> dict:
    """Low-level exact certificate for rational half channel (private).

    Computes raw (binary-float-exact) and rounded (1/den) residuals freshly
    (never reusing solver b,G). Checks b_round<=b_raw+maxdiff. Caller must
    ensure W_frac is the true half-channel matrix for this k.
    """
    k = _check_k(k)
    t0 = time.perf_counter()
    g_raw, g_round, maxdiff = round_coefficients(g_float_list, den)
    if len(g_raw) != k + 1:
        raise ValueError(f"g length {len(g_raw)} vs k+1={k+1}")
    b_raw, res_raw, hs_raw = _residuals_exact(W_frac, g_raw)
    b_round, res_round, hs_round = _residuals_exact(W_frac, g_round)
    G_raw = _G_of_frac(g_raw)
    G_round = _G_of_frac(g_round)
    ok = b_round <= b_raw + maxdiff
    if not ok:
        raise RuntimeError(f"rounding cert failed: b_round={b_round} b_raw={b_raw} maxdiff={maxdiff}")
    t1 = time.perf_counter()
    return {
        "channel": CHANNEL_HALF, "k": k, "den": den,
        "g_raw_exact": [str(v) for v in g_raw],
        "g_round": [str(v) for v in g_round],
        "g_round_float": [float(v) for v in g_round],
        "maxdiff": str(maxdiff), "maxdiff_float": float(maxdiff),
        "b_raw": str(b_raw), "b_raw_float": float(b_raw),
        "b_round": str(b_round), "b_round_float": float(b_round),
        "G_raw": str(G_raw), "G_raw_float": float(G_raw),
        "G_round": str(G_round), "G_round_float": float(G_round),
        "residuals_raw": [str(v) for v in res_raw],
        "residuals_round": [str(v) for v in res_round],
        "hstar_raw": hs_raw, "hstar_round": hs_round,
        "cert_ok": bool(ok), "cert_time_sec": t1 - t0,
        "b_cert": str(b_round), "G_cert": str(G_round),
    }


def _certify_common(k: int, W_mid, eta_info: dict, g_float_list, den: int = ROUND_DEN) -> dict:
    """Low-level exact certificate for common channel via midpoint (private).

    W_mid exact at eta_mid; b_round_mid/coupling as before. Caller must
    ensure W_mid/eta_info are the fresh common-channel values for this k
    (see eta_interval_common + build_channel_mid). Coefficient vs channel
    errors distinct.
    """
    k = _check_k(k)
    t0 = time.perf_counter()
    max_err: Fraction = eta_info["max_eta_error"]
    if not isinstance(max_err, Fraction) or max_err <= 0:
        raise ValueError("max_eta_error must be positive Fraction")
    g_raw, g_round, maxdiff = round_coefficients(g_float_list, den)
    if len(g_raw) != k + 1:
        raise ValueError("g length mismatch")
    b_raw_mid, res_raw, hs_raw = _residuals_exact(W_mid, g_raw)
    b_round_mid, res_round, hs_round = _residuals_exact(W_mid, g_round)
    G_raw = _G_of_frac(g_raw)
    G_round = _G_of_frac(g_round)
    ok_round = b_round_mid <= b_raw_mid + maxdiff
    if not ok_round:
        raise RuntimeError("rounding part of common cert failed")
    coupling = Fraction(2, 1) * G_round * k * max_err
    b_cert = b_round_mid + coupling
    t1 = time.perf_counter()
    return {
        "channel": CHANNEL_COMMON, "k": k, "den": den,
        "sqrt3_lo": str(eta_info["sqrt3_lo"]), "sqrt3_hi": str(eta_info["sqrt3_hi"]),
        "eta_lo": str(eta_info["eta_lo"]), "eta_hi": str(eta_info["eta_hi"]),
        "eta_mid": str(eta_info["eta_mid"]),
        "max_eta_error": str(max_err), "max_eta_error_float": float(max_err),
        "bits": eta_info["bits"],
        "g_raw_exact": [str(v) for v in g_raw],
        "g_round": [str(v) for v in g_round],
        "g_round_float": [float(v) for v in g_round],
        "maxdiff": str(maxdiff), "maxdiff_float": float(maxdiff),
        "b_raw_mid": str(b_raw_mid), "b_raw_mid_float": float(b_raw_mid),
        "b_round_mid": str(b_round_mid), "b_round_mid_float": float(b_round_mid),
        "G_raw": str(G_raw), "G_raw_float": float(G_raw),
        "G_round": str(G_round), "G_round_float": float(G_round),
        "residuals_raw_mid": [str(v) for v in res_raw],
        "residuals_round_mid": [str(v) for v in res_round],
        "hstar_raw": hs_raw, "hstar_round": hs_round,
        "coupling": str(coupling), "coupling_float": float(coupling),
        "b_cert": str(b_cert), "b_cert_float": float(b_cert),
        "G_cert": str(G_round),
        "cert_ok": bool(ok_round), "cert_time_sec": t1 - t0,
    }


def _fresh_b_G(k: int, channel: str, g_tuple: tuple[Fraction, ...],
               bits: int = ENCLOSURE_BITS) -> tuple[Fraction, Fraction, dict]:
    """Fresh rigorous (b,G) for executed Fractions at selected channel/k.

    Never re-rounds ``g``; uses it exactly. For half builds exact W; for
    common builds fresh eta enclosure + W_mid and adds coupling
    ``2*G*k*max_err``. For k=0 requires ``g == (1,)`` and returns (0,1).
    Returns ``(b_fresh, G_fresh, info)`` where info holds channel details
    (W not returned to avoid large copies; residuals recomputed by caller
    if needed). Raises ValueError on length/type mismatch.
    """
    k = _check_k(k)
    _require_channel(channel)
    if not isinstance(g_tuple, (tuple, list)):
        raise ValueError("g must be tuple/list of Fractions")
    g_list = list(g_tuple)
    if len(g_list) != k + 1:
        raise ValueError(f"g length {len(g_list)} vs k+1={k+1}")
    for v in g_list:
        if not isinstance(v, Fraction):
            raise ValueError(f"g entries must be Fractions (no silent re-rounding), got {type(v).__name__}={v!r}")
    if k == 0:
        if not (len(g_list) == 1 and g_list[0] == Fraction(1, 1)):
            raise ValueError(f"k=0 requires exact indicator score [1] for Bernoulli shortcut, got {g_list!r}")
        return (Fraction(0, 1), Fraction(1, 1), {"channel": channel, "k": 0})
    G_fresh = _G_of_frac(g_list)
    if channel == CHANNEL_HALF:
        W_frac, _ = build_channel_half(k)
        b_fresh, _, _ = _residuals_exact(W_frac, g_list)
        return (b_fresh, G_fresh, {"channel": channel, "k": k})
    # common: fresh enclosure + midpoint
    einfo = eta_interval_common(bits)
    W_mid = build_channel_mid(k, einfo["eta_mid"])
    b_mid, _, _ = _residuals_exact(W_mid, g_list)
    coupling = Fraction(2, 1) * G_fresh * k * einfo["max_eta_error"]
    return (b_mid + coupling, G_fresh, {"channel": channel, "k": k,
                                       "b_mid": b_mid, "coupling": coupling,
                                       "eta_info": einfo})


# --- confidence (rigorous outward, exact + display) -------------------------

def surrogate_exact(b_frac: Fraction, G_frac: Fraction, N: int) -> Fraction:
    """Exact surrogate ``b+G/sqrt(N)`` for perfect-square N (legacy helper).

    For non-squares raises ValueError (use ``surrogate_bounds`` for rigorous
    upper). Kept for frozen-grid exactness; new code prefers
    ``surrogate_bounds`` which labels exact vs upper.
    """
    N = _require_positive_int("N", N)
    if not isinstance(b_frac, Fraction) or not isinstance(G_frac, Fraction):
        raise ValueError("b,G must be Fractions")
    s = math.isqrt(N)
    if s * s == N:
        return b_frac + G_frac / s
    raise ValueError(f"surrogate_exact requires perfect-square N, got {N} (use surrogate_bounds)")


def surrogate_bounds(b_frac: Fraction, G_frac: Fraction, N: int,
                     bits: int = SURROGATE_BITS) -> dict:
    """Rigorous surrogate bounds for any positive N.

    Returns dict with ``upper`` (Fraction), ``upper_float`` (outward float),
    ``is_exact`` (True iff perfect square or G==0), ``symbolic``
    (``"b+G/sqrt(N)"``). Perfect squares give exact ``b+G/s``; nonsquares
    give proven upper ``b+G*SCALE/isqrt(N*SCALE^2)`` (see
    ``_surrogate_upper_frac``). No arbitrary padding.
    """
    up, is_ex = _surrogate_upper_frac(b_frac, G_frac, N, bits)
    return {"upper": up, "upper_float": _float_upper(up), "is_exact": bool(is_ex),
            "symbolic": "b+G/sqrt(N)", "N": N,
            "b": str(b_frac), "G": str(G_frac)}


def hoeffding_radius_exact(b_frac: Fraction, G_frac: Fraction, N: int, delta: float) -> Fraction:
    """Exact rational UPPER for Hoeffding radius (see ``_hoeffding_radius_frac``)."""
    return _hoeffding_radius_frac(b_frac, G_frac, N, delta)


def hoeffding_radius_conservative(b_frac: Fraction, G_frac: Fraction, N: int, delta: float) -> float:
    """Rigorous outward float UPPER for ``b+G*sqrt(2 ln(2/delta)/N)``.

    Computed as ``_float_upper(_hoeffding_radius_frac(...))``. No arbitrary
    ``1e-12`` budget; tight to ≤few ulps above the proven rational upper.
    Returns exact ``Fraction`` upper via ``hoeffding_radius_exact`` alongside
    if needed (see ``hoeffding_radius_bounds``).
    """
    return _float_upper(_hoeffding_radius_frac(b_frac, G_frac, N, delta))


def hoeffding_radius_bounds(b_frac: Fraction, G_frac: Fraction, N: int, delta: float) -> dict:
    """Both exact rational upper and outward float display for Hoeffding radius."""
    exact = _hoeffding_radius_frac(b_frac, G_frac, N, delta)
    return {"exact": exact, "exact_str": str(exact), "display": _float_upper(exact)}


def bernoulli_radius_exact(N: int, delta: float) -> Fraction:
    """Exact rational UPPER for Bernoulli radius (see ``_bernoulli_radius_frac``)."""
    return _bernoulli_radius_frac(N, delta)


def bernoulli_radius_conservative(N: int, delta: float) -> float:
    """Rigorous outward float UPPER for ``sqrt(ln(2/delta)/(2N))``."""
    return _float_upper(_bernoulli_radius_frac(N, delta))


def bernoulli_radius_bounds(N: int, delta: float) -> dict:
    """Both exact and display for Bernoulli radius."""
    exact = _bernoulli_radius_frac(N, delta)
    return {"exact": exact, "exact_str": str(exact), "display": _float_upper(exact)}


def sufficient_copies_bernoulli(b_frac: Fraction, delta: float, eps: float = 0.05):
    """Rigorous Bernoulli sufficient copies ``ceil(ln(2/delta)/(2(eps-b)^2))``.

    Exact ``eps-b`` Fraction comparison (``eps`` binary-exact via
    ``Fraction(eps)``); ``ln`` upper via ``_ln_upper_frac``; exact integer
    ceil of upper rational (conservative). Handles ``G``-free case, subnormal
    delta, large N/A, bias very close to eps (huge need as int, no overflow).
    Returns ``(need or None, reason or None)`` plus exact ratio in reason?
    For ``b >= eps`` returns ``(None, reason)``. Validates ``b`` Fraction
    >=0, ``delta`` in (0,1), ``eps`` finite >0. At reference
    ``b=0, delta=1/3, eps=1/20`` gives 359.
    """
    delta_f = _require_delta(delta)
    eps_f = _require_eps(eps)
    if not isinstance(b_frac, Fraction):
        raise ValueError("b must be Fraction")
    if b_frac < 0:
        raise ValueError("b must be nonnegative")
    eps_frac = Fraction(eps_f)
    if not (b_frac < eps_frac):
        return (None, f"b={float(b_frac):.6g} >= eps={eps_f}; formula needs eps>b")
    delta_frac = Fraction(delta_f)
    arg = Fraction(2, 1) / delta_frac
    ln_up = _ln_upper_frac(arg, LN_PREC)
    denom = (eps_frac - b_frac) * (eps_frac - b_frac)
    # denom > 0 since b < eps.
    ratio = ln_up / (2 * denom)
    # Exact ceil of positive Fraction p/q.
    p, q = ratio.numerator, ratio.denominator
    need = (p + q - 1) // q
    # Documented resource bound: need fits in Python int (arbitrary); float
    # display of ratio may overflow to inf, but need remains exact. If need
    # exceeds 10**18 callers should note large-size limits (no scaling claim).
    if need < 0:
        raise RuntimeError("negative need (unreachable)")
    return (need, None)


def sufficient_copies_conservative(b_frac: Fraction, G_frac: Fraction, delta: float, eps: float = 0.05):
    """Rigorous sufficient copies ``ceil(2G^2 ln(2/delta)/(eps-b)^2)``.

    Exact ceil of upper rational; ``G==0`` short-circuits to ``0`` (no
    ``0*inf``) when ``b<eps``. Validates ``b,G`` Fractions >=0, ``delta``,
    ``eps``. Returns ``(need or None, reason or None)``. Sufficient only.
    """
    delta_f = _require_delta(delta)
    eps_f = _require_eps(eps)
    if not isinstance(b_frac, Fraction) or not isinstance(G_frac, Fraction):
        raise ValueError("b,G must be Fractions")
    if b_frac < 0 or G_frac < 0:
        raise ValueError("b,G must be nonnegative")
    eps_frac = Fraction(eps_f)
    if not (b_frac < eps_frac):
        return (None, f"b={float(b_frac):.6g} >= eps={eps_f}; formula needs eps>b")
    if G_frac == 0:
        return (0, None)
    delta_frac = Fraction(delta_f)
    arg = Fraction(2, 1) / delta_frac
    ln_up = _ln_upper_frac(arg, LN_PREC)
    denom = (eps_frac - b_frac) * (eps_frac - b_frac)
    ratio = Fraction(2, 1) * G_frac * G_frac * ln_up / denom
    p, q = ratio.numerator, ratio.denominator
    need = (p + q - 1) // q
    if need < 0:
        raise RuntimeError("negative need (unreachable)")
    return (need, None)


# --- immutable plan ---------------------------------------------------------

@dataclass(frozen=True)
class EstimatorPlan:
    """Frozen pre-data measurement design (immutable, deep-copied).

    All sequence fields are tuples (immutable; lists rejected at execution);
    Fractions/floats/ints/str are immutable values. Constructed ONLY via
    ``plan_estimator`` which freshly computes ``b_cert/G_cert`` from
    ``g_round`` at ``channel``/``k`` and validates any caller claims.
    ``rs_hat`` floats approximate exact ``r/||r||``. Mutating the original
    ``rs``/``g`` lists afterwards, or attempting ``plan.N = ...``, cannot
    alter collection (``run_streaming`` revalidates via one canonical
    reconstruction before any callback/RNG and uses only that snapshot,
    never re-solves/rebuilds per copy).
    """
    rs_raw: tuple[tuple[float, float, float], ...]
    rs_hat: tuple[tuple[float, float, float], ...]
    n: int
    k: int
    K_idx: tuple[int, ...]
    S_idx: tuple[int, ...]
    L_hat_list: tuple[float, ...]
    stab_axis: tuple[str | None, ...]
    stab_want: tuple[int | None, ...]
    N: int
    delta: float
    channel: str
    g_round: tuple[Fraction, ...]
    b_cert: Fraction
    G_cert: Fraction
    bits: int
    den: int

    def __getitem__(self, key):  # minimal dict-compat for legacy readers
        if key == "rs":
            return list(self.rs_raw)
        if key == "classified":
            return {"rs": list(self.rs_raw), "rs_hat": list(self.rs_hat),
                    "n": self.n, "k": self.k,
                    "K_idx": list(self.K_idx), "S_idx": list(self.S_idx),
                    "L_list": [abs(r[0]) + abs(r[1]) + abs(r[2]) for r in self.rs_raw],
                    "L_hat_list": list(self.L_hat_list),
                    "stab_axis": list(self.stab_axis), "stab_want": list(self.stab_want)}
        return getattr(self, key)


def _extract_design(design):
    """Extract (rs_hat, n, k, K_idx, S_idx, stab_axis, stab_want, L_hat) tuples.

    Accepts ``EstimatorPlan`` (immutable, preferred) or legacy ``classified``
    dict from ``classify_target``. For dicts, computes normalized hats if
    missing and validates. Returns tuples/lists of immutable values (copies).
    """
    if isinstance(design, EstimatorPlan):
        return (tuple(design.rs_hat), design.n, design.k,
                tuple(design.K_idx), tuple(design.S_idx),
                tuple(design.stab_axis), tuple(design.stab_want),
                tuple(design.L_hat_list))
    if isinstance(design, dict) and "rs" in design and "n" in design and "k" in design:
        rs_raw = design["rs"]
        n = design["n"]
        k = design["k"]
        K_idx = tuple(design["K_idx"])
        S_idx = tuple(design["S_idx"])
        stab_axis = tuple(design.get("stab_axis", [None] * n))
        stab_want = tuple(design.get("stab_want", [None] * n))
        # Normalized hats: use stored rs_hat if present, else compute.
        if "rs_hat" in design and design["rs_hat"] is not None:
            rs_hat = tuple(tuple(float(v) for v in r) for r in design["rs_hat"])
        else:
            rs_hat = tuple(_normalized_hat(tuple(float(v) for v in r)) for r in rs_raw)
            rs_hat = tuple(rs_hat)  # type: ignore
        if "L_hat_list" in design and design["L_hat_list"] is not None:
            L_hat = tuple(float(v) for v in design["L_hat_list"])
        else:
            L_hat = tuple(abs(h[0]) + abs(h[1]) + abs(h[2]) for h in rs_hat)
        return (rs_hat, n, k, K_idx, S_idx, stab_axis, stab_want, L_hat)
    raise ValueError("design must be EstimatorPlan or classified dict from classify_target")


# --- operational sampling ---------------------------------------------------

def make_rng(seed=None) -> random.Random:
    if seed is None:
        return random.Random()
    if _is_bool(seed) or not isinstance(seed, int):
        raise ValueError(f"seed must be int, got {seed!r}")
    return random.Random(seed)


def _validate_axes(axes, n: int):
    if not isinstance(axes, (list, tuple)) or len(axes) != n:
        raise ValueError(f"axes must be length-{n} list, got {axes!r}")
    for a in axes:
        if not isinstance(a, str) or a not in PAULI_AXES:
            raise ValueError(f"invalid axis {a!r}; must be X/Y/Z")


def _validate_outcomes(outcomes, n: int):
    if not isinstance(outcomes, (list, tuple)) or len(outcomes) != n:
        raise ValueError(f"outcomes must be length-{n} list, got {outcomes!r}")
    for o in outcomes:
        if _is_bool(o) or not isinstance(o, int) or o not in (1, -1):
            raise ValueError(f"invalid outcome {o!r}; must be +1/-1 int (bool rejected)")


def _require_rng_draw(u) -> float:
    """Validate RNG draw satisfies 0 <= u < 1 (1.0 rejected, never clamped)."""
    try:
        f = float(u)
    except Exception:
        raise ValueError(f"rng draw must be number in [0,1), got {u!r}")
    if _is_bool(u):
        raise ValueError(f"rng draw must be number in [0,1), got {u!r}")
    if not math.isfinite(f) or not (0.0 <= f < 1.0):
        raise ValueError(f"rng.random() must satisfy 0<=u<1 (1.0 rejected), got {u!r}")
    return f


def sample_axes_for_copy(classified, rng: random.Random) -> list[str]:
    """Sample one copy's Pauli axes. Noisy: |r_hat_i|/L_hat; stabilizer: deterministic.

    ``classified`` may be ``EstimatorPlan`` (preferred, immutable) or legacy
    dict from ``classify_target``. Uses normalized directions consistently.
    ``rng`` must have ``.random()`` returning ``0<=u<1`` (``1.0`` rejected,
    never silently remapped). Returns length-n axes. Zero-prob axes (hat
    coord exactly 0.0) never selected; if float partitioning would select
    one, raises RuntimeError to surface the bug (score path rejects with
    ValueError).
    """
    rs_hat, n, k, K_idx_t, S_idx_t, stab_axis_t, stab_want_t, L_hat_t = _extract_design(classified)
    if not hasattr(rng, "random"):
        raise ValueError("rng must have .random()")
    K_set = set(K_idx_t)
    # Map qubit index -> L_hat for noisy qubits.
    L_by_j = {j: L_hat_t[j] for j in K_idx_t}
    axes: list[str] = []
    for j in range(n):
        if j in K_set:
            hat = rs_hat[j]
            L = float(L_by_j[j])
            if not math.isfinite(L) or L <= 0:
                raise ValueError("nonpositive L_hat")
            pX = abs(float(hat[0])) / L
            pY = abs(float(hat[1])) / L
            u = _require_rng_draw(rng.random())
            if u < pX:
                axes.append("X")
            elif u < pX + pY:
                axes.append("Y")
            else:
                axes.append("Z")
            sel = axes[-1]
            idx = {"X": 0, "Y": 1, "Z": 2}[sel]
            if float(hat[idx]) == 0.0:
                raise RuntimeError(f"sampled zero-prob axis {sel} for r_hat={hat}")
        else:
            ax = stab_axis_t[j]
            if ax not in PAULI_AXES:
                raise RuntimeError(f"missing stabilizer axis for qubit {j}")
            axes.append(ax)  # type: ignore
    _validate_axes(axes, n)
    return axes


def _sign_of(r_hat, axis: str) -> int:
    idx = {"X": 0, "Y": 1, "Z": 2}[axis]
    v = float(r_hat[idx])
    if v > 0:
        return 1
    if v < 0:
        return -1
    raise ValueError(f"zero coordinate has no sign (axis {axis}, r_hat={r_hat})")


def _degradation_mean(L_hat: float, channel: str) -> float:
    """Degradation coin mean from normalized ``L_hat`` (float proxy).

    ``m = L_hat/2`` (half) or ``L_hat/sqrt(3)`` (common). True mathematical
    ``m`` satisfies ``|m|<=1`` (``L_hat<=sqrt(3)`` for unit directions, with
    equality at 111; ``L_hat<=2``? Actually ``L_hat<=sqrt(3)<2``, so both
    ``<=1``). Float rounding may give ``1+eps`` (eps~1e-16); values within
    ``1e-12`` above 1 are clipped to exactly 1.0 as the numerical proxy
    (rounding outside certificate, documented); larger overshoots rejected.
    """
    _require_channel(channel)
    if not math.isfinite(L_hat) or L_hat <= 0:
        raise ValueError(f"invalid L_hat {L_hat}")
    if channel == CHANNEL_HALF:
        m = L_hat / 2.0
    else:
        m = L_hat / math.sqrt(3.0)
    if not math.isfinite(m):
        raise ValueError(f"nonfinite degradation mean {m}")
    if m > 1.0 + 1e-12 or m < -1.0 - 1e-12:
        raise ValueError(f"degradation mean out of [-1,1]: {m} (L_hat={L_hat}, channel={channel})")
    if m > 1.0:
        m = 1.0
    if m < -1.0:
        m = -1.0
    return m


def score_one_copy(classified, g_round: list[Fraction], axes, outcomes, rng: random.Random, channel: str) -> dict:
    """Score single copy: sign-correct, degrade, H=g[w]*F.

    ``classified`` may be ``EstimatorPlan`` or dict; uses normalized hats.
    Validates stabilizer axes match exactly (mismatches rejected) and noisy
    zero-prob axes rejected (no silent wrong-axis scoring). ``g_round``
    length k+1 Fractions (executed, never re-rounded). ``rng`` draws must
    satisfy 0<=u<1 (1.0 rejected). Returns H Fraction, w, flag, bits.
    Uses ALL-copy denominator convention.
    """
    _require_channel(channel)
    if not isinstance(g_round, (list, tuple)) or len(g_round) == 0:
        raise ValueError("g_round must be nonempty list")
    for v in g_round:
        if not isinstance(v, Fraction):
            raise ValueError(f"g_round must be Fractions, got {type(v).__name__}")
    rs_hat, n, k, K_idx_t, S_idx_t, stab_axis_t, stab_want_t, L_hat_t = _extract_design(classified)
    if len(g_round) != k + 1:
        raise ValueError(f"g length {len(g_round)} vs k+1={k+1}")
    _validate_axes(axes, n)
    _validate_outcomes(outcomes, n)
    if not hasattr(rng, "random"):
        raise ValueError("rng must have .random()")
    K_idx: list[int] = list(K_idx_t)
    S_idx: list[int] = list(S_idx_t)
    # Stabilizer axes must match exactly; reject mismatches (no silent scoring).
    for j in S_idx:
        exp_ax = stab_axis_t[j]
        if axes[j] != exp_ax:
            raise ValueError(f"stabilizer axis mismatch at qubit {j}: got {axes[j]!r}, expected {exp_ax!r}")
    # Noisy zero-prob axes rejected.
    for j in K_idx:
        ax = axes[j]
        idx = {"X": 0, "Y": 1, "Z": 2}[ax]
        if float(rs_hat[j][idx]) == 0.0:
            raise ValueError(f"zero-prob noisy axis {ax} at qubit {j} for r_hat={rs_hat[j]}")
    # Stabilizer flag: all S outcomes match want.
    flag = 1
    for j in S_idx:
        want = stab_want_t[j]
        if outcomes[j] != want:
            flag = 0
            break
    if k == 0:
        H = Fraction(flag, 1) * g_round[0]
        return {"H": H, "w": 0, "flag": flag, "t_bits": [], "s_bits": []}
    t_bits: list[int] = []
    s_bits: list[int] = []
    for jj, j in enumerate(K_idx):
        ax = axes[j]
        o = outcomes[j]
        s = _sign_of(rs_hat[j], ax) * o
        if s not in (1, -1):
            raise RuntimeError("sign correction bug")
        s_bits.append(s)
        L = float(L_hat_t[j])
        m = _degradation_mean(L, channel)
        u = _require_rng_draw(rng.random())
        p_plus = (1.0 + m) / 2.0
        d = 1 if u < p_plus else -1
        t_bits.append(s * d)
    w = sum(1 for t in t_bits if t == -1)
    if not (0 <= w <= k):
        raise RuntimeError(f"weight out of range {w}")
    H = g_round[w] * flag
    return {"H": H, "w": w, "flag": flag, "t_bits": t_bits, "s_bits": s_bits}


def plan_estimator(rs, N: int, delta: float = DELTA_DEFAULT,
                   channel: str = DEFAULT_CHANNEL,
                   g_round=None,
                   b_cert=None, G_cert=None, *,
                   bits: int = ENCLOSURE_BITS, den: int = ROUND_DEN) -> EstimatorPlan:
    """Freeze operational plan with FRESH certificate binding (immutable).

    Args:
      rs: heterogeneous pure-direction target list (validated; represented
        target is ``r/||r||`` exactly, stored as ``rs_hat`` explicitly).
      N: any ordinary positive int (not bool), e.g. 3,17,359 (not only squares).
      delta: failure level in (0,1) (default 1/3).
      channel: "common" (default, theorem) or "degraded_half" (explicit).
      g_round: executed score Fractions length k+1 (never re-rounded; each
        must be ``Fraction``, bool rejected). For k=0 must be exactly
        ``[Fraction(1)]`` (Bernoulli shortcut); any other k=0 score raises.
      b_cert/G_cert: OPTIONAL caller claims (Fractions) for compatibility.
        If supplied they are VALIDATED against freshly recomputed bounds and
        understated claims raise ValueError (prevents false certificates
        such as zero-score ``[0]`` with ``b=G=0`` or wrong-channel reuse).
        The plan always stores the FRESH bounds, never the claims.
      bits/den: enclosure bits and rounding denominator reference (den stored
        for provenance; user Fractions are used exactly regardless of den).

    Returns frozen ``EstimatorPlan`` (tuples, deep-copied before any callback).
    Caller must then collect exactly N fresh i.i.d. copies with no
    stopping/reuse. Raises ValueError on any mismatch/understatement.
    """
    # Deep-copy inputs before validation (preserve originals, freeze).
    # Validate target first (raw + normalized).
    vrs = validate_target(rs)
    # Copy raw tuples (floats immutable, but list/tuple container copied).
    rs_raw_t = tuple((float(r[0]), float(r[1]), float(r[2])) for r in vrs)
    rs_hat_t = tuple(_normalized_hat(r) for r in rs_raw_t)
    N = _require_positive_int("N", N)
    delta = _require_delta(delta)
    _require_channel(channel)
    if _is_bool(bits) or not isinstance(bits, int) or bits <= 0:
        raise ValueError(f"bits must be positive int, got {bits!r}")
    if _is_bool(den) or not isinstance(den, int) or den <= 0:
        raise ValueError(f"den must be positive int, got {den!r}")
    classified = classify_target(list(rs_raw_t))
    k = classified["k"]
    n = classified["n"]
    if g_round is None:
        raise ValueError("g_round must be supplied (Fractions length k+1)")
    if not isinstance(g_round, (list, tuple)):
        raise ValueError(f"g_round must be list/tuple of Fractions, got {type(g_round).__name__}")
    if len(g_round) != k + 1:
        raise ValueError(f"g_round length {len(g_round)} vs k+1={k+1}")
    g_list_checked: list[Fraction] = []
    for v in g_round:
        if not isinstance(v, Fraction):
            raise ValueError(f"g_round must be Fractions (no silent re-rounding), got {type(v).__name__}={v!r}")
        g_list_checked.append(v)
    g_t = tuple(g_list_checked)
    # k=0 exact-indicator enforcement (prefer exact indicator only).
    if k == 0 and not (len(g_t) == 1 and g_t[0] == Fraction(1, 1)):
        raise ValueError(f"k=0 requires exact indicator score [1] for Bernoulli shortcut, got {[str(v) for v in g_t]}")
    # Fresh rigorous bounds from executed scores at selected channel/k.
    b_fresh, G_fresh, _info = _fresh_b_G(k, channel, g_t, bits)
    # Validate optional caller claims (reject understatements, wrong-channel).
    if b_cert is not None or G_cert is not None:
        if not isinstance(b_cert, Fraction) or not isinstance(G_cert, Fraction):
            raise ValueError("caller b_cert/G_cert must both be Fractions when supplied")
        if b_cert < 0 or G_cert < 0:
            raise ValueError("caller b_cert/G_cert must be nonnegative")
        if b_cert < b_fresh:
            raise ValueError(f"caller b_cert {b_cert} understates fresh bound {b_fresh} at channel {channel!r} k={k} (false certificate rejected)")
        if G_cert < G_fresh:
            raise ValueError(f"caller G_cert {G_cert} understates fresh range {G_fresh} at channel {channel!r} k={k} (false certificate rejected)")
    # Freeze all fields (tuples/Fractions/immutable).
    K_t = tuple(int(v) for v in classified["K_idx"])
    S_t = tuple(int(v) for v in classified["S_idx"])
    L_hat_t = tuple(float(v) for v in classified["L_hat_list"])
    stab_ax_t = tuple(classified["stab_axis"])
    stab_want_t = tuple(classified["stab_want"])
    return EstimatorPlan(
        rs_raw=rs_raw_t, rs_hat=rs_hat_t, n=int(n), k=int(k),
        K_idx=K_t, S_idx=S_t, L_hat_list=L_hat_t,
        stab_axis=stab_ax_t, stab_want=stab_want_t,
        N=int(N), delta=float(delta), channel=str(channel),
        g_round=g_t, b_cert=b_fresh, G_cert=G_fresh,
        bits=int(bits), den=int(den),
    )


def run_streaming(plan, measure_fn, rng: random.Random) -> dict:
    """Stream N copies with exact accumulation and rigorous ordered interval.

    ``plan`` must be a frozen ``EstimatorPlan`` from ``plan_estimator``
    (mutable dicts rejected to prevent false-certificate construction).
    At entry, BEFORE any callback/RNG consumption, the complete plan is
    revalidated against authoritative raw targets/N/delta/channel/exact
    score via one canonical ``plan_estimator`` reconstruction (fresh b/G
    exactly once, never per-copy, timed as ``revalidate_time_sec``),
    rejecting underclaimed certs and inconsistent derived
    n/k/rs_hat/K/S/L/axes/signs/channel/cert metadata; only the canonical
    snapshot is used. All fields are snapshotted into locals BEFORE the
    first ``measure_fn`` call, so a callback mutating external ``rs``/``g``
    lists or attempting ``plan.N=...`` cannot alter collection. Never
    re-solves/rebuilds the channel per copy (O(n+k) streaming).

    ``measure_fn(axes)->outcomes`` is called once per copy with a fresh axes
    list (correlated outcomes required; never simulate qubits independently).
    Exact ``Fraction`` accumulation with ALL-N denominator (flag failures
    count as 0). Interval endpoints are computed rationally as
    ``clip01(mean-R)``, ``clip01(mean+R)`` with proven upper radius ``R``
    (Hoeffding/Bernoulli) then converted outward (lower downward, upper
    upward), so endpoints are ordered in [0,1] even when the raw interval is
    disjoint (e.g. mean outside [0,1] due to G>1) or on the failure event.
    Returns both rational exact and float display values.
    """
    t0 = time.perf_counter()
    if not isinstance(plan, EstimatorPlan):
        raise ValueError("plan must be frozen EstimatorPlan from plan_estimator (dicts rejected)")
    if not callable(measure_fn):
        raise ValueError("measure_fn must be callable")
    if not hasattr(rng, "random"):
        raise ValueError("rng must have .random()")
    # Strict execution-boundary validation BEFORE any callback/RNG draw.
    # Do not trust public fields: check original types before any int/float
    # casts (bool/Fraction N must not silently become ints).
    if not isinstance(plan.rs_raw, tuple) or len(plan.rs_raw) == 0:
        raise ValueError("plan rs_raw must be nonempty tuple")
    for _r in plan.rs_raw:
        if not isinstance(_r, tuple) or len(_r) != 3:
            raise ValueError(f"plan rs_raw entries must be 3-tuples, got {_r!r}")
        for _v in _r:
            if _is_bool(_v) or not isinstance(_v, (int, float)):
                raise ValueError(f"plan rs_raw coords must be numbers (bool rejected), got {_v!r}")
            if not math.isfinite(float(_v)):
                raise ValueError(f"plan rs_raw coord nonfinite {_v!r}")
    if not isinstance(plan.rs_hat, tuple) or len(plan.rs_hat) != len(plan.rs_raw):
        raise ValueError("plan rs_hat must be tuple matching rs_raw")
    for _r in plan.rs_hat:
        if not isinstance(_r, tuple) or len(_r) != 3:
            raise ValueError(f"plan rs_hat entries must be 3-tuples, got {_r!r}")
        for _v in _r:
            if _is_bool(_v) or not isinstance(_v, (int, float)):
                raise ValueError(f"plan rs_hat coords must be numbers, got {_v!r}")
            if not math.isfinite(float(_v)):
                raise ValueError(f"plan rs_hat coord nonfinite {_v!r}")
    if _is_bool(plan.n) or not isinstance(plan.n, int):
        raise ValueError(f"plan n must be int (bool rejected), got {plan.n!r}")
    if _is_bool(plan.k) or not isinstance(plan.k, int):
        raise ValueError(f"plan k must be int, got {plan.k!r}")
    if _is_bool(plan.N) or not isinstance(plan.N, int):
        raise ValueError(f"plan N must be int (bool/Fraction rejected), got {type(plan.N).__name__}={plan.N!r}")
    if _is_bool(plan.delta) or not isinstance(plan.delta, (int, float)):
        raise ValueError(f"plan delta must be number in (0,1) (bool rejected), got {plan.delta!r}")
    if not isinstance(plan.channel, str):
        raise ValueError(f"plan channel must be str, got {plan.channel!r}")
    _require_channel(plan.channel)
    if not isinstance(plan.g_round, tuple):
        raise ValueError(f"plan g_round must be tuple (mutable list rejected), got {type(plan.g_round).__name__}")
    for _v in plan.g_round:
        if not isinstance(_v, Fraction):
            raise ValueError(f"plan g_round must be Fractions, got {type(_v).__name__}={_v!r}")
    if not isinstance(plan.b_cert, Fraction) or not isinstance(plan.G_cert, Fraction):
        raise ValueError("plan b_cert/G_cert must be Fractions")
    if _is_bool(plan.bits) or not isinstance(plan.bits, int):
        raise ValueError(f"plan bits must be int, got {plan.bits!r}")
    if _is_bool(plan.den) or not isinstance(plan.den, int):
        raise ValueError(f"plan den must be int, got {plan.den!r}")
    if not isinstance(plan.K_idx, tuple):
        raise ValueError("plan K_idx must be tuple (mutable list rejected)")
    for _v in plan.K_idx:
        if _is_bool(_v) or not isinstance(_v, int):
            raise ValueError(f"plan K_idx entries must be ints, got {_v!r}")
    if not isinstance(plan.S_idx, tuple):
        raise ValueError("plan S_idx must be tuple (mutable list rejected)")
    for _v in plan.S_idx:
        if _is_bool(_v) or not isinstance(_v, int):
            raise ValueError(f"plan S_idx entries must be ints, got {_v!r}")
    if not isinstance(plan.L_hat_list, tuple):
        raise ValueError("plan L_hat_list must be tuple")
    for _v in plan.L_hat_list:
        if _is_bool(_v) or not isinstance(_v, (int, float)):
            raise ValueError(f"plan L_hat entries must be numbers, got {_v!r}")
        if not math.isfinite(float(_v)):
            raise ValueError(f"plan L_hat nonfinite {_v!r}")
    if not isinstance(plan.stab_axis, tuple):
        raise ValueError("plan stab_axis must be tuple")
    for _v in plan.stab_axis:
        if not (_v is None or (isinstance(_v, str) and _v in PAULI_AXES)):
            raise ValueError(f"plan stab_axis entries must be X/Y/Z/None, got {_v!r}")
    if not isinstance(plan.stab_want, tuple):
        raise ValueError("plan stab_want must be tuple")
    for _v in plan.stab_want:
        if not (_v is None or (isinstance(_v, int) and not _is_bool(_v) and _v in (1, -1))):
            raise ValueError(f"plan stab_want entries must be +1/-1/None, got {_v!r}")
    # Canonical reconstruction exactly once (fresh b/G, validates target/N/
    # delta/channel/score, rejects underclaimed certs); no per-copy rebuild.
    t_r0 = time.perf_counter()
    canonical = plan_estimator(
        list(plan.rs_raw), plan.N, plan.delta, plan.channel,
        tuple(plan.g_round), plan.b_cert, plan.G_cert,
        bits=plan.bits, den=plan.den,
    )
    t_r1 = time.perf_counter()
    revalidate_time = t_r1 - t_r0
    # Reject inconsistent derived/metadata (zero callbacks/RNG draws so far).
    if plan.n != canonical.n:
        raise ValueError(f"plan n {plan.n!r} != canonical {canonical.n!r}")
    if plan.k != canonical.k:
        raise ValueError(f"plan k {plan.k!r} != canonical {canonical.k!r}")
    if plan.rs_raw != canonical.rs_raw:
        raise ValueError("plan rs_raw != canonical raw target")
    if plan.rs_hat != canonical.rs_hat:
        raise ValueError("plan rs_hat != canonical normalized target")
    if tuple(plan.K_idx) != canonical.K_idx:
        raise ValueError("plan K_idx != canonical")
    if tuple(plan.S_idx) != canonical.S_idx:
        raise ValueError("plan S_idx != canonical")
    if tuple(plan.L_hat_list) != canonical.L_hat_list:
        raise ValueError("plan L_hat_list != canonical")
    if tuple(plan.stab_axis) != canonical.stab_axis:
        raise ValueError("plan stab_axis != canonical")
    if tuple(plan.stab_want) != canonical.stab_want:
        raise ValueError("plan stab_want != canonical")
    if plan.N != canonical.N:
        raise ValueError("plan N != canonical")
    if float(plan.delta) != canonical.delta:
        raise ValueError("plan delta != canonical")
    if plan.channel != canonical.channel:
        raise ValueError("plan channel != canonical")
    if tuple(plan.g_round) != canonical.g_round:
        raise ValueError("plan g_round != canonical")
    if plan.b_cert != canonical.b_cert:
        raise ValueError(f"plan b_cert {plan.b_cert} != fresh {canonical.b_cert}")
    if plan.G_cert != canonical.G_cert:
        raise ValueError(f"plan G_cert {plan.G_cert} != fresh {canonical.G_cert}")
    if plan.bits != canonical.bits or plan.den != canonical.den:
        raise ValueError("plan bits/den != canonical")
    # Use ONLY the canonical snapshot for streaming.
    N = canonical.N
    n = canonical.n
    k = canonical.k
    channel = canonical.channel
    g_round = canonical.g_round
    b_cert = canonical.b_cert
    G_cert = canonical.G_cert
    delta = canonical.delta
    rs_hat_snap = canonical.rs_hat
    K_snap = canonical.K_idx
    S_snap = canonical.S_idx
    L_hat_snap = canonical.L_hat_list
    ax_snap = canonical.stab_axis
    want_snap = canonical.stab_want
    snap_design = {"rs": [tuple(r) for r in canonical.rs_raw], "rs_hat": [tuple(r) for r in rs_hat_snap],
                   "n": n, "k": k, "K_idx": list(K_snap), "S_idx": list(S_snap),
                   "L_hat_list": list(L_hat_snap),
                   "stab_axis": list(ax_snap), "stab_want": list(want_snap)}
    total_frac = Fraction(0, 1)
    n_flag1 = 0
    weight_counts = [0] * (k + 1)
    for _t in range(N):
        axes = sample_axes_for_copy(snap_design, rng)
        # Pass a copy so callee mutation cannot affect our axes.
        outcomes = measure_fn(list(axes))
        _validate_outcomes(outcomes, n)
        scored = score_one_copy(snap_design, g_round, axes, outcomes, rng, channel)
        H: Fraction = scored["H"]
        w = scored["w"]
        flag = scored["flag"]
        if flag == 1:
            n_flag1 += 1
        if flag == 1 and 0 <= w <= k:
            weight_counts[w] += 1
        total_frac += H
    mean_exact = total_frac / N
    est = clipped_estimate(float(mean_exact))
    # Rigorous radius (rational upper + outward display).
    if k == 0:
        R_frac = _bernoulli_radius_frac(N, delta)
    else:
        R_frac = _hoeffding_radius_frac(b_cert, G_cert, N, delta)
    R_float = _float_upper(R_frac)
    # Rational endpoints clipped (monotone, preserves order even if disjoint).
    lo_rat = _clip01_frac(mean_exact - R_frac)
    hi_rat = _clip01_frac(mean_exact + R_frac)
    # Outward float conversion (lo downward, hi upward); stays in [0,1] and ordered.
    # Rational clip preserves order even when the raw interval is disjoint
    # (mean outside [0,1] due to G>1); outward conversion preserves it.
    lo = _float_lower(lo_rat)
    hi = _float_upper(hi_rat)
    # Clamp for safety (should already be in [0,1]).
    if lo < 0.0:
        lo = 0.0
    if hi > 1.0:
        hi = 1.0
    if lo > hi:
        # Unreachable by construction (lo_rat<=hi_rat and outward maps
        # preserve order). Fail loudly rather than returning an unsupported
        # zero-width confidence claim.
        raise RuntimeError("outward interval inversion (unreachable bug)")
    t1 = time.perf_counter()
    return {
        "N": N, "n": n, "k": k, "channel": channel,
        "n_measurements": n * N,
        "mean_exact": str(mean_exact), "mean_exact_float": float(mean_exact),
        "estimate": est, "radius": R_float, "radius_exact": str(R_frac),
        "radius_exact_float": float(R_frac) if R_frac < Fraction(10**308) else math.inf,
        "interval": [lo, hi], "interval_exact": [str(lo_rat), str(hi_rat)],
        "b_cert": str(b_cert), "G_cert": str(G_cert),
        "n_flag1": n_flag1, "weight_counts_flag1": weight_counts,
        "stream_time_sec": t1 - t0,
        "revalidate_time_sec": revalidate_time,
    }


# --- high-level solve+certify for one (k,N,channel,delta) --------------------

def solve_and_certify(k: int, N: int, channel: str = DEFAULT_CHANNEL, *,
                      delta: float = DELTA_DEFAULT, bits: int = ENCLOSURE_BITS,
                      den: int = ROUND_DEN, options: dict | None = None,
                      eps: float = 0.05) -> dict:
    """End-to-end LP solve + independent exact certification (any N, delta).

    Args:
      k: noisy count >=0. N: any ordinary positive int (e.g. 3,17,359; not
        only perfect squares). channel: "common" (default, theorem) or
        "degraded_half". delta: failure level in (0,1) (default 1/3,
        configurable). bits/den/options/eps: enclosure, rounding, solver,
        accuracy for sufficient copies.

    Returns comprehensive record with timings, resources, versions, channel
    data, solver record, certificate, rigorous surrogate (exact for squares,
    proven upper otherwise, labelled), rigorous radius (exact+display),
    sufficient copies (exact ceil of upper), cond, bytes, bit lengths. k=0
    exact path (no solver, g=[1], b=0, G=1; sufficient 359 at eps0/delta0).
    Raises on failure. Frozen 56-grid (k=0..6, N=1,4,16,256) retained for
    validation; no broader benchmark.
    """
    t_all0 = time.perf_counter()
    k = _check_k(k)
    N = _require_positive_int("N", N)
    _require_channel(channel)
    delta = _require_delta(delta)
    _require_eps(eps)
    if _is_bool(bits) or not isinstance(bits, int) or bits <= 0:
        raise ValueError("bits must be positive int")
    # channel build
    t_c0 = time.perf_counter()
    import numpy as np

    if channel == CHANNEL_HALF:
        W_frac, W_float = build_channel_half(k)
        eta_info = None
        W_mid = None
    else:
        if k == 0:
            W_frac = [[Fraction(1, 1)]]
            W_float = np.ones((1, 1), dtype=np.float64)
            eta_info = eta_interval_common(bits)
            W_mid = [[Fraction(1, 1)]]
        else:
            W_float = build_channel_common_float(k)
            eta_info = eta_interval_common(bits)
            W_mid = build_channel_mid(k, eta_info["eta_mid"])
            W_frac = None  # type: ignore (no single exact true W)
    t_c1 = time.perf_counter()
    # solve
    t_s0 = time.perf_counter()
    if k == 0:
        sol = solve_lp(np.ones((1, 1), dtype=np.float64), N, options=options or {})
        g_float = [1.0]
        # override solver record for exact path clarity (solve_lp already exact)
    else:
        sol = solve_lp(W_float, N, options=options)
        g_float = sol["g_float"]
    t_s1 = time.perf_counter()
    # certify from executed rounds (private low-level helpers on fresh W)
    t_v0 = time.perf_counter()
    if channel == CHANNEL_HALF:
        if W_frac is None:
            raise RuntimeError("missing half channel (unreachable)")
        cert = _certify_half(k, W_frac, g_float, den)
        b_cert = Fraction(cert["b_cert"])
        G_cert = Fraction(cert["G_cert"])
        g_round = [Fraction(s) for s in cert["g_round"]]
    else:
        if eta_info is None or W_mid is None:
            raise RuntimeError("missing common channel (unreachable)")
        if k == 0:
            g_raw, g_round, maxdiff = round_coefficients([1.0], den)
            cert = {
                "channel": channel, "k": 0, "den": den,
                "sqrt3_lo": str(eta_info["sqrt3_lo"]), "sqrt3_hi": str(eta_info["sqrt3_hi"]),
                "eta_lo": str(eta_info["eta_lo"]), "eta_hi": str(eta_info["eta_hi"]),
                "eta_mid": str(eta_info["eta_mid"]),
                "max_eta_error": str(eta_info["max_eta_error"]),
                "max_eta_error_float": float(eta_info["max_eta_error"]),
                "bits": bits,
                "g_raw_exact": ["1"], "g_round": ["1"], "g_round_float": [1.0],
                "maxdiff": "0", "maxdiff_float": 0.0,
                "b_raw_mid": "0", "b_raw_mid_float": 0.0,
                "b_round_mid": "0", "b_round_mid_float": 0.0,
                "G_raw": "1", "G_raw_float": 1.0, "G_round": "1", "G_round_float": 1.0,
                "residuals_raw_mid": ["0"], "residuals_round_mid": ["0"],
                "hstar_raw": 0, "hstar_round": 0,
                "coupling": "0", "coupling_float": 0.0,
                "b_cert": "0", "b_cert_float": 0.0, "G_cert": "1",
                "cert_ok": True, "cert_time_sec": 0.0,
            }
            b_cert = Fraction(0, 1)
            G_cert = Fraction(1, 1)
            g_round = [Fraction(1, 1)]
        else:
            cert = _certify_common(k, W_mid, eta_info, g_float, den)
            b_cert = Fraction(cert["b_cert"])
            G_cert = Fraction(cert["G_cert"])
            g_round = [Fraction(s) for s in cert["g_round"]]
    t_v1 = time.perf_counter()
    # Rigorous confidence for any N and configurable delta (no square assert).
    surr_info = surrogate_bounds(b_cert, G_cert, N, SURROGATE_BITS)
    surr_exact = surr_info["upper"]
    surr_is_exact = surr_info["is_exact"]
    if k == 0:
        R_frac = _bernoulli_radius_frac(N, delta)
        rad = _float_upper(R_frac)
        need, reason = sufficient_copies_bernoulli(Fraction(0, 1), delta, eps)
        # Bernoulli 359 at eps0/delta0; generic Hoeffding would give 1434 (loose).
    else:
        R_frac = _hoeffding_radius_frac(b_cert, G_cert, N, delta)
        rad = _float_upper(R_frac)
        need, reason = sufficient_copies_conservative(b_cert, G_cert, delta, eps)
    # resources: cond, bytes, bitlens
    try:
        cond = float(np.linalg.cond(W_float)) if W_float.size else float("nan")
    except Exception:
        cond = float("nan")
    W_bytes = int(W_float.nbytes)
    # A_ub bytes
    try:
        _c, _A, _b, _bd = assemble_lp(W_float, N)
        Aub_bytes = int(_A.nbytes + _b.nbytes + _c.nbytes)
    except Exception:
        Aub_bytes = 0
    bitlens = [(v.numerator.bit_length(), v.denominator.bit_length()) for v in g_round]
    max_num = max(b[0] for b in bitlens) if bitlens else 0
    max_den = max(b[1] for b in bitlens) if bitlens else 0
    t_all1 = time.perf_counter()
    # memory: current tracemalloc if tracing, else None with caveat
    if tracemalloc.is_tracing():
        cur, peak = tracemalloc.get_traced_memory()
    else:
        cur, peak = None, None
    return {
        "k": k, "N": N, "channel": channel, "delta": delta, "eps": eps,
        "bits": bits, "den": den,
        "W_float": W_float.tolist(),
        "W_float_bytes": W_bytes,
        "Aub_bytes": Aub_bytes,
        "cond": cond,
        "solver": sol,
        "certificate": cert,
        "g_round": [str(v) for v in g_round],
        "g_round_float": [float(v) for v in g_round],
        "b_cert": str(b_cert), "b_cert_float": float(b_cert),
        "G_cert": str(G_cert), "G_cert_float": float(G_cert),
        "surrogate_exact": str(surr_exact), "surrogate_float": _float_upper(surr_exact),
        "surrogate_is_exact": bool(surr_is_exact), "surrogate_symbolic": "b+G/sqrt(N)",
        "radius": rad, "radius_exact": str(R_frac),
        "sufficient_copies": need, "sufficient_reason": reason,
        "bit_lengths": bitlens, "max_bitlen_num": max_num, "max_bitlen_den": max_den,
        "timings": {
            "channel_sec": t_c1 - t_c0, "solve_sec": t_s1 - t_s0,
            "cert_sec": t_v1 - t_v0, "total_sec": t_all1 - t_all0,
        },
        "memory": {
            "tracemalloc_current": cur, "tracemalloc_peak": peak,
            "caveat": "Python allocations only; HiGHS native allocations excluded",
        },
        "eta_info": ({k2: str(v) if isinstance(v, Fraction) else v for k2, v in eta_info.items()} if eta_info else None),
    }


# --- demo backend (real iid R-diagonal mixture, testable) --------------------

def born_plus_prob(r_axis: float, s: int) -> float:
    """Born Pr[outcome=+1 | R-eigenstate sign s, requested Pauli axis].

    For qubit state ``tau_s=(I+sR)/2`` with ``R=r_hat.(X,Y,Z)`` and projector
    ``(I+o P_a)/2``, ``Pr[o|s,a]=(1+s*o*r_a)/2`` so ``Pr[+1]=(1+s*r_a)/2``
    with ``r_a`` the ``r_hat`` component along ``a`` and ``s=+1/-1`` for
    latent ``x=0/1``. Validates finite ``r_a`` in [-1,1] (tiny 1e-12
    overshoot clipped as float proxy) and ``s``. Pure arithmetic, no RNG.
    """
    if s not in (1, -1):
        raise ValueError(f"s must be +1/-1, got {s!r}")
    if _is_bool(r_axis) or not isinstance(r_axis, (int, float)):
        raise ValueError(f"r_axis must be number, got {r_axis!r}")
    r = float(r_axis)
    if not math.isfinite(r):
        raise ValueError(f"r_axis nonfinite {r_axis!r}")
    if r < -1.0 - 1e-12 or r > 1.0 + 1e-12:
        raise ValueError(f"r_axis out of [-1,1]: {r}")
    if r < -1.0:
        r = -1.0
    if r > 1.0:
        r = 1.0
    p = (1.0 + s * r) / 2.0
    if p < 0.0:
        p = 0.0 if p > -1e-12 else 0.0
    if p > 1.0:
        p = 1.0 if p < 1.0 + 1e-12 else 1.0
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Born prob out of range {p}")
    return p


def make_demo_mixture_measure(rs_hat_list, backend_rng, p_target: float = 0.7):
    """Return full-copy ``measure_fn(axes)->outcomes`` for the demo mixture.

    Iid R-diagonal mixture: with prob ``p_target`` latent ``x=0^n``
    (all-target ``s=+1``) else ``x=1^n`` (all-opposite ``s=-1``); one
    full-copy latent draw per copy (correlated across qubits), then
    conditional per-qubit Born outcomes via ``born_plus_prob`` with the
    exact-requested axes and ``rs_hat`` floats (which approximate exact
    ``r/||r||``). Uses ONLY ``backend_rng`` (independent reproducible
    ``random.Random`` stream, separate from design RNG); never simulates
    qubits independently of the latent. True fidelity to the all-target
    projector is ``p_target`` (opposite component orthogonal).
    """
    if not isinstance(rs_hat_list, (list, tuple)) or len(rs_hat_list) == 0:
        raise ValueError("rs_hat_list must be nonempty")
    hats = []
    for _r in rs_hat_list:
        if not isinstance(_r, (list, tuple)) or len(_r) != 3:
            raise ValueError(f"rs_hat entries must be 3-tuples, got {_r!r}")
        hats.append((float(_r[0]), float(_r[1]), float(_r[2])))
    if not hasattr(backend_rng, "random"):
        raise ValueError("backend_rng must have .random()")
    if _is_bool(p_target) or not isinstance(p_target, (int, float)):
        raise ValueError(f"p_target must be number, got {p_target!r}")
    pt = float(p_target)
    if not math.isfinite(pt) or not (0.0 < pt < 1.0):
        raise ValueError(f"p_target must be in (0,1), got {p_target!r}")
    n = len(hats)

    def measure(axes):
        _validate_axes(axes, n)
        u0 = _require_rng_draw(backend_rng.random())
        s = 1 if u0 < pt else -1
        out: list[int] = []
        for j, ax in enumerate(axes):
            idx = {"X": 0, "Y": 1, "Z": 2}[ax]
            p_plus = born_plus_prob(hats[j][idx], s)
            u = _require_rng_draw(backend_rng.random())
            out.append(1 if u < p_plus else -1)
        _validate_outcomes(out, n)
        return out

    measure.p_target = pt  # type: ignore
    measure.true_fidelity = pt  # type: ignore
    return measure


# --- demo -------------------------------------------------------------------

def demo(seed: int = 12345) -> int:
    """Runnable example: heterogeneous n=3 (k=2 noisy +1 stabilizer), both channels.

    Solves LP at N=16, builds plan, streams N copies from a REAL iid
    correlated R-diagonal mixture (0.7 all-target + 0.3 all-opposite,
    full-copy latent draw, conditional per-qubit Born outcomes with r_hat
    and requested axes; independent backend stream), prints stated mixture,
    true fidelity, and estimates/intervals as illustration only.
    Deterministic given seed; Monte Carlo illustrates execution only
    (not a mathematical gate).
    """
    print("=== product_state_estimator demo  ===")
    print(f"python {platform.python_version()}")
    try:
        import numpy, scipy
        print(f"numpy {numpy.__version__} scipy {scipy.__version__}")
    except Exception as e:
        print(f"NumPy/SciPy import failed: {e}")
        return 1
    import math as _m
    s3 = _m.sqrt(3.0)
    # heterogeneous target: 111 (noisy), (1/sqrt2,1/sqrt2,0) (noisy), +Z (stabilizer)
    rs = [(1 / s3, 1 / s3, 1 / s3), (1 / _m.sqrt(2.0), 1 / _m.sqrt(2.0), 0.0), (0.0, 0.0, 1.0)]
    p_target = 0.7
    true_fid = 0.7
    print(f"target rs={rs}")
    print(f"demo observation model: iid R-diagonal mixture {p_target} all-target (x=0^n) + 0.3 all-opposite (x=1^n); true fidelity={true_fid} (illustration only)")
    classified = classify_target(rs)
    print(f"n={classified['n']} k={classified['k']} K={classified['K_idx']} S={classified['S_idx']}")
    for ch in (CHANNEL_COMMON, CHANNEL_HALF):
        print(f"\n--- channel {ch} ---")
        rec = solve_and_certify(classified["k"], 16, ch)
        print(f"solver: {rec['solver']['method']} status={rec['solver']['status']} nit={rec['solver']['nit']} viol={rec['solver'].get('max_constraint_violation')}")
        print(f"g_round={rec['g_round']} b_cert={rec['b_cert']} G_cert={rec['G_cert']}")
        print(f"surrogate={rec['surrogate_exact']} (exact={rec['surrogate_is_exact']}) radius={rec['radius']:.6f} cond={rec['cond']:.3g} Wbytes={rec['W_float_bytes']}")
        print(f"represented rs_hat (float approx to exact r/||r||)={classify_target(rs)['rs_hat']}")
        design_rng = make_rng(seed)
        backend_rng = random.Random(seed + 7919)
        hats = classify_target(rs)["rs_hat"]
        measure = make_demo_mixture_measure(hats, backend_rng, p_target)
        # Fresh binding: plan recomputes b/G; caller claims validated (pass-through here).
        plan = plan_estimator(rs, 16, DELTA_DEFAULT, ch,
                              [Fraction(s) for s in rec["g_round"]],
                              Fraction(rec["b_cert"]), Fraction(rec["G_cert"]))
        out = run_streaming(plan, measure, design_rng)
        print(f"stream: est={out['estimate']:.4f} interval={out['interval']} radius_exact={out['radius_exact']} flag1={out['n_flag1']}/{out['N']} reval_ms={out['revalidate_time_sec']*1e3:.3f} stream_ms={out['stream_time_sec']*1e3:.2f}ms (true fid {true_fid} illustration only)")
    print("\nFrozen design: N,delta,score fixed before data; no stopping/reuse.")
    print("Certificates imply arbitrary-state bias via R-diagonal reduction (see docs/REPRODUCIBILITY.md).")
    print("Validity (bias/range) certified; LP optimality NOT claimed.")
    print("=== demo done ===")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bounded certified scalable estimator (frozen k<=6 validation)")
    ap.add_argument("--demo", action="store_true", help="run runnable demo")
    ap.add_argument("--k", type=int, default=None, help="single k to solve (default: demo grid subset)")
    ap.add_argument("--N", type=int, default=None, help="single N, any positive int (e.g. 3,17,359)")
    ap.add_argument("--channel", type=str, default=DEFAULT_CHANNEL, choices=list(ALLOWED_CHANNELS))
    ap.add_argument("--delta", type=float, default=DELTA_DEFAULT, help="failure level in (0,1)")
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args(argv)
    if args.demo or (args.k is None):
        return demo(seed=args.seed)
    k = args.k
    N = args.N if args.N is not None else 16
    rec = solve_and_certify(k, N, args.channel, delta=args.delta)
    import json as _json
    print(_json.dumps(rec, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
