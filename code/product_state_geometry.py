"""Bounded partial Pauli approximation with certified geometry.

Certified partial Pauli approximation for heterogeneous targets. 
 No new classical theorem claimed.

Model (same as certified estimator, docs/REPRODUCIBILITY.md):
  target P = tensor_j tau^j_+ with tau^j_+ = (I + R^j)/2,
  R^j = r_hat^j.(X,Y,Z), each r_hat^j unit (pure). n total qubits,
  k = #non-Pauli factors (exact raw-axis rule), N copies. B=109 median
  conversion is the prior fixed-error lower transfer (fixed-error conversion);
  geometry itself uses no B. eps0=1/20, delta0=1/3 reference.
  Unknown arbitrary i.i.d. mixed sigma (worst case). Per copy: product Pauli
  basis, one measurement per qubit, full n outcome bits. Across-copy and
  within-copy adaptivity allowed in lower bounds; upper uses nonadaptive
  weighted sampling plus exact stabilizer measurement (via accepted
  estimator). Forbidden: entangling preprocessing, non-Pauli axes, repeat
  measurement within copy. Classical postprocessing arbitrary for lowers.

Target convention (identical to certified estimator):
  Each input triple r is a finite binary-rational direction. The represented
  pure factor is the mathematically normalized direction r_hat = r/||r||_2
  exactly (real division, generally irrational). Ideal axis-choice,
  degradation-coin, and Born probabilities refer to these exact r_hat.
  Numerical sampling values are computed consistently from normalized
  quantities (see product_state_estimator.py). Finite float rounding of those
  probabilities and finite RNG/control errors remain OUTSIDE all certificates;
  no hardware guarantee is claimed. Inputs must satisfy | ||r||_2 - 1 | <=1e-9
  (float norm, gates acceptance only); zero vectors rejected. Pauli
  classification uses original exact signed axes on RAW inputs (two coords
  exactly 0.0, third exactly +1.0/-1.0); all other accepted vectors are noisy
  (safe side). Near-axis genuinely non-axis directions are never stabilizers.

Partial replacement (proven in docs/REPRODUCIBILITY.md):
  For deterministic subset D (before data), define nearest signed Pauli axis
  per factor j in D: axis i^*_j attaining a_j = max_i |r_hat_{ji}| with
  sign matching r, pi^j_+ = (I + sign*P_{i^*})/2. Proxy
  P_D = (tensor_{j not in D} tau^j_+) tensor (tensor_{j in D} pi^j_+).
  Both P and P_D are rank-1 projectors. Overlap
  c_D = Tr(P P_D) = prod_{j in D} (1+a_j)/2 exactly, operator bias
  d_D = ||P - P_D||_op = sqrt(1 - c_D) exactly (rank-two identity).
  For every sigma, |Tr(sigma P) - Tr(sigma P_D)| <= d_D (Holder).
  Hence any certified estimator for P_D with bias b_inner and range G_inner
  transfers to P with bias <= b_inner + d_D and same range; all-copy
  stabilizer flags preserved (proxy has MORE stabilizers than the original,
  flags are exact Pauli matches, still all-copy denominator via accepted
  estimator). Geometry/plan selection is before
  data, so no multiple-selection confidence penalty (no union over D).
  This is a SUFFICIENT upper, not minimax optimum (discarding may be
  suboptimal; retaining noisy factors may be better for some sigma).

Rigorous enclosures (no floating tolerance alone certifies geometry):
  Exact Fraction conversion of binary-rational inputs (Fraction(float)
  exact), integer-square-root rational enclosures with explicit bits
  parameter, outward floats via exact Fraction(float) comparison plus
  nextafter (same logic as certified estimator, reimplemented here for
  self-containment). a_j = max|r_ji|/sqrt(sum r_ji^2) enclosed via
  sqrt(s2) interval; product overlap c_D via monotone products with bounded
  outward rounding to SCALE=2**bits after each factor (floor for lower,
  ceil for upper; denominators divide SCALE; each endpoint's additional
  rounding error relative to its exact (infinite-precision) endpoint is
  <=|D|/SCALE, so total interval-width additional rounding is
  <=2|D|/SCALE plus input a-enclosure propagation; sound by induction on
  [0,1]); d_D via sqrt(1-c) interval. No nextafter-alone or float
  tolerance certifies accumulated products. Geometry certificate fields
  (a/c/d, product_scale, product_rounding_bound) have denominators
  dividing SCALE (or small exact values for empty D), so their str()
  serialization stays below the Python digit limit for every allowed
  n/bits combination without changing any process-global integer
  safeguard. Inner estimator plans, user-supplied scores, mean/radius
  totals involving N, and arbitrary huge user Fractions or enormous N are
  NOT covered by this geometry guarantee; certified estimator resource
  restrictions still apply to inner plans.
  Handles exact axes (singleton [1,1]), near-axis tiny transverse (interval
  strictly below 1 on lower edge), signs/ties (smallest axis X<Y<Z, sign
  matching), empty D (c=1,d=0), all D, and all invalid inputs
  (duplicate/out-of-range/bool indices, invalid precision, invalid
  directions). Resource limits: n <= MAX_N (1024), bits in
  [1, MAX_GEOMETRY_BITS] (1024); larger requests raise ValueError (safe,
  documented). Certificate cost is O(n) sqrt intervals plus O(|D|)
  bounded products; harness covers n=200 default-bit regression plus
  high-precision boundary (n=200 bits=1024 tighter than bits=80); no broad
  benchmark.

Selection (deterministic, before data):
  select_discards(rs, beta, bits) sorts factors by certified proximity
  (midpoint (a_lo+a_hi)/2 descending, tie by index) and returns the longest
  prefix with certified upper d_hi <= beta (exact Fraction comparison).
  Implementation rebuilds each prefix with a fresh geometry_certificate call
  (not an incremental product update), so cost is quadratic in n: O(n)
  certificates each O(n) work plus O(n) classifications. At n=50 this is
  under one second; at n=200 it is several seconds; at MAX_N=1024 callers
  must budget quadratic time. beta must satisfy 0 <= beta < 1 (bool
  rejected). Exact Pauli factors (a=1) sort first and are always feasible
  (d=0), including at beta=0. Optimality claimed ONLY for exact costs with
  equal per-factor value (cheapest-first maximizes cardinality under
  sum-cost; see theory Sec 2). With conservative enclosures the order may
  invert within width and the prefix may underselect (safe side, still
  valid, only less efficient). Exposes chosen indices (sorted ascending
  canonical plus proximity order), original vs retained noisy counts (via
  classifier), bias interval, proxy target, arithmetic/resource
  metadata including measured select_time_sec.

Frozen plan + streaming (uses UNCHANGED certified estimator via public API):
  plan_geometry(rs, D, N, delta, channel, g_round, b_cert, G_cert, bits, den)
  builds proxy P_D, computes fresh geometry (a/c/d) at bits, builds inner
  EstimatorPlan for proxy via product_state_estimator.plan_estimator
  (fresh b/G, rejects understated/wrong-channel, k'=0 requires [1]), and
  freezes GeometryPlan (tuples/Fractions, deep-copied). g_round may be None
  to auto-solve via solve_and_certify(k',N,channel,delta) at defaults.
  run_geometry_streaming(plan, measure_fn, rng) revalidates original target,
  subset, proxy, geometry certificate, and inner-plan binding BEFORE any
  callback/RNG (zero draws), rejecting copied/tampered plans
  (dataclasses.replace), understated geometry, or wrong proxy; does not
  trust cached derived fields (recomputes n/k/rs_hat/K/S/L/axes/signs).
  Then delegates to run_streaming for exactly N fresh copies,
  adds geometric bias upper to inner exact radius (R_total = R_inner + d_hi
  as Fractions), and produces outward clipped endpoints from exact raw mean
  (clip01(mean-R_total), clip01(mean+R_total) rationally, then outward
  floats). Clearly separates proxy fidelity estimate (inner mean/interval
  for P_D) from confidence for original target (widened interval for P).
  Default common channel; half explicit. Ideal-probability + fresh-iid
  assumptions load-bearing; no hardware guarantee.

Provenance: PSW17 primary PDF checked on 2026-09-08 (printed p5
  Theorem 3, p20 Proposition 14); invoked, not reproved. B=109 prior
  fixed-error conversion reused for heterogeneous core bounds (see theory
  file). No new classical theorem. Homogeneous gap explicitly unresolved.

Run demo: python code/product_state_geometry.py --demo
Requires NumPy/SciPy in .venv for inner solves (see requirements-estimator.txt).
Exits nonzero on any failure.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import math
import platform
import random
import sys
import time
import tracemalloc
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import product_state_estimator as est

# --- constants ------------------------------------------------------------

GEOMETRY_BITS_DEFAULT = 80
GEOMETRY_BITS_MIN = 1
GEOMETRY_BITS_MAX = 1024
MAX_N = 1024
PAULI_AXES = ("X", "Y", "Z")
CHANNEL_COMMON = est.CHANNEL_COMMON
CHANNEL_HALF = est.CHANNEL_HALF
ALLOWED_CHANNELS = est.ALLOWED_CHANNELS
DEFAULT_CHANNEL = est.DEFAULT_CHANNEL
ROUND_DEN_DEFAULT = est.ROUND_DEN

EPS0_FRAC = Fraction(1, 20)


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


def _require_channel(ch: str) -> str:
    if not isinstance(ch, str) or ch not in ALLOWED_CHANNELS:
        raise ValueError(f"channel must be one of {ALLOWED_CHANNELS}, got {ch!r}")
    return ch


def _require_delta(delta) -> float:
    if _is_bool(delta) or not isinstance(delta, (int, float)):
        raise ValueError(f"delta must be number in (0,1), got {delta!r}")
    f = float(delta)
    if not math.isfinite(f) or not (0.0 < f < 1.0):
        raise ValueError(f"delta must satisfy 0<delta<1, got {delta!r}")
    return f


def _require_beta(beta) -> Fraction:
    """Validate geometry budget 0 <= beta < 1, return exact Fraction."""
    if _is_bool(beta):
        raise ValueError(f"beta must be number in [0,1), got {beta!r}")
    if isinstance(beta, Fraction):
        bf = beta
        # Fraction is exact; check range. Denominator zero impossible.
        if bf < 0 or bf >= 1:
            raise ValueError(f"beta must satisfy 0<=beta<1, got {beta!r}")
        return bf
    if not isinstance(beta, (int, float)):
        raise ValueError(f"beta must be number in [0,1), got {type(beta).__name__}={beta!r}")
    f = float(beta)
    if not math.isfinite(f) or not (0.0 <= f < 1.0):
        raise ValueError(f"beta must satisfy 0<=beta<1, got {beta!r}")
    # Exact binary conversion (same convention as estimator eps handling).
    return Fraction(f)


def _require_bits(bits) -> int:
    if _is_bool(bits) or not isinstance(bits, int):
        raise ValueError(f"bits must be int (bool rejected), got {type(bits).__name__}={bits!r}")
    if bits < GEOMETRY_BITS_MIN or bits > GEOMETRY_BITS_MAX:
        raise ValueError(
            f"bits must satisfy {GEOMETRY_BITS_MIN}<=bits<={GEOMETRY_BITS_MAX} "
            f"(resource limit, documented), got {bits}"
        )
    return bits


def _require_den(den) -> int:
    if _is_bool(den) or not isinstance(den, int) or den <= 0:
        raise ValueError(f"den must be positive int, got {den!r}")
    return den


def _coord_to_fraction(v) -> Fraction:
    """Exact Fraction of one binary-rational coordinate (int/float, bool rejected)."""
    if _is_bool(v):
        raise ValueError(f"coordinate must be number (bool rejected), got {v!r}")
    if isinstance(v, int):
        return Fraction(v, 1)
    if isinstance(v, float):
        if not math.isfinite(v):
            raise ValueError(f"coordinate nonfinite {v!r}")
        return Fraction(v)  # exact binary
    raise ValueError(f"coordinate must be int/float, got {type(v).__name__}={v!r}")


def _validate_subset(D, n: int) -> tuple[int, ...]:
    """Validate discard subset D for n qubits, return sorted ascending canonical tuple.

    D must be list/tuple (empty allowed) of ints (bool rejected), each
    0<=j<n, no duplicates. Sorted ascending for canonical binding; order of
    input does not matter (set semantics, documented). Raises ValueError.
    """
    if not isinstance(D, (list, tuple)):
        raise ValueError(f"D must be list/tuple of ints, got {type(D).__name__}={D!r}")
    if _is_bool(n) or not isinstance(n, int) or n <= 0:
        raise ValueError(f"n must be positive int, got {n!r}")
    seen: set[int] = set()
    out: list[int] = []
    for v in D:
        if _is_bool(v) or not isinstance(v, int):
            raise ValueError(f"D entries must be ints (bool rejected), got {v!r}")
        if v < 0 or v >= n:
            raise ValueError(f"D index out of range [0,{n}): {v}")
        if v in seen:
            raise ValueError(f"D contains duplicate index {v}")
        seen.add(v)
        out.append(v)
    out.sort()
    return tuple(out)


def _validate_rs(rs) -> list[tuple[float, float, float]]:
    """Validate directions via certified estimator (identical acceptance).

    Returns validated raw list. Raises ValueError on empty/wrong dims/bool/
    nonfinite/zero/far-from-unit. Also enforces MAX_N resource limit.
    """
    # Delegate acceptance to unchanged estimator for identical gating.
    vrs = est.validate_target(rs)
    if len(vrs) > MAX_N:
        raise ValueError(f"n={len(vrs)} exceeds resource limit MAX_N={MAX_N} (documented)")
    return vrs


# --- rigorous sqrt + outward floats ----------------------------------------

def _sqrt_interval(s: Fraction, bits: int) -> tuple[Fraction, Fraction]:
    """Rigorous [lo,hi] enclosing sqrt(s) with width <= 2**-bits (or exact).

    s must be Fraction >= 0. bits validated in range. If s==0 returns (0,0).
    If s is a perfect-square rational (isqrt(p*q)^2==p*q) returns exact
    singleton (t/q,t/q). Otherwise SCALE=2**bits, Q=(p*SCALE^2)//q,
    t=isqrt(Q), lo=t/SCALE, hi=(t+1)/SCALE, verified lo*lo<=s<hi*hi.
    Pure integer/Fraction arithmetic. Raises ValueError for s<0/bits invalid.
    """
    if not isinstance(s, Fraction):
        raise ValueError(f"s must be Fraction, got {type(s).__name__}")
    bits = _require_bits(bits)
    if s < 0:
        raise ValueError(f"sqrt of negative {s}")
    if s == 0:
        return (Fraction(0, 1), Fraction(0, 1))
    p, q = s.numerator, s.denominator  # q>0, p>0 since s>0
    if p <= 0 or q <= 0:
        raise RuntimeError("unexpected Fraction sign (unreachable)")
    t0 = math.isqrt(p * q)
    if t0 * t0 == p * q:
        exact = Fraction(t0, q)
        return (exact, exact)
    scale = 1 << bits
    # Q = floor(p*scale^2 / q)
    Q = (p * scale * scale) // q
    t = math.isqrt(Q)
    lo = Fraction(t, scale)
    hi = Fraction(t + 1, scale)
    # Verify enclosure (integer-exact, raises if violated).
    if not (lo * lo <= s < hi * hi):
        raise RuntimeError(f"sqrt enclosure violated for s={s} bits={bits}: lo={lo} hi={hi}")
    return (lo, hi)


def _float_upper(q: Fraction) -> float:
    """Smallest binary64 float >= q (inf if beyond max). Same logic as estimator."""
    if not isinstance(q, Fraction):
        raise ValueError("q must be Fraction")
    try:
        f = float(q)
    except OverflowError:
        if q > 0:
            return math.inf
        return -sys.float_info.max
    if math.isnan(f):
        raise RuntimeError("float conversion gave NaN (unreachable)")
    if math.isinf(f):
        if f > 0:
            return math.inf
        return -sys.float_info.max
    fq = Fraction(f)
    if fq >= q:
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
    for _ in range(1000):
        f_up = math.nextafter(f, math.inf)
        if math.isinf(f_up):
            return math.inf
        if Fraction(f_up) >= q:
            return f_up
        f = f_up
    raise RuntimeError("float upper did not converge (unreachable)")


def _float_lower(q: Fraction) -> float:
    """Largest binary64 float <= q (-inf if below min). Mirror of _float_upper."""
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
    if x < 0:
        return Fraction(0, 1)
    if x > 1:
        return Fraction(1, 1)
    return x


# --- per-factor geometry ----------------------------------------------------

def _a_interval_for_factor(r_raw: tuple[float, float, float], bits: int) -> dict:
    """Rigorous [a_lo,a_hi] for a=max|r_hat| with r_hat=r/||r|| exactly.

    r_raw validated 3-tuple floats. Exact Fractions via _coord_to_fraction.
    s2=sum f^2 exact; sqrt interval via _sqrt_interval(s2,bits); then
    a_lo=maxAbs/sqrt_hi, a_hi=maxAbs/sqrt_lo (or 1 if sqrt_lo==0, still valid
    since true<=1). Clipped upper to 1 (valid, tighter). For exact axes
    (e.g. (0,0,1)) returns singleton [1,1]. Returns dict with Fractions.
    """
    bits = _require_bits(bits)
    if not isinstance(r_raw, (list, tuple)) or len(r_raw) != 3:
        raise ValueError(f"r must be length-3, got {r_raw!r}")
    fracs = [_coord_to_fraction(v) for v in r_raw]
    max_abs = max(abs(f) for f in fracs)
    if max_abs == 0:
        raise ValueError(f"zero direction {r_raw!r}")
    s2 = sum(f * f for f in fracs)
    if s2 <= 0:
        raise ValueError(f"nonpositive squared norm {s2} from {r_raw!r}")
    # Prove maxAbs^2 <= s2 (hence true a<=1) via integers; raises if violated.
    if max_abs * max_abs > s2:
        raise RuntimeError("maxAbs^2 > s2 (unreachable for real vectors)")
    sqrt_lo, sqrt_hi = _sqrt_interval(s2, bits)
    # sqrt_hi >0 always (hi>=1/SCALE). sqrt_lo may be 0 only for tiny s2.
    if sqrt_hi <= 0:
        raise RuntimeError("sqrt_hi<=0 (unreachable for s2>0)")
    a_lo = max_abs / sqrt_hi
    if sqrt_lo == 0:
        # No finite upper from division; 1 is valid (true<=1 proven above).
        a_hi = Fraction(1, 1)
    else:
        a_hi = max_abs / sqrt_lo
        if a_hi > 1:
            # Clip: true<=1 so min(a_hi,1) still valid upper and tighter.
            a_hi = Fraction(1, 1)
    if a_lo < 0:
        a_lo = Fraction(0, 1)
    if a_lo > 1:
        # Unreachable because a_lo<=true<=1 is proven above. Raise rather than
        # silently clipping, so a logic bug cannot hide as a valid interval.
        raise RuntimeError(f"a_lo>1 (unreachable): {a_lo}")
    if a_lo > a_hi:
        raise RuntimeError(f"a interval inverted: lo={a_lo} hi={a_hi}")
    return {
        "a_lo": a_lo, "a_hi": a_hi,
        "max_abs": max_abs, "s2": s2,
        "sqrt_lo": sqrt_lo, "sqrt_hi": sqrt_hi,
        "bits": bits,
    }


def _nearest_axis_for_raw(r_raw: tuple[float, float, float]) -> dict:
    """Deterministic nearest signed Pauli axis for one raw direction.

    Uses exact Fractions (no float tolerance). Finds max |coord| exactly;
    tie among axes broken by smallest index X<Y<Z (documented); sign is sign
    of that coord (+1/-1; zero cannot be max for nonzero vector). Returns
    dict with axis, sign, proxy triple (floats, exact Pauli), max_abs Fraction.
    """
    if not isinstance(r_raw, (list, tuple)) or len(r_raw) != 3:
        raise ValueError(f"r must be length-3, got {r_raw!r}")
    fracs = [_coord_to_fraction(v) for v in r_raw]
    abses = [abs(f) for f in fracs]
    max_abs = max(abses)
    if max_abs == 0:
        raise ValueError(f"zero direction {r_raw!r}")
    # Tie: smallest index wins (exact equality on Fractions).
    idx = min(i for i, a in enumerate(abses) if a == max_abs)
    sgn = 1 if fracs[idx] > 0 else -1
    # fracs[idx] !=0 since max>0.
    if fracs[idx] == 0:
        raise RuntimeError("tied zero max (unreachable)")
    axis = PAULI_AXES[idx]
    proxy = [0.0, 0.0, 0.0]
    proxy[idx] = 1.0 if sgn > 0 else -1.0
    return {
        "axis": axis, "sign": sgn, "index": idx,
        "proxy": (float(proxy[0]), float(proxy[1]), float(proxy[2])),
        "max_abs": max_abs,
    }


def proxy_for_subset(rs_raw: list[tuple[float, float, float]], D) -> list[tuple[float, float, float]]:
    """Build proxy target P_D: j in D replaced by nearest signed Pauli axis.

    rs_raw validated raws; D validated subset (sorted canonical inside).
    Retained factors keep original raw floats exactly. Returns list of
    float triples (proxies are exact Pauli triples 1.0/0.0/-1.0).
    """
    vrs = _validate_rs(rs_raw)
    n = len(vrs)
    Dc = _validate_subset(D, n)
    Dset = set(Dc)
    out: list[tuple[float, float, float]] = []
    for j, r in enumerate(vrs):
        if j in Dset:
            info = _nearest_axis_for_raw(tuple(float(v) for v in r))
            out.append(info["proxy"])
        else:
            out.append((float(r[0]), float(r[1]), float(r[2])))
    # Validate proxy passes estimator acceptance (Pauli triples norm 1).
    # Retained raws already validated; Pauli proxies norm 1 exactly.
    est.validate_target(out)
    return out


def _floor_to_scale(q: Fraction, scale: int) -> Fraction:
    """Greatest multiple of 1/scale at or below q (q arbitrary Fraction).

    scale must be positive int. Returns Fraction with denominator dividing
    scale, hence bounded bit length. Exact integer arithmetic.
    """
    if not isinstance(q, Fraction):
        raise ValueError("q must be Fraction")
    if not isinstance(scale, int) or scale <= 0:
        raise ValueError("scale must be positive int")
    # floor(q*scale)/scale. Python // floors, so correct for negatives too.
    n = (q.numerator * scale) // q.denominator
    return Fraction(n, scale)


def _ceil_to_scale(q: Fraction, scale: int) -> Fraction:
    """Least multiple of 1/scale at or above q (outward upper).

    scale must be positive int. Bounded denominator dividing scale.
    """
    if not isinstance(q, Fraction):
        raise ValueError("q must be Fraction")
    if not isinstance(scale, int) or scale <= 0:
        raise ValueError("scale must be positive int")
    # ceil(q*scale)/scale = -floor(-q*scale)/scale with denominator > 0.
    p, b = q.numerator, q.denominator
    n = -(((-p * scale) // b))
    return Fraction(n, scale)


def geometry_certificate(rs, D, bits: int = GEOMETRY_BITS_DEFAULT) -> dict:
    """Fresh rigorous certificate for subset D: a/c/d intervals (Fractions).

    Computes a intervals for ALL n factors (for transparency) plus product
    overlap c_D=prod_{j in D}(1+a_j)/2 interval and operator bias
    d_D=sqrt(1-c_D) interval, all as Fractions with proofs via integer
    sqrt plus bounded outward product rounding. Empty D gives c=[1,1],
    d=[0,0]. Returns dict with Fractions, outward floats, bits, n, D,
    proxy, timings. Product uses SCALE=2**bits outward rounding after each
    factor (floor for lower, ceil for upper), so geometry denominators
    divide SCALE and str() of geometry fields stays below the Python digit
    limit for all n<=MAX_N and bits<=1024 including default-bit n=200.
    Each endpoint's additional rounding error relative to its exact
    endpoint is <=|D|/SCALE; total width additional rounding is
    <=2|D|/SCALE plus a-enclosure propagation. Sound by induction since
    all factors and partial products lie in [0,1] and rounding is outward.
    This bounded-serialization guarantee covers geometry certificate
    fields only, not arbitrary huge user Fractions, scores, or enormous N;
    inner plans obey certified estimator resource limits.
    """
    t0 = time.perf_counter()
    bits = _require_bits(bits)
    vrs = _validate_rs(rs)
    n = len(vrs)
    Dc = _validate_subset(D, n)
    # Per-factor a intervals.
    a_list: list[dict] = []
    for r in vrs:
        a_list.append(_a_interval_for_factor((float(r[0]), float(r[1]), float(r[2])), bits))
    a_los = [d["a_lo"] for d in a_list]
    a_his = [d["a_hi"] for d in a_list]
    # Product overlap for D with bounded outward rounding.
    # SCALE=2**bits keeps every intermediate geometry denominator dividing
    # SCALE, so bit length stays O(bits) independent of |D| and str() of
    # geometry fields never exceeds the Python digit limit. Soundness by
    # induction: assume c_lo<=true_prev and true_prev<=c_hi with all values
    # in [0,1]; per-factor f_lo<=f_true<=f_hi; then q_lo=c_lo*f_lo<=true and
    # q_hi>=true by monotonicity of products on [0,1]; floor/ceil to SCALE
    # preserves the side (down<=q_lo, up>=q_hi). Each endpoint's additional
    # rounding error vs its exact endpoint is <=|D|/SCALE, so width
    # additional rounding is <=2|D|/SCALE plus a-enclosure propagation
    # (product_rounding_bound=|D|/SCALE is per-endpoint, not whole width).
    # No process-global integer safeguard is changed; geometry serialization
    # is bounded by construction (inner scores/N obey estimator limits).
    if len(Dc) == 0:
        c_lo = Fraction(1, 1)
        c_hi = Fraction(1, 1)
        product_scale = 1 << bits
        product_rounding_bound = Fraction(0, 1)
    else:
        product_scale = 1 << bits
        c_lo = Fraction(1, 1)
        c_hi = Fraction(1, 1)
        for j in Dc:
            f_lo = (Fraction(1, 1) + a_los[j]) / 2
            f_hi = (Fraction(1, 1) + a_his[j]) / 2
            q_lo = c_lo * f_lo
            q_hi = c_hi * f_hi
            c_lo = _floor_to_scale(q_lo, product_scale)
            c_hi = _ceil_to_scale(q_hi, product_scale)
            if c_lo < 0:
                c_lo = Fraction(0, 1)
            if c_hi > 1:
                # Clipping hi to 1 preserves validity since true product <= 1.
                c_hi = Fraction(1, 1)
            if c_lo > 1:
                # Defensive: true <= 1, so lo > 1 signals a bug; fail loudly.
                raise RuntimeError(f"c_lo>1 after rounding (unreachable): {c_lo}")
        if c_lo < 0:
            c_lo = Fraction(0, 1)
        if c_hi > 1:
            c_hi = Fraction(1, 1)
        if c_lo > c_hi:
            raise RuntimeError(f"c interval inverted: {c_lo} > {c_hi}")
        product_rounding_bound = Fraction(len(Dc), product_scale)
    # Operator bias interval.
    s_lo = Fraction(1, 1) - c_hi
    s_hi = Fraction(1, 1) - c_lo
    # s_lo<=s_hi, both in [0,1] (since c in [0,1]).
    if s_lo < 0:
        s_lo = Fraction(0, 1)
    if s_hi < 0:
        s_hi = Fraction(0, 1)
    if s_hi > 1:
        s_hi = Fraction(1, 1)
    if s_lo > s_hi:
        raise RuntimeError("s interval inverted (unreachable)")
    if s_hi == 0:
        d_lo = Fraction(0, 1)
        d_hi = Fraction(0, 1)
    elif s_lo == s_hi:
        d_lo, d_hi = _sqrt_interval(s_lo, bits)
    else:
        # Lower is lo(s_lo), upper is hi(s_hi) by monotonicity of sqrt.
        if s_lo == 0:
            d_lo = Fraction(0, 1)
        else:
            d_lo, _ = _sqrt_interval(s_lo, bits)
        _, d_hi = _sqrt_interval(s_hi, bits)
    if d_lo < 0 or d_hi < 0 or d_lo > 1 or d_hi > 1:
        raise RuntimeError(f"d interval out of [0,1]: [{d_lo},{d_hi}]")
    if d_lo > d_hi:
        raise RuntimeError(f"d interval inverted: {d_lo} > {d_hi}")
    # Certified product rounding metadata: SCALE and worst-case outward error.
    # True overlap lies in [c_lo, c_hi] by the induction above; each
    # endpoint is within |D|/SCALE of its exact endpoint, so the interval
    # width is at most the a-width propagation plus 2|D|/SCALE rounding
    # (product_rounding_bound is per-endpoint, not whole width).
    t1 = time.perf_counter()
    # Proxy for convenience.
    proxy = proxy_for_subset([tuple(float(v) for v in r) for r in vrs], Dc)
    return {
        "n": n, "D": Dc, "bits": bits,
        "a_lo": [str(v) for v in a_los],
        "a_hi": [str(v) for v in a_his],
        "a_lo_frac": tuple(a_los),
        "a_hi_frac": tuple(a_his),
        "c_lo": c_lo, "c_hi": c_hi,
        "c_lo_str": str(c_lo), "c_hi_str": str(c_hi),
        "d_lo": d_lo, "d_hi": d_hi,
        "d_lo_str": str(d_lo), "d_hi_str": str(d_hi),
        "d_lo_float_lower": _float_lower(d_lo),
        "d_hi_float_upper": _float_upper(d_hi),
        "c_lo_float_lower": _float_lower(c_lo),
        "c_hi_float_upper": _float_upper(c_hi),
        "proxy": proxy,
        "cert_time_sec": t1 - t0,
        "product_scale": product_scale,
        "product_rounding_bound": str(product_rounding_bound),
    }


# --- selection -----------------------------------------------------------------

def select_discards(rs, beta, bits: int = GEOMETRY_BITS_DEFAULT) -> dict:
    """Deterministic longest certified-feasible prefix under budget beta.

    Sorts factors by certified proximity midpoint (a_lo+a_hi)/2 descending,
    tie by index ascending. For each prefix length t, computes certified
    upper d_hi(t) for D_t = first t in that order; selects largest t with
    d_hi(t) <= beta (exact Fraction comparison). Returns dict with sorted
    order, chosen D (canonical ascending + proximity order), bias interval
    for chosen, proxy, original/retained noisy counts, metadata.

    Cost: rebuilds every prefix with a fresh geometry_certificate call, so
    total work is quadratic in n (O(n) certificates, each O(n)). Measured
    select_time_sec is reported; n=50 completes in under a second with
    default bits. Optimality only for exact costs with equal per-factor
    value (cheapest-first maximizes cardinality); conservative enclosures
    can underselect (documented). Exact Pauli factors (midpoint 1) sort
    first and are always feasible (d=0), including beta=0.
    """
    t0 = time.perf_counter()
    bits = _require_bits(bits)
    beta_frac = _require_beta(beta)
    vrs = _validate_rs(rs)
    n = len(vrs)
    # Fresh a intervals for ordering.
    a_infos = [_a_interval_for_factor((float(r[0]), float(r[1]), float(r[2])), bits) for r in vrs]
    a_los = [d["a_lo"] for d in a_infos]
    a_his = [d["a_hi"] for d in a_infos]
    mids = [(lo + hi) / 2 for lo, hi in zip(a_los, a_his)]
    order = sorted(range(n), key=lambda j: (-mids[j], j))
    # Prefix feasibility via certified d_hi. Each prefix is rebuilt with a fresh
    # geometry_certificate call (quadratic repeated-certification cost, not an
    # incremental product update). For each t, D_t = order[:t]; the bounded
    # product rounding inside each certificate keeps every call bounded.
    best_t = 0
    best_D: tuple[int, ...] = ()
    best_cert: dict | None = None
    # t=0 always feasible (d=0<=beta since beta>=0).
    cert0 = geometry_certificate([tuple(float(v) for v in r) for r in vrs], [], bits)
    assert cert0["d_hi"] == 0
    best_cert = cert0
    # Iterate t=1..n.
    for t in range(1, n + 1):
        Dt = tuple(sorted(order[:t]))  # canonical ascending for certificate
        cert = geometry_certificate([tuple(float(v) for v in r) for r in vrs], list(Dt), bits)
        if cert["d_hi"] <= beta_frac:
            best_t = t
            best_D = Dt
            best_cert = cert
        # d_hi is monotone nondecreasing in t because each added factor lies
        # in [0.5, 1], so the overlap product can only decrease and d can
        # only increase. Hence once a prefix is infeasible all larger
        # prefixes are infeasible, and breaking here is exact.
        else:
            break
    assert best_cert is not None
    # Proxy and counts.
    proxy = best_cert["proxy"]
    cls_orig = est.classify_target([tuple(float(v) for v in r) for r in vrs])
    cls_proxy = est.classify_target(proxy)
    k_orig = cls_orig["k"]
    k_ret = cls_proxy["k"]
    t1 = time.perf_counter()
    # Proximity order of chosen (prefix order, not ascending).
    chosen_prox_order = tuple(order[:best_t])
    return {
        "n": n, "beta": str(beta_frac), "beta_float": float(beta_frac),
        "bits": bits,
        "sorted_order": tuple(order),
        "midpoints": [str(m) for m in mids],
        "chosen_D": best_D,  # canonical ascending
        "chosen_prox_order": chosen_prox_order,
        "prefix_len": best_t,
        "d_lo": best_cert["d_lo"], "d_hi": best_cert["d_hi"],
        "d_lo_str": str(best_cert["d_lo"]), "d_hi_str": str(best_cert["d_hi"]),
        "c_lo": best_cert["c_lo"], "c_hi": best_cert["c_hi"],
        "c_lo_str": str(best_cert["c_lo"]), "c_hi_str": str(best_cert["c_hi"]),
        "a_lo": best_cert["a_lo"], "a_hi": best_cert["a_hi"],
        "proxy": proxy,
        "k_original": k_orig, "k_retained": k_ret,
        "K_orig": tuple(cls_orig["K_idx"]), "K_ret": tuple(cls_proxy["K_idx"]),
        "select_time_sec": t1 - t0,
        "cert_time_sec": best_cert["cert_time_sec"],
        "optimality_note": (
            "Optimal (max cardinality) only for exact costs with equal "
            "per-factor value (cheapest-first). Conservative enclosures can "
            "invert order within width and underselect (safe side)."
        ),
    }


# --- frozen plan ---------------------------------------------------------------

@dataclass(frozen=True)
class GeometryPlan:
    """Frozen pre-data partial-approximation design (immutable, deep-copied).

    Fields are tuples/Fractions/immutable values. Constructed ONLY via
    plan_geometry which freshly computes geometry (a/c/d) and inner
    EstimatorPlan for proxy. run_geometry_streaming revalidates complete
    plan BEFORE any callback/RNG via canonical reconstruction (fresh
    geometry + fresh inner b/G exactly once) and uses only canonical
    snapshot. Mutating original rs/D/g lists or attempting plan.D=...
    cannot alter collection. Dict plans rejected.
    """

    rs_raw: tuple[tuple[float, float, float], ...]
    D: tuple[int, ...]
    proxy_raw: tuple[tuple[float, float, float], ...]
    N: int
    delta: float
    channel: str
    g_round: tuple[Fraction, ...]
    inner_plan: object  # EstimatorPlan (typed as object to avoid import cycle in repr)
    d_lo: Fraction
    d_hi: Fraction
    c_lo: Fraction
    c_hi: Fraction
    bits: int
    den: int

    def __getitem__(self, key):  # minimal dict-compat
        return getattr(self, key)


def _extract_inner_k(proxy) -> int:
    cls = est.classify_target(proxy)
    return int(cls["k"])


def plan_geometry(rs, D, N: int, delta: float = est.DELTA_DEFAULT,
                  channel: str = DEFAULT_CHANNEL,
                  g_round=None, b_cert=None, G_cert=None, *,
                  bits: int = GEOMETRY_BITS_DEFAULT,
                  den: int = ROUND_DEN_DEFAULT,
                  inner_bits: int = est.ENCLOSURE_BITS) -> GeometryPlan:
    """Freeze partial-approximation plan with fresh geometry + inner binding.

    Args:
      rs: heterogeneous pure-direction target list (validated; represented
        target is r/||r|| exactly).
      D: deterministic discard subset (list/tuple, empty/all allowed).
      N: any ordinary positive int (not bool).
      delta: failure level in (0,1) (default 1/3).
      channel: "common" (default, theorem) or "degraded_half" (explicit).
      g_round: executed inner score Fractions length k'+1 for proxy (k' =
        #noisy in proxy). If None, auto-solves via solve_and_certify at
        (k',N,channel,delta) defaults. Floats rejected (no silent re-round).
        For k'=0 must be exactly [Fraction(1)].
      b_cert/G_cert: OPTIONAL caller claims (Fractions) validated against
        fresh inner bounds (understated/wrong-channel rejected). Stored plan
        keeps FRESH bounds via inner_plan, never claims.
      bits: geometry enclosure bits (resource-limited, default 80).
      den: rounding denominator provenance for inner (stored; user Fractions
        used exactly regardless).
      inner_bits: channel enclosure bits for inner plan_estimator (default 80).

    Returns frozen GeometryPlan. Caller must then collect exactly N fresh
    i.i.d. copies with no stopping/reuse. Raises ValueError on mismatch.
    """
    # Validate all before building (deep-copy semantics: copy containers).
    vrs = _validate_rs(rs)
    n = len(vrs)
    Dc = _validate_subset(D, n)
    if _is_bool(N) or not isinstance(N, int) or N <= 0:
        # Reuse estimator message style: must be int (bool rejected).
        raise ValueError(f"N must be int (bool rejected), got {type(N).__name__}={N!r}")
    N = _require_positive_int("N", N)
    delta_f = _require_delta(delta)
    _require_channel(channel)
    bits = _require_bits(bits)
    den = _require_den(den)
    if _is_bool(inner_bits) or not isinstance(inner_bits, int) or inner_bits <= 0:
        raise ValueError(f"inner_bits must be positive int, got {inner_bits!r}")
    if inner_bits < 1 or inner_bits > GEOMETRY_BITS_MAX:
        raise ValueError(f"inner_bits out of resource range: {inner_bits}")
    # Optional caller certs must both be Fractions if supplied.
    if (b_cert is not None) or (G_cert is not None):
        if not isinstance(b_cert, Fraction) or not isinstance(G_cert, Fraction):
            raise ValueError("caller b_cert/G_cert must both be Fractions when supplied")
        if b_cert < 0 or G_cert < 0:
            raise ValueError("caller b_cert/G_cert must be nonnegative")
    # Canonical raws/proxy/geometry (fresh).
    rs_raw_t = tuple((float(r[0]), float(r[1]), float(r[2])) for r in vrs)
    proxy_list = proxy_for_subset([tuple(float(v) for v in r) for r in rs_raw_t], list(Dc))
    proxy_t = tuple((float(r[0]), float(r[1]), float(r[2])) for r in proxy_list)
    cert = geometry_certificate([tuple(float(v) for v in r) for r in rs_raw_t], list(Dc), bits)
    # Inner k' and g handling.
    cls_proxy = est.classify_target(list(proxy_t))
    kprime = int(cls_proxy["k"])
    if g_round is None:
        # Auto-solve (uses SciPy HiGHS; may raise clearly). Pass inner_bits/den
        # so solve enclosure matches plan_estimator fresh check (avoids
        # midpoint-mismatch false rejection when inner_bits != default).
        rec = est.solve_and_certify(kprime, N, channel, delta=delta_f, bits=inner_bits, den=den)
        g_list = [Fraction(s) for s in rec["g_round"]]
        # Use rec certs as claims (validated inside plan_estimator).
        b_claim = Fraction(rec["b_cert"])
        G_claim = Fraction(rec["G_cert"])
        # When the caller also supplies b_cert/G_cert alongside auto-solve,
        # those claims are validated against the fresh inner bounds after the
        # inner plan is built (understatements rejected below). The stored
        # plan always keeps the fresh bounds, never the caller claims.
        if b_cert is not None or G_cert is not None:
            pass
        g_t = tuple(g_list)
        # Build inner with rec claims (fresh check inside plan_estimator),
        # then separately validate any caller-supplied claims.
        inner = est.plan_estimator(
            list(proxy_t), N, delta_f, channel, g_t, b_claim, G_claim,
            bits=inner_bits, den=den,
        )
        # Additionally validate caller claims if supplied (must not understate).
        if b_cert is not None and G_cert is not None:
            if b_cert < inner.b_cert or G_cert < inner.G_cert:
                raise ValueError(
                    f"caller b/G understates fresh inner bounds "
                    f"({b_cert},{G_cert}) vs fresh ({inner.b_cert},{inner.G_cert})"
                )
    else:
        if not isinstance(g_round, (list, tuple)):
            raise ValueError(f"g_round must be list/tuple of Fractions, got {type(g_round).__name__}")
        if len(g_round) != kprime + 1:
            raise ValueError(f"g_round length {len(g_round)} vs k'+1={kprime+1} for proxy")
        g_checked: list[Fraction] = []
        for v in g_round:
            if not isinstance(v, Fraction):
                raise ValueError(f"g_round must be Fractions (no silent re-rounding), got {type(v).__name__}={v!r}")
            g_checked.append(v)
        g_t = tuple(g_checked)
        if kprime == 0 and not (len(g_t) == 1 and g_t[0] == Fraction(1, 1)):
            raise ValueError(f"k'=0 requires exact indicator [1], got {[str(v) for v in g_t]}")
        if b_cert is not None or G_cert is not None:
            inner = est.plan_estimator(
                list(proxy_t), N, delta_f, channel, g_t, b_cert, G_cert,
                bits=inner_bits, den=den,
            )
        else:
            inner = est.plan_estimator(
                list(proxy_t), N, delta_f, channel, g_t,
                bits=inner_bits, den=den,
            )
    # Freeze outer (tuples/Fractions/immutable, deep-copied).
    return GeometryPlan(
        rs_raw=rs_raw_t, D=Dc, proxy_raw=proxy_t,
        N=int(N), delta=float(delta_f), channel=str(channel),
        g_round=g_t, inner_plan=inner,
        d_lo=cert["d_lo"], d_hi=cert["d_hi"],
        c_lo=cert["c_lo"], c_hi=cert["c_hi"],
        bits=int(bits), den=int(den),
    )


def run_geometry_streaming(plan, measure_fn, rng: random.Random) -> dict:
    """Stream N copies for original target via proxy estimator + geometry widening.

    plan must be frozen GeometryPlan from plan_geometry (dicts rejected).
    At entry, BEFORE any callback/RNG, revalidates original target, subset,
    proxy, geometry certificate, and inner-plan binding via canonical
    reconstruction (fresh geometry + fresh inner b/G exactly once, timed as
    revalidate_time_sec), rejecting underclaimed geometry/certs and
    inconsistent derived metadata and using only canonical snapshot; never
    re-solves/rebuilds per copy. Mutating external rs/g lists or attempting
    plan.D=... cannot alter collection.

    measure_fn(axes)->outcomes called once per copy with fresh axes
    (correlated outcomes required). Exact Fraction accumulation with ALL-N
    denominator (via certified estimator). Total radius R_total = R_inner +
    d_hi (Fractions exact); endpoints clip01(mean-R_total),
    clip01(mean+R_total) rationally then outward (lower downward, upper
    upward), ordered in [0,1] even when raw interval disjoint. Returns both
    proxy (inner) estimate/interval and original-target interval separately.
    """
    t0 = time.perf_counter()
    if not isinstance(plan, GeometryPlan):
        raise ValueError("plan must be frozen GeometryPlan from plan_geometry (dicts rejected)")
    if not callable(measure_fn):
        raise ValueError("measure_fn must be callable")
    if not hasattr(rng, "random"):
        raise ValueError("rng must have .random()")
    # Strict type checks before casts (bool/Fraction rejections).
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
    if not isinstance(plan.D, tuple):
        raise ValueError("plan D must be tuple (mutable list rejected)")
    for _v in plan.D:
        if _is_bool(_v) or not isinstance(_v, int):
            raise ValueError(f"plan D entries must be ints, got {_v!r}")
    if not isinstance(plan.proxy_raw, tuple) or len(plan.proxy_raw) != len(plan.rs_raw):
        raise ValueError("plan proxy_raw must be tuple matching rs_raw")
    for _r in plan.proxy_raw:
        if not isinstance(_r, tuple) or len(_r) != 3:
            raise ValueError(f"plan proxy entries must be 3-tuples, got {_r!r}")
        for _v in _r:
            if _is_bool(_v) or not isinstance(_v, (int, float)):
                raise ValueError(f"plan proxy coords must be numbers, got {_v!r}")
            if not math.isfinite(float(_v)):
                raise ValueError(f"plan proxy coord nonfinite {_v!r}")
    if _is_bool(plan.N) or not isinstance(plan.N, int):
        raise ValueError(f"plan N must be int (bool rejected), got {plan.N!r}")
    if _is_bool(plan.delta) or not isinstance(plan.delta, (int, float)):
        raise ValueError(f"plan delta must be number (bool rejected), got {plan.delta!r}")
    if not isinstance(plan.channel, str):
        raise ValueError(f"plan channel must be str, got {plan.channel!r}")
    _require_channel(plan.channel)
    if not isinstance(plan.g_round, tuple):
        raise ValueError(f"plan g_round must be tuple (mutable list rejected), got {type(plan.g_round).__name__}")
    for _v in plan.g_round:
        if not isinstance(_v, Fraction):
            raise ValueError(f"plan g_round must be Fractions, got {type(_v).__name__}={_v!r}")
    if not isinstance(plan.d_lo, Fraction) or not isinstance(plan.d_hi, Fraction):
        raise ValueError("plan d_lo/d_hi must be Fractions")
    if not isinstance(plan.c_lo, Fraction) or not isinstance(plan.c_hi, Fraction):
        raise ValueError("plan c_lo/c_hi must be Fractions")
    if _is_bool(plan.bits) or not isinstance(plan.bits, int):
        raise ValueError(f"plan bits must be int, got {plan.bits!r}")
    if _is_bool(plan.den) or not isinstance(plan.den, int):
        raise ValueError(f"plan den must be int, got {plan.den!r}")
    # Inner plan must be EstimatorPlan (frozen) from certified estimator.
    if not isinstance(plan.inner_plan, est.EstimatorPlan):
        raise ValueError("plan inner_plan must be frozen EstimatorPlan from plan_estimator")
    # Canonical reconstruction exactly once (fresh geometry + inner binding),
    # zero callbacks/RNG so far. Timed.
    t_r0 = time.perf_counter()
    # Recompute canonical raws/proxy/geometry from authoritative rs_raw/D/bits.
    # This validates target acceptance, subset, proxy derivation, and cert.
    try:
        canon_vrs = _validate_rs(list(plan.rs_raw))
    except Exception as e:
        raise ValueError(f"plan rs_raw invalid: {e}") from e
    n_canon = len(canon_vrs)
    try:
        Dc_canon = _validate_subset(list(plan.D), n_canon)
    except Exception as e:
        raise ValueError(f"plan D invalid: {e}") from e
    if Dc_canon != plan.D:
        raise ValueError(f"plan D {plan.D!r} != canonical {Dc_canon!r}")
    # Proxy must match deterministic derivation.
    proxy_expected = proxy_for_subset([tuple(float(v) for v in r) for r in canon_vrs], list(Dc_canon))
    proxy_expected_t = tuple((float(r[0]), float(r[1]), float(r[2])) for r in proxy_expected)
    if tuple(plan.proxy_raw) != proxy_expected_t:
        raise ValueError("plan proxy_raw != canonical proxy from rs+D (wrong proxy rejected)")
    # Fresh geometry must match stored (reject understated or any tamper).
    fresh_cert = geometry_certificate(
        [tuple(float(v) for v in r) for r in canon_vrs], list(Dc_canon), plan.bits
    )
    if plan.c_lo != fresh_cert["c_lo"] or plan.c_hi != fresh_cert["c_hi"]:
        raise ValueError(f"plan c [{plan.c_lo},{plan.c_hi}] != fresh [{fresh_cert['c_lo']},{fresh_cert['c_hi']}]")
    if plan.d_lo != fresh_cert["d_lo"] or plan.d_hi != fresh_cert["d_hi"]:
        raise ValueError(
            f"plan d [{plan.d_lo},{plan.d_hi}] != fresh [{fresh_cert['d_lo']},{fresh_cert['d_hi']}] "
            f"(understated geometry rejected)"
        )
    if plan.bits != fresh_cert["bits"]:
        raise ValueError("plan bits != canonical")
    # Inner-plan binding: outer proxy/N/delta/channel/g must match inner.
    inner = plan.inner_plan
    if tuple(inner.rs_raw) != tuple(plan.proxy_raw):
        raise ValueError("inner_plan rs_raw != outer proxy_raw (binding rejected)")
    if inner.N != plan.N:
        raise ValueError("inner_plan N != outer N")
    if float(inner.delta) != float(plan.delta):
        raise ValueError("inner_plan delta != outer delta")
    if inner.channel != plan.channel:
        raise ValueError("inner_plan channel != outer channel")
    if tuple(inner.g_round) != tuple(plan.g_round):
        raise ValueError("inner_plan g_round != outer g_round")
    # Canonical inner reconstruction (fresh b/G exactly once for outer check;
    # run_streaming will do its own once more before data, still
    # zero callbacks so far). This catches understated inner certs early
    # before any data, without consuming RNG.
    try:
        canon_inner = est.plan_estimator(
            list(proxy_expected_t), plan.N, plan.delta, plan.channel,
            tuple(plan.g_round), inner.b_cert, inner.G_cert,
            bits=inner.bits, den=inner.den,
        )
    except Exception as e:
        raise ValueError(f"inner-plan canonical reconstruction failed: {e}") from e
    if canon_inner.b_cert != inner.b_cert or canon_inner.G_cert != inner.G_cert:
        raise ValueError("inner_plan b/G != fresh (understated inner rejected)")
    # Derived n/k fields are not trusted: compare against canonical proxy
    # classification here; est.run_streaming revalidates the full inner plan
    # again before any data.
    cls_canon_proxy = est.classify_target(list(proxy_expected_t))
    if inner.n != cls_canon_proxy["n"] or inner.k != cls_canon_proxy["k"]:
        raise ValueError("inner_plan n/k != canonical proxy classification")
    t_r1 = time.perf_counter()
    revalidate_time = t_r1 - t_r0
    # Use ONLY canonical snapshot values for streaming (fresh d_hi equals stored).
    d_hi_canon: Fraction = fresh_cert["d_hi"]
    # Delegate to certified estimator for exactly N fresh copies. Its own
    # pre-data revalidation occurs before any callback/RNG inside.
    inner_out = est.run_streaming(inner, measure_fn, rng)
    # Compose geometry + LP confidence (exact Fractions).
    try:
        mean_exact = Fraction(inner_out["mean_exact"])
        R_inner = Fraction(inner_out["radius_exact"])
    except Exception as e:
        raise RuntimeError(f"inner result mean/radius parse failed: {e}") from e
    if R_inner < 0:
        raise RuntimeError("inner radius negative (unreachable)")
    if d_hi_canon < 0:
        raise RuntimeError("geometry d_hi negative (unreachable)")
    R_total = R_inner + d_hi_canon
    lo_rat = _clip01_frac(mean_exact - R_total)
    hi_rat = _clip01_frac(mean_exact + R_total)
    lo = _float_lower(lo_rat)
    hi = _float_upper(hi_rat)
    if lo < 0.0:
        lo = 0.0
    if hi > 1.0:
        hi = 1.0
    if lo > hi:
        raise RuntimeError("outward interval inversion (unreachable bug)")
    t1 = time.perf_counter()
    # Separate proxy estimate/interval from original-target interval.
    return {
        "N": inner_out["N"], "n": inner_out["n"],
        "k_original": est.classify_target(list(canon_vrs))["k"],
        "k_proxy": inner_out["k"],
        "channel": plan.channel,
        "n_measurements": inner_out["n_measurements"],
        "D": plan.D,
        "proxy_estimate": inner_out["estimate"],
        "proxy_interval": inner_out["interval"],
        "proxy_mean_exact": inner_out["mean_exact"],
        "proxy_mean_exact_float": inner_out["mean_exact_float"],
        "proxy_radius_exact": inner_out["radius_exact"],
        "proxy_radius": inner_out["radius"],
        "proxy_b_cert": inner_out["b_cert"],
        "proxy_G_cert": inner_out["G_cert"],
        "d_lo": str(plan.d_lo), "d_hi": str(plan.d_hi),
        "d_lo_float_lower": _float_lower(plan.d_lo),
        "d_hi_float_upper": _float_upper(plan.d_hi),
        "c_lo": str(plan.c_lo), "c_hi": str(plan.c_hi),
        "radius_total_exact": str(R_total),
        "radius_total": _float_upper(R_total),
        "mean_exact": inner_out["mean_exact"],
        "mean_exact_float": inner_out["mean_exact_float"],
        "estimate_proxy": inner_out["estimate"],
        "interval_original": [lo, hi],
        "interval_original_exact": [str(lo_rat), str(hi_rat)],
        "estimate": inner_out["estimate"],  # alias: proxy fidelity point estimate
        "interval": [lo, hi],  # alias: confidence for ORIGINAL target
        "n_flag1": inner_out["n_flag1"],
        "weight_counts_flag1": inner_out["weight_counts_flag1"],
        "stream_time_sec": t1 - t0,
        "revalidate_time_sec": revalidate_time,
        "inner_revalidate_time_sec": inner_out.get("revalidate_time_sec"),
        "bits": plan.bits, "den": plan.den,
    }


# --- channel geometry identities (for analysis and validation) -----------------

def mu_upper_of_L(L_frac: Fraction) -> Fraction:
    """Exact mu for upper channel: (L^2-1)/4 with L>=1 Fraction.

    Proof: eta=(1-1/L)/2, mu=eta(1-eta)/(1-2eta)^2=(L^2-1)/4 (see theory).
    Validates L>=1. Returns Fraction exact.
    """
    if not isinstance(L_frac, Fraction):
        raise ValueError("L must be Fraction")
    if L_frac < 1:
        raise ValueError(f"L must be >=1, got {L_frac}")
    return (L_frac * L_frac - 1) / 4


def mu_lower_of_a(a_frac: Fraction) -> Fraction:
    """Exact mu for lower channel: (a^{-2}-1)/4 with 0<a<=1 Fraction.

    Proof: eta_*=(1-a)/2, mu=(a^{-2}-1)/4. Validates 0<a<=1.
    """
    if not isinstance(a_frac, Fraction):
        raise ValueError("a must be Fraction")
    if a_frac <= 0 or a_frac > 1:
        raise ValueError(f"a must satisfy 0<a<=1, got {a_frac}")
    # (1/a^2 -1)/4 = (1-a^2)/(4a^2)
    return (Fraction(1, 1) - a_frac * a_frac) / (4 * a_frac * a_frac)


def homogeneous_bias_all(theta: float, n: int, bits: int = GEOMETRY_BITS_DEFAULT) -> dict:
    """Float-angle diagnostic plus rigorous certificate for supplied directions.

    Requires 0<theta<=pi/4 with strict boundary (float, finite, no tolerance),
    n positive int. The float quantities c_float=cos(theta/2)^{2n} and
    d_float=sqrt(1-c_float) are UNCERTIFIED diagnostics computed in binary64;
    they are kept separate from the rigorous certificate. The rigorous part
    calls geometry_certificate on the explicitly supplied binary sin/cos
    direction triples (math.sin(th), 0.0, math.cos(th)) repeated n times with
    D=all; the represented target is r/||r|| exactly with ||r|| within 1e-9
    of 1 for float sin/cos, and the returned c_lo/c_hi/d_lo/d_hi intervals
    are the certificate. For a certificate on another exactly normalized
    direction, call geometry_certificate directly. Validates inputs strictly.
    """
    if _is_bool(theta) or not isinstance(theta, (int, float)):
        raise ValueError(f"theta must be number, got {theta!r}")
    th = float(theta)
    if not math.isfinite(th) or not (0.0 < th <= math.pi / 4):
        raise ValueError(f"theta must satisfy 0<theta<=pi/4 (strict, no tolerance), got {theta!r}")
    n = _require_positive_int("n", n)
    bits = _require_bits(bits)
    # Uncertified float-angle diagnostic (binary64, illustration only).
    c_float = (math.cos(th / 2.0)) ** (2 * n)
    d_float = math.sqrt(max(0.0, 1.0 - c_float))
    # Rigorous certificate for the explicitly supplied direction triples.
    # Float sin/cos satisfy sin^2+cos^2=1 up to binary64 rounding; the
    # validator accepts them because the norm is within 1e-9 of 1, and the
    # certificate is for r/||r|| exactly.
    r = (math.sin(th), 0.0, math.cos(th))
    try:
        vrs = _validate_rs([r] * n)
    except Exception as e:
        raise ValueError(f"homogeneous triple validation failed: {e}") from e
    cert = geometry_certificate([tuple(float(v) for v in q) for q in vrs], list(range(n)), bits)
    return {
        "theta": th, "n": n, "bits": bits,
        "c_float": c_float, "d_float": d_float,
        "c_lo": cert["c_lo"], "c_hi": cert["c_hi"],
        "d_lo": cert["d_lo"], "d_hi": cert["d_hi"],
    }


# --- demo -------------------------------------------------------------------

def demo(seed: int = 20260908) -> int:
    """Runnable example: heterogeneous n=4 (core 1 gapped + tail near-axis).

    Selects discards under beta=0.04, builds geometry plan at N=16 common
    default, streams from REAL iid R-diagonal mixture (0.7/0.3), prints proxy
    vs original intervals as illustration only. Deterministic given seed.
    """
    print("=== product_state_geometry demo  ===")
    print(f"python {platform.python_version()}")
    try:
        import numpy, scipy  # noqa: F401
        print(f"numpy {numpy.__version__} scipy {scipy.__version__}")
    except Exception as e:
        print(f"NumPy/SciPy import failed: {e}")
        return 1
    import math as _m
    s3 = _m.sqrt(3.0)
    # Core gapped 111 + tail near-axis (small tilts) + one stabilizer.
    th_small = 0.02
    r_tail = (_m.sin(th_small), 0.0, _m.cos(th_small))
    rs = [(1 / s3, 1 / s3, 1 / s3), r_tail, r_tail, (0.0, 0.0, 1.0)]
    print(f"target rs={rs}")
    beta = 0.04
    print(f"budget beta={beta}")
    sel = select_discards(rs, beta, GEOMETRY_BITS_DEFAULT)
    print(f"sorted_order={sel['sorted_order']} chosen_D={sel['chosen_D']} "
          f"k_orig={sel['k_original']} k_ret={sel['k_retained']} "
          f"d=[{sel['d_lo']},{sel['d_hi']}]")
    print(f"proxy={sel['proxy']}")
    N = 16
    # Build geometry plan (auto inner solve).
    plan = plan_geometry(rs, list(sel["chosen_D"]), N, 1.0 / 3.0, "common",
                         None, None, None, bits=GEOMETRY_BITS_DEFAULT)
    print(f"plan: n={len(plan.rs_raw)} D={plan.D} N={plan.N} channel={plan.channel} "
          f"k_proxy={plan.inner_plan.k} d_hi={plan.d_hi} "
          f"b_inner={plan.inner_plan.b_cert} G_inner={plan.inner_plan.G_cert}")
    hats = est.classify_target(rs)["rs_hat"]
    backend_rng = random.Random(seed + 7919)
    measure = est.make_demo_mixture_measure(hats, backend_rng, 0.7)
    design_rng = random.Random(seed)
    out = run_geometry_streaming(plan, measure, design_rng)
    print(f"proxy estimate={out['proxy_estimate']:.4f} proxy_interval={out['proxy_interval']} "
          f"original interval={out['interval_original']} R_total_exact={out['radius_total_exact']}")
    print("Frozen design: D/proxy/N/delta/score fixed before data; no stopping/reuse.")
    print("Proxy estimate is fidelity to P_D; widened interval covers fidelity to P (ideal probs).")
    print("Validity (geometry+LP) certified; optimality NOT claimed.")
    print("=== demo done ===")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Bounded partial Pauli geometry (frozen small harness)")
    ap.add_argument("--demo", action="store_true", help="run runnable demo")
    ap.add_argument("--seed", type=int, default=20260908)
    args = ap.parse_args(argv)
    if args.demo:
        return demo(seed=args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
