"""Bounded homogeneous-fidelity prototype via complex local scores.

Scope: Homogeneous complex-score estimator
(docs/REPRODUCIBILITY.md). Bounded exact-rational prototype only:
homogeneous pure target with exact-rational UNIT Bloch triple, finite rational-circle
atoms, fresh exact residual/mass/variance certificates, median-of-means with 3 blocks
for delta=1/3.  
no scalability/practical-size claim, no novelty claim.

Measurement model: known pure homogeneous product
P = tau_+^{tensor n}, tau_+=(I+R)/2, R=r.(X,Y,Z), r unit. Unknown arbitrary iid mixed
sigma (worst case). Per copy: product Pauli basis, one measurement per qubit, full n
outcome bits. Copy cost N only; nN locals, preprocessing, runtime, memory, precision
reported separately. No total-runtime bound claimed. Fixed (eps0,delta0)=(1/20,1/3)
reference; prototype supports loose eps for tiny demo plus exact-law evaluation.
Ideal axis-choice/Born probabilities assumed; hardware/control errors outside
certificates. No hardware guarantee.

Target convention (EXPLICITLY bounded rational, DISTINCT from normalized-float API):
  Each input triple r is (int/Fraction, int/Fraction, int/Fraction) with sum squares
  EXACTLY 1 as Fractions (bool/float/str rejected). The represented target is exactly
  r (no normalization; already unit). Pauli classification is exact: (0,0,1) etc. with
  exactly one coordinate +-1 and others 0. Dominant axis = argmax |r_i| with tie break
  smallest X<Y<Z (documented); sign = sign of that coordinate. s^2=1-a^2 exact,
  lambda=n s^2 exact. Signed/permuted/tied/Pauli all supported. Float directions are
  REJECTED here (use product_state_estimator.py for float API); no claim that all real
  targets have finite-precision certificates.

Atoms (rational circle, exact):
  T is Fraction with 0<T<=1/4 (float rejected). u is Fraction/int with |u|<=T/2
  (sufficient angle bound). scale=1-T^2, den=1+u^2, re0=(1-u^2)/den, im0=2u/den,
  z=scale*(re0+i im0) as pair of Fractions. Then |z|=scale exactly and
  |arg z|=2|arctan u|<=2|u|<=T (since |arctan|<=|.|). Stored as (u,T); z derived.
  Membership rechecked exactly before callbacks (|z|^2==scale^2 and |u|<=T/2).

Scores (exact complex Fractions):
  t=1-z. If axis==dominant (longitudinal, prob 3/4):
    h = 1-2t/3+2 t c_true o/3 with c_true=signed dominant coordinate (possibly
    negative; first moment holds for any sign, second-moment diagonals swap,
    max bound unchanged).
  Else (transverse, prob 1/8 each):
    h = 1+4 t r_axis o with r_axis=signed coordinate for that axis.
  Per-qubit h as (re,im) Fractions. Product over n qubits per atom gives H_j
  (complex Fractions). Final per-copy real score S=Re(sum_j c_j H_j)
  =sum_j (re_c*re_H-im_c*im_H) as Fraction. First-moment operator exactly
  A_z=tau_++z tau_-; second-moment M_z=E|h|^2 with top/bottom/off as in theory;
  M_z<=[1+9 s^2 T^2]I for all physical c in [0,1] (Young verified in theory).
  Tensorization gives E|H|^2<=mass^2 exp(9 lambda T^2), conservatively
  V=mass^2 3^{ceil(9 lambda T^2)} (e<3, ceil0 gives 1) exact.

Coefficients (exact residual, float LP never certificate):
  coeffs list of (re,im) Fractions, one per atom. Residual
  b=max_{k=0..n}|Re(sum_j c_j z_j^k)-delta_{k0}| exact Fractions (z^k exact pow).
  Mass M=sum(|re|+|im|) exact (conservative upper for sum|c|; |c|<=|re|+|im|).
  Small finite-grid LP (SciPy HiGHS) may PROPOSE floats minimizing mass subject to
  tight bias, but rounded rationals receive FRESH exact residual/mass/V; float
  feasibility never certificate. Finite n grid n=1..4, T=1/4, beta=1/40 only;
  infeasible/large weights reported honestly (no silent threshold weakening).

Frozen plan + streaming (delta=1/3 only, no log arithmetic):
  HomogeneousPlan frozen dataclass (tuples/Fractions). plan_homogeneous validates
  target/n/T/atoms/coeffs/beta, recomputes residual/mass/lambda/V/dom, requires
  b_cert<=beta (strict margin for rounding must come from tighter LP solve).
  plan_copies (count-only) computes tol=eps-b>0, m=ceil(4V/tol^2), N=3m exact
  (Chebyshev p_bad<=1/4 per block, median of 3 fails <=5/32<1/3). run_... revalidates
  complete plan BEFORE any callback/RNG (zero draws on tamper), recomputes bindings,
  checks N<=max_copies BEFORE callbacks (huge requests fail), samples axes via
  rng.randrange(8) (0->transverse1,1->transverse2,2-7->longitudinal), calls
  measure_fn(axes)->outcomes once per copy (full-copy only, correlated outcomes
  required), accumulates exact Fraction block means, median then clip to [0,1].
  General delta optional (not implemented to avoid log). Tiny demo uses loose eps
  or exact-law evaluation without enormous simulation. No fake iid labelled physical.

Run demo: system python code/verify_homogeneous_fidelity.py (writes JSON, exits nonzero
on failure). Requires NumPy/SciPy in .venv for LP+eigvalsh diagnostics only;
exact certificates are stdlib Fractions.
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

# --- constants ------------------------------------------------------------

T_MAX = Fraction(1, 4)
BETA_DEFAULT = Fraction(1, 40)
EPS0 = Fraction(1, 20)
MAX_N = 32
MAX_ATOMS = 32
MAX_COPIES = 100000
ROUND_DEN_DEFAULT = 4096
PAULI_AXES = ("X", "Y", "Z")
AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}


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


def _as_fraction_exact(name: str, v) -> Fraction:
    """Accept int/Fraction only (bool/float/str rejected), return Fraction."""
    if _is_bool(v):
        raise ValueError(f"{name} must be int/Fraction (bool rejected), got {v!r}")
    if isinstance(v, int):
        return Fraction(v, 1)
    if isinstance(v, Fraction):
        if v.denominator == 0:
            raise ValueError(f"{name} zero denominator")
        return v
    raise ValueError(
        f"{name} must be int/Fraction for exact-rational API (float rejected; "
        f"use product_state_estimator for float directions), got {type(v).__name__}={v!r}"
    )


def _require_T(T) -> Fraction:
    Tf = _as_fraction_exact("T", T)
    if Tf <= 0 or Tf > T_MAX:
        raise ValueError(f"T must satisfy 0<T<=1/4, got {Tf} ({float(Tf):.6g})")
    return Tf


def _require_beta(beta) -> Fraction:
    bf = _as_fraction_exact("beta", beta)
    if bf <= 0 or bf >= 1:
        raise ValueError(f"beta must satisfy 0<beta<1, got {bf}")
    return bf


def _require_eps(eps) -> Fraction:
    ef = _as_fraction_exact("eps", eps)
    if ef <= 0 or ef > 1:
        raise ValueError(f"eps must satisfy 0<eps<=1, got {ef}")
    return ef


def _require_n(n) -> int:
    if _is_bool(n) or not isinstance(n, int):
        raise ValueError(f"n must be int (bool rejected), got {type(n).__name__}={n!r}")
    if n < 1 or n > MAX_N:
        raise ValueError(f"n must satisfy 1<=n<={MAX_N} (bounded prototype), got {n}")
    return n


# --- rational target ------------------------------------------------------

def validate_rational_target(r) -> tuple[Fraction, Fraction, Fraction]:
    """Validate exact-rational UNIT Bloch triple, return Fractions.

    r must be list/tuple length 3 of int/Fraction (bool/float rejected).
    Requires sum squares EXACTLY 1 as Fractions. Returns tuple of Fractions.
    """
    if not isinstance(r, (list, tuple)) or len(r) != 3:
        raise ValueError(f"r must be length-3 sequence, got {r!r}")
    fracs = [_as_fraction_exact(f"r[{i}]", v) for i, v in enumerate(r)]
    s2 = fracs[0] * fracs[0] + fracs[1] * fracs[1] + fracs[2] * fracs[2]
    if s2 != 1:
        raise ValueError(
            f"rational target must satisfy sum squares EXACTLY 1, got {s2}={float(s2):.12g} "
            f"from {r!r} (no silent rescaling; floats rejected)"
        )
    return (fracs[0], fracs[1], fracs[2])


def dominant_info(r_frac: tuple[Fraction, Fraction, Fraction]) -> dict:
    """Dominant axis info for exact rational triple (validated).

    Returns dict with dom_idx (0/1/2, smallest on tie), dom_axis (X/Y/Z),
    dom_sign (+1/-1 of that coordinate), a=max|r| (Fraction), s2=1-a^2,
    transverse indices, is_pauli (exact Pauli eigenstate).
    """
    if not isinstance(r_frac, (list, tuple)) or len(r_frac) != 3:
        raise ValueError(f"r_frac must be length-3, got {r_frac!r}")
    for v in r_frac:
        if not isinstance(v, Fraction):
            raise ValueError(f"r_frac entries must be Fractions, got {type(v).__name__}")
    abses = [abs(v) for v in r_frac]
    a = max(abses)
    # Tie: smallest index wins (exact equality).
    dom_idx = min(i for i, av in enumerate(abses) if av == a)
    if r_frac[dom_idx] == 0:
        raise ValueError(f"dominant coordinate zero (unreachable for unit vector): {r_frac!r}")
    dom_sign = 1 if r_frac[dom_idx] > 0 else -1
    s2 = Fraction(1, 1) - a * a
    if s2 < 0:
        raise RuntimeError(f"s2 negative (unreachable for unit): {s2}")
    # Exact Pauli: one coord +-1, others 0.
    is_pauli = (a == 1) and sum(1 for v in r_frac if v == 0) == 2
    transverse = [i for i in range(3) if i != dom_idx]
    return {
        "dom_idx": dom_idx,
        "dom_axis": PAULI_AXES[dom_idx],
        "dom_sign": dom_sign,
        "a": a,
        "s2": s2,
        "transverse": tuple(transverse),
        "is_pauli": bool(is_pauli),
        "r": tuple(r_frac),
    }


# --- rational-circle atoms ------------------------------------------------

def atom_from_u(u, T) -> dict:
    """Build exact rational-circle atom from u,T (Fractions).

    Requires T in (0,1/4], |u|<=T/2 (sufficient angle bound). Returns dict with
    u,T (Fractions), scale=1-T^2, z_re,z_im (Fractions), plus strings for freezing.
    z=(1-T^2)*(1-u^2+2iu)/(1+u^2). Verifies |z|^2==scale^2.
    """
    uf = _as_fraction_exact("u", u)
    Tf = _require_T(T)
    if abs(uf) > Tf / 2:
        raise ValueError(
            f"|u|<=T/2 sufficient angle bound violated: |{uf}|={float(abs(uf)):.6g} "
            f"> T/2={float(Tf/2):.6g} (exact check)"
        )
    scale = Fraction(1, 1) - Tf * Tf
    den = Fraction(1, 1) + uf * uf
    re0 = (Fraction(1, 1) - uf * uf) / den
    im0 = (Fraction(2, 1) * uf) / den
    z_re = scale * re0
    z_im = scale * im0
    # Verify circle membership exactly.
    if z_re * z_re + z_im * z_im != scale * scale:
        raise RuntimeError(f"atom circle membership violated: |z|^2 != scale^2 for u={uf} T={Tf}")
    return {
        "u": uf, "T": Tf, "scale": scale,
        "z_re": z_re, "z_im": z_im,
        "u_str": str(uf), "T_str": str(Tf),
        "z_re_str": str(z_re), "z_im_str": str(z_im),
    }


def check_atom(atom: dict) -> dict:
    """Revalidate atom dict (exact), return derived (scale,z_re,z_im). Raises."""
    if not isinstance(atom, dict):
        raise ValueError(f"atom must be dict, got {type(atom).__name__}")
    for k in ("u", "T"):
        if k not in atom:
            raise ValueError(f"atom missing {k!r}")
    uf = _as_fraction_exact("atom.u", atom["u"])
    Tf = _require_T(atom["T"])
    if abs(uf) > Tf / 2:
        raise ValueError(f"atom angle sufficient bound violated: |u|={uf} > T/2={Tf/2}")
    scale = Fraction(1, 1) - Tf * Tf
    den = Fraction(1, 1) + uf * uf
    z_re = scale * ((Fraction(1, 1) - uf * uf) / den)
    z_im = scale * ((Fraction(2, 1) * uf) / den)
    # If dict carries z, must match exactly (tamper check).
    for k, v in (("z_re", z_re), ("z_im", z_im), ("scale", scale)):
        if k in atom:
            av = _as_fraction_exact(f"atom.{k}", atom[k])
            if av != v:
                raise ValueError(f"atom {k} {av} != derived {v} (tamper rejected)")
    if z_re * z_re + z_im * z_im != scale * scale:
        raise RuntimeError("atom circle check failed (unreachable)")
    return {"u": uf, "T": Tf, "scale": scale, "z_re": z_re, "z_im": z_im}


def _cpow(z_re: Fraction, z_im: Fraction, k: int) -> tuple[Fraction, Fraction]:
    """Exact (z_re+i z_im)^k for k>=0 (Fractions). k int."""
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError(f"k must be nonnegative int, got {k!r}")
    re, im = Fraction(1, 1), Fraction(0, 1)
    for _ in range(k):
        nre = re * z_re - im * z_im
        nim = re * z_im + im * z_re
        re, im = nre, nim
    return (re, im)


# --- residual / mass / lambda / variance ----------------------------------

def _validate_coeffs(coeffs) -> list[tuple[Fraction, Fraction]]:
    """Validate coeffs as list/tuple of (re,im) int/Fraction pairs, return Fractions."""
    if not isinstance(coeffs, (list, tuple)):
        raise ValueError(f"coeffs must be list/tuple, got {type(coeffs).__name__}")
    if len(coeffs) == 0:
        raise ValueError("empty coeffs")
    if len(coeffs) > MAX_ATOMS:
        raise ValueError(f"too many atoms {len(coeffs)} > MAX_ATOMS={MAX_ATOMS}")
    out: list[tuple[Fraction, Fraction]] = []
    for j, c in enumerate(coeffs):
        if not isinstance(c, (list, tuple)) or len(c) != 2:
            raise ValueError(f"coeffs[{j}] must be (re,im) pair, got {c!r}")
        re = _as_fraction_exact(f"coeffs[{j}].re", c[0])
        im = _as_fraction_exact(f"coeffs[{j}].im", c[1])
        out.append((re, im))
    return out


def residual_of_coeffs(atoms: list[dict], coeffs, n: int) -> dict:
    """Fresh exact residual max_{k=0..n}|Re sum c z^k - delta_{k0}| (Fractions).

    atoms: list of atom dicts (checked). coeffs: list of (re,im). n: int.
    Returns dict with b (Fraction max), diffs list, per-k sums.
    """
    n = _require_n(n)
    if not isinstance(atoms, (list, tuple)):
        raise ValueError("atoms must be list/tuple")
    if len(atoms) != len(coeffs):
        raise ValueError(f"atoms/coeffs length mismatch {len(atoms)} vs {len(coeffs)}")
    if len(atoms) == 0 or len(atoms) > MAX_ATOMS:
        raise ValueError(f"atoms count out of range 1..{MAX_ATOMS}")
    # Check atoms exactly, collect z.
    zs: list[tuple[Fraction, Fraction]] = []
    Ts = set()
    for a in atoms:
        ck = check_atom(a)
        zs.append((ck["z_re"], ck["z_im"]))
        Ts.add(ck["T"])
    if len(Ts) != 1:
        raise ValueError(f"all atoms must share common T (bounded prototype), got {Ts}")
    cs = _validate_coeffs(coeffs)
    diffs: list[Fraction] = []
    sums: list[Fraction] = []
    for k in range(n + 1):
        # S_k = Re sum_j c_j z_j^k
        tot = Fraction(0, 1)
        for (re_c, im_c), (zr, zi) in zip(cs, zs):
            pr, pi = _cpow(zr, zi, k)
            # Re((re+i im)(pr+i pi)) = re*pr - im*pi
            tot += re_c * pr - im_c * pi
        target = Fraction(1, 1) if k == 0 else Fraction(0, 1)
        diff = abs(tot - target)
        diffs.append(diff)
        sums.append(tot)
    b = max(diffs)
    return {"b": b, "b_str": str(b), "diffs": diffs, "diffs_str": [str(d) for d in diffs],
            "sums": sums, "n": n, "T": list(Ts)[0]}


def mass_of_coeffs(coeffs) -> Fraction:
    """Exact l1 mass sum(|re|+|im|) (Fraction, conservative for sum|c|)."""
    cs = _validate_coeffs(coeffs)
    return sum(abs(re) + abs(im) for re, im in cs)


def lambda_of_target(r_frac, n: int) -> Fraction:
    """Exact lambda=n s^2 with s^2=1-a^2 (Fractions)."""
    n = _require_n(n)
    info = dominant_info(tuple(r_frac) if not isinstance(r_frac, tuple) else r_frac)
    # validate_rational_target already ensures unit; re-derive s2 exactly:
    return Fraction(n, 1) * info["s2"]


def variance_bound(mass: Fraction, lam: Fraction, T: Fraction) -> dict:
    """Exact conservative V=mass^2 3^{ceil(9 lam T^2)} (Fractions, e<3).

    lam>=0 Fraction, T in (0,1/4], mass>=0 Fraction. ceil0 gives 1 (3^0).
    Returns dict with V, ceil_exp, pow3.
    """
    if not isinstance(mass, Fraction) or mass < 0:
        raise ValueError(f"mass must be nonnegative Fraction, got {mass!r}")
    if not isinstance(lam, Fraction) or lam < 0:
        raise ValueError(f"lam must be nonnegative Fraction, got {lam!r}")
    T = _require_T(T)
    e = Fraction(9, 1) * lam * T * T
    if e < 0:
        raise RuntimeError("e negative (unreachable)")
    if e == 0:
        ceil_e = 0
    else:
        # e>0 Fraction p/q, ceil=(p+q-1)//q
        p, q = e.numerator, e.denominator
        ceil_e = (p + q - 1) // q
    if ceil_e < 0:
        raise RuntimeError("ceil negative (unreachable)")
    # Guard enormous exponents (bounded prototype): ceil_e bred from MAX_N is small;
    # refuse absurd (>1000) to avoid huge ints before callbacks.
    if ceil_e > 1000:
        raise ValueError(f"variance exponent {ceil_e} exceeds bounded-prototype guard 1000 (lam={lam} T={T})")
    pow3 = pow(3, ceil_e)
    V = mass * mass * pow3
    return {"V": V, "V_str": str(V), "ceil_exp": ceil_e, "pow3": pow3, "pow3_str": str(pow3),
            "e": e, "e_str": str(e)}


def single_qubit_bound_C(s2: Fraction, T: Fraction) -> Fraction:
    """Exact single-qubit operator bound C=1+9 s^2 T^2 (Fraction)."""
    if not isinstance(s2, Fraction) or s2 < 0:
        raise ValueError(f"s2 must be nonnegative Fraction, got {s2!r}")
    T = _require_T(T)
    return Fraction(1, 1) + Fraction(9, 1) * s2 * T * T


# --- single-qubit moments (exact complex Fractions) -----------------------
# Matrices as 2x2 of (re,im) Fractions. Pauli Y carries i.

def _cadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def _cmul(a, b):
    # (ar+i ai)(br+i bi) = (ar br-ai bi)+i(ar bi+ai br)
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def _cscale_real(p: Fraction, a) -> tuple[Fraction, Fraction]:
    return (p * a[0], p * a[1])


def _pauli_matrices() -> dict:
    F0 = Fraction(0, 1)
    F1 = Fraction(1, 1)
    Fm1 = Fraction(-1, 1)
    I = [[(F1, F0), (F0, F0)], [(F0, F0), (F1, F0)]]
    X = [[(F0, F0), (F1, F0)], [(F1, F0), (F0, F0)]]
    # Y = [[0,-i],[i,0]]
    Y = [[(F0, F0), (F0, Fm1)], [(F0, F1), (F0, F0)]]
    Z = [[(F1, F0), (F0, F0)], [(F0, F0), (Fm1, F0)]]
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def _mat_add(A, B):
    return [[_cadd(A[i][j], B[i][j]) for j in range(2)] for i in range(2)]


def _mat_scale_real(p: Fraction, A):
    return [[_cscale_real(p, A[i][j]) for j in range(2)] for i in range(2)]


def _projector(axis: str, o: int):
    """Exact (I+o P_axis)/2 as 2x2 (re,im) Fractions. o=+-1 int."""
    if axis not in ("X", "Y", "Z"):
        raise ValueError(f"axis {axis!r}")
    if _is_bool(o) or not isinstance(o, int) or o not in (1, -1):
        raise ValueError(f"outcome must be +-1 int (bool/float rejected), got {o!r}")
    PM = _pauli_matrices()
    I = PM["I"]
    P = PM[axis]
    of = Fraction(o, 1)
    out = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            # (I+oP)/2
            re = (I[i][j][0] + of * P[i][j][0]) / 2
            im = (I[i][j][1] + of * P[i][j][1]) / 2
            out[i][j] = (re, im)
    return out


def _h_for_axis(r_frac: tuple[Fraction, Fraction, Fraction], dom_idx: int,
                axis: str, o: int, z_re: Fraction, z_im: Fraction) -> tuple[Fraction, Fraction]:
    """Exact local complex score h (re,im) Fractions for one qubit.

    r_frac validated triple, dom_idx 0/1/2, axis X/Y/Z, o=+-1, z Fractions.
    Longitudinal (axis==dom): h=1-2t/3+2 t c o/3. Transverse: h=1+4 t r_axis o.
    """
    if dom_idx not in (0, 1, 2):
        raise ValueError(f"dom_idx {dom_idx!r}")
    if axis not in ("X", "Y", "Z"):
        raise ValueError(f"axis {axis!r}")
    if _is_bool(o) or not isinstance(o, int) or o not in (1, -1):
        raise ValueError(f"o must be +-1 int (bool/float rejected), got {o!r}")
    t_re = Fraction(1, 1) - z_re
    t_im = Fraction(0, 1) - z_im
    ax_idx = AXIS_INDEX[axis]
    if ax_idx == dom_idx:
        c = r_frac[ax_idx]
        of = Fraction(o, 1)
        # re = 1-2 t_re/3+2 c o t_re/3 ; im = -2 t_im/3+2 c o t_im/3
        re = Fraction(1, 1) - Fraction(2, 3) * t_re + Fraction(2, 3) * c * of * t_re
        im = Fraction(0, 1) - Fraction(2, 3) * t_im + Fraction(2, 3) * c * of * t_im
        return (re, im)
    else:
        ra = r_frac[ax_idx]
        of = Fraction(o, 1)
        re = Fraction(1, 1) + Fraction(4, 1) * t_re * ra * of
        im = Fraction(0, 1) + Fraction(4, 1) * t_im * ra * of
        return (re, im)


def single_qubit_moments(r, atom: dict) -> dict:
    """Exact single-qubit first/second moments + target operator (Fractions).

    r: exact-rational unit triple (validated inside). atom: checked dict.
    Returns dict with M1 (2x2), M2 (2x2), Az (2x2), C (Fraction bound),
    s2, dom_idx, z. All entries (re,im) Fractions except C/s2 Fractions.
    """
    r_frac = validate_rational_target(r)
    ck = check_atom(atom)
    z_re, z_im = ck["z_re"], ck["z_im"]
    info = dominant_info(r_frac)
    dom_idx = info["dom_idx"]
    s2 = info["s2"]
    T = ck["T"]
    C = single_qubit_bound_C(s2, T)
    F0 = Fraction(0, 1)
    M1 = [[(F0, F0), (F0, F0)], [(F0, F0), (F0, F0)]]
    M2 = [[(F0, F0), (F0, F0)], [(F0, F0), (F0, F0)]]
    # Axis probs: longitudinal 3/4, transverses 1/8.
    for ax in ("X", "Y", "Z"):
        p = Fraction(3, 4) if AXIS_INDEX[ax] == dom_idx else Fraction(1, 8)
        for o in (1, -1):
            h_re, h_im = _h_for_axis(r_frac, dom_idx, ax, o, z_re, z_im)
            Pi = _projector(ax, o)
            # M1 += p * Pi * h
            for i in range(2):
                for j in range(2):
                    # Pi*h
                    pr, pi = Pi[i][j]
                    # (pr+i pi)(h_re+i h_im)
                    cr = pr * h_re - pi * h_im
                    ci = pr * h_im + pi * h_re
                    M1[i][j] = (M1[i][j][0] + p * cr, M1[i][j][1] + p * ci)
            # M2 += p * Pi * |h|^2 (|h|^2 real)
            w = h_re * h_re + h_im * h_im
            for i in range(2):
                for j in range(2):
                    pr, pi = Pi[i][j]
                    M2[i][j] = (M2[i][j][0] + p * pr * w, M2[i][j][1] + p * pi * w)
    # Az = tau_+ + z tau_- with R from r.
    PM = _pauli_matrices()
    I = PM["I"]
    X, Y, Z = PM["X"], PM["Y"], PM["Z"]
    # R = rx X + ry Y + rz Z (rx real Fractions): scale each matrix by real.
    R = [[(F0, F0), (F0, F0)], [(F0, F0), (F0, F0)]]
    for coeff, P in ((r_frac[0], X), (r_frac[1], Y), (r_frac[2], Z)):
        for i in range(2):
            for j in range(2):
                # coeff real * (pr+i pi) = (coeff*pr, coeff*pi)
                R[i][j] = (R[i][j][0] + coeff * P[i][j][0], R[i][j][1] + coeff * P[i][j][1])
    # tau_+ = (I+R)/2, tau_-=(I-R)/2
    tau_p = [[None, None], [None, None]]
    tau_m = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            tau_p[i][j] = ((I[i][j][0] + R[i][j][0]) / 2, (I[i][j][1] + R[i][j][1]) / 2)
            tau_m[i][j] = ((I[i][j][0] - R[i][j][0]) / 2, (I[i][j][1] - R[i][j][1]) / 2)
    Az = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            # tau_p + z*tau_m ; z*tau_m = (zr+i zi)(mr+i mi)
            mr, mi = tau_m[i][j]
            ztm_re = z_re * mr - z_im * mi
            ztm_im = z_re * mi + z_im * mr
            Az[i][j] = (tau_p[i][j][0] + ztm_re, tau_p[i][j][1] + ztm_im)
    return {"M1": M1, "M2": M2, "Az": Az, "C": C, "s2": s2, "dom_idx": dom_idx,
            "z_re": z_re, "z_im": z_im, "T": T, "r": r_frac}


def check_psd_bound(M2, C: Fraction) -> dict:
    """Exact 2x2 PSD check for C*I-M2 via minors (Fractions, no float).

    M2: 2x2 of (re,im) Fractions, Hermitian (diagonal imag 0). C: Fraction.
    Returns dict with a,d,off_re,off_im,det,ok.
    """
    if not isinstance(C, Fraction):
        raise ValueError("C must be Fraction")
    # Validate shape
    if not (isinstance(M2, (list, tuple)) and len(M2) == 2 and all(len(row) == 2 for row in M2)):
        raise ValueError("M2 must be 2x2")
    for i in range(2):
        for j in range(2):
            if not (isinstance(M2[i][j], (list, tuple)) and len(M2[i][j]) == 2):
                raise ValueError("M2 entries must be (re,im)")
            for v in M2[i][j]:
                if not isinstance(v, Fraction):
                    raise ValueError("M2 entries must be Fractions")
    # Diagonal imag must be exactly 0 (since Pi diagonal real, w real).
    if M2[0][0][1] != 0 or M2[1][1][1] != 0:
        raise ValueError(f"M2 diagonal imag nonzero: {M2[0][0]}, {M2[1][1]} (Hermitian violation)")
    # Hermitian off-diagonal: M2[1][0] == conj(M2[0][1])
    if M2[1][0][0] != M2[0][1][0] or M2[1][0][1] != -M2[0][1][1]:
        raise ValueError(f"M2 not Hermitian: {M2[0][1]} vs {M2[1][0]}")
    a = C - M2[0][0][0]
    d = C - M2[1][1][0]
    off_re = M2[0][1][0]
    off_im = M2[0][1][1]
    # C I - M2 off = -off
    det = a * d - (off_re * off_re + off_im * off_im)
    ok = (a >= 0) and (d >= 0) and (det >= 0)
    return {"a": a, "d": d, "off_re": off_re, "off_im": off_im,
            "det": det, "ok": bool(ok),
            "a_str": str(a), "d_str": str(d), "det_str": str(det)}


def matrices_equal(A, B) -> bool:
    """Exact equality of 2x2 (re,im) Fraction matrices."""
    for i in range(2):
        for j in range(2):
            if A[i][j][0] != B[i][j][0] or A[i][j][1] != B[i][j][1]:
                return False
    return True


# --- finite-grid LP proposal (float, diagnostic only) ---------------------

def _float_pow(zr: float, zi: float, k: int) -> tuple[float, float]:
    re, im = 1.0, 0.0
    for _ in range(k):
        nre = re * zr - im * zi
        nim = re * zi + im * zr
        re, im = nre, nim
    return (re, im)


def solve_finite_lp_float(atoms_z_float: list[tuple[float, float]], n: int,
                          beta_tight: float) -> dict:
    """Float LP proposal minimizing mass subject to bias<=beta_tight (diagnostic).

    atoms_z_float: list of (zr,zi) floats. n: int. beta_tight: float>0.
    Variables [a(M),b(M),u(M),v(M)], minimize sum u+v subject to bias + abs.
    Uses SciPy HiGHS. Float feasibility is NOT a certificate.
    Returns dict with a_float,b_float,u,v,objective,versions,times.
    """
    import numpy as np

    if not isinstance(atoms_z_float, (list, tuple)) or len(atoms_z_float) == 0:
        raise ValueError("atoms_z_float nonempty required")
    M = len(atoms_z_float)
    if M > MAX_ATOMS:
        raise ValueError(f"M={M} > MAX_ATOMS")
    if not isinstance(n, int) or n < 1 or n > MAX_N:
        raise ValueError(f"n {n!r}")
    if not isinstance(beta_tight, float) or not (0.0 < beta_tight < 1.0):
        raise ValueError(f"beta_tight {beta_tight!r}")
    # Build X,Y matrices: X[j,k]=Re(z_j^k), Y=Im
    Xmat = np.zeros((M, n + 1), dtype=np.float64)
    Ymat = np.zeros((M, n + 1), dtype=np.float64)
    for j, (zr, zi) in enumerate(atoms_z_float):
        for k in range(n + 1):
            pr, pi = _float_pow(float(zr), float(zi), k)
            if not (math.isfinite(pr) and math.isfinite(pi)):
                raise ValueError("nonfinite pow")
            Xmat[j, k] = pr
            Ymat[j, k] = pi
    d = 4 * M
    c = np.zeros(d, dtype=np.float64)
    c[2 * M:] = 1.0
    rows = []
    rhs = []
    # Bias constraints for each k.
    for k in range(n + 1):
        delta = 1.0 if k == 0 else 0.0
        r1 = np.zeros(d, dtype=np.float64)
        # sum a_j X - b_j Y - delta <= beta
        for j in range(M):
            r1[j] = Xmat[j, k]
            r1[M + j] = -Ymat[j, k]
        rows.append(r1)
        rhs.append(delta + beta_tight)
        r2 = np.zeros(d, dtype=np.float64)
        for j in range(M):
            r2[j] = -Xmat[j, k]
            r2[M + j] = Ymat[j, k]
        rows.append(r2)
        rhs.append(-delta + beta_tight)
    # Abs constraints: a-u<=0, -a-u<=0, b-v<=0, -b-v<=0
    for j in range(M):
        for (idx_a, idx_u, sgn) in ((j, 2 * M + j, 1.0), (j, 2 * M + j, -1.0)):
            r = np.zeros(d, dtype=np.float64)
            r[idx_a] = sgn
            r[idx_u] = -1.0
            rows.append(r)
            rhs.append(0.0)
        for (idx_b, idx_v, sgn) in ((M + j, 3 * M + j, 1.0), (M + j, 3 * M + j, -1.0)):
            r = np.zeros(d, dtype=np.float64)
            r[idx_b] = sgn
            r[idx_v] = -1.0
            rows.append(r)
            rhs.append(0.0)
    import numpy as _np

    A_ub = _np.vstack(rows).astype(_np.float64)
    b_ub = _np.array(rhs, dtype=_np.float64)
    bounds = [(None, None)] * (2 * M) + [(0.0, None)] * (2 * M)
    if not (bool(_np.isfinite(A_ub).all()) and bool(_np.isfinite(b_ub).all()) and bool(_np.isfinite(c).all())):
        raise ValueError("nonfinite LP data")
    t0 = time.perf_counter()
    try:
        import scipy.optimize as _opt
        import scipy as _scipy
        scipy_ver = _scipy.__version__
    except Exception as e:
        raise RuntimeError(f"SciPy required for LP proposal: {e}") from e
    try:
        res = _opt.linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds,
                           method="highs", options={"presolve": True})
    except Exception as e:
        raise RuntimeError(f"linprog raised {type(e).__name__}: {e}") from e
    t1 = time.perf_counter()
    if not bool(getattr(res, "success", False)) or int(getattr(res, "status", -1)) != 0:
        raise RuntimeError(
            f"HiGHS LP failed: success={getattr(res,'success',None)} "
            f"status={getattr(res,'status',None)} message={getattr(res,'message',None)}"
        )
    x = _np.asarray(res.x, dtype=_np.float64)
    if x.shape != (d,) or not bool(_np.isfinite(x).all()):
        raise RuntimeError("HiGHS returned nonfinite/wrong-shape")
    a_f = [float(v) for v in x[:M]]
    b_f = [float(v) for v in x[M:2 * M]]
    u_f = [float(v) for v in x[2 * M:3 * M]]
    v_f = [float(v) for v in x[3 * M:]]
    fun = float(getattr(res, "fun", float(_np.dot(c, x))))
    viol = float(_np.max(A_ub @ x - b_ub)) if A_ub.size else 0.0
    return {"M": M, "n": n, "beta_tight": beta_tight,
            "a_float": a_f, "b_float": b_f, "u_float": u_f, "v_float": v_f,
            "objective": fun, "max_violation": viol,
            "solve_time_sec": t1 - t0,
            "scipy_version": scipy_ver, "numpy_version": _np.__version__,
            "python_version": platform.python_version()}


def _round_one_to_den(gw: Fraction, den: int) -> tuple[Fraction, Fraction]:
    if den <= 0:
        raise ValueError("den must be positive")
    f = gw * den
    num, dd = f.numerator, f.denominator
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


def round_coeffs_float(a_float: list[float], b_float: list[float],
                       den: int = ROUND_DEN_DEFAULT) -> dict:
    """Round float proposal (exact binary) to denominator den (ties to even).

    Returns dict with coeffs (list of (re,im) Fractions), maxdiff, den.
    """
    if _is_bool(den) or not isinstance(den, int) or den <= 0:
        raise ValueError(f"den must be positive int, got {den!r}")
    if len(a_float) != len(b_float):
        raise ValueError("a/b length mismatch")
    coeffs: list[tuple[Fraction, Fraction]] = []
    maxdiff = Fraction(0, 1)
    for af, bf in zip(a_float, b_float):
        for v in (af, bf):
            if _is_bool(v) or not isinstance(v, (int, float)):
                raise ValueError(f"float coeff must be number, got {v!r}")
            if isinstance(v, float) and not math.isfinite(v):
                raise ValueError(f"coeff nonfinite {v!r}")
        ar = Fraction(af) if isinstance(af, float) else Fraction(af, 1) if isinstance(af, int) else af
        br = Fraction(bf) if isinstance(bf, float) else Fraction(bf, 1) if isinstance(bf, int) else bf
        gr, dr = _round_one_to_den(ar, den)
        gi, di = _round_one_to_den(br, den)
        coeffs.append((gr, gi))
        if dr > maxdiff:
            maxdiff = dr
        if di > maxdiff:
            maxdiff = di
    return {"coeffs": coeffs, "coeffs_str": [(str(r), str(i)) for r, i in coeffs],
            "maxdiff": maxdiff, "maxdiff_str": str(maxdiff), "den": den}


# --- frozen plan ----------------------------------------------------------

@dataclass(frozen=True)
class HomogeneousPlan:
    """Frozen pre-data homogeneous design (immutable, deep-copied).

    Fields tuples/Fractions only. Construct ONLY via plan_homogeneous which freshly
    computes residual/mass/lambda/V/dom. run_... revalidates before any callback/RNG
    via canonical reconstruction and uses only canonical snapshot. Mutating external
    lists or attempting plan.field=... cannot alter collection. Dict plans rejected.
    """

    r: tuple[Fraction, Fraction, Fraction]
    n: int
    T: Fraction
    beta: Fraction
    us: tuple[Fraction, ...]
    coeffs_re: tuple[Fraction, ...]
    coeffs_im: tuple[Fraction, ...]
    b_cert: Fraction
    mass: Fraction
    lam: Fraction
    V: Fraction
    ceil_exp: int
    C_single: Fraction
    dom_idx: int
    den: int

    def __getitem__(self, key):
        return getattr(self, key)


def plan_homogeneous(r, n: int, T, us, coeffs, beta, *,
                     den: int = ROUND_DEN_DEFAULT) -> HomogeneousPlan:
    """Freeze homogeneous plan with fresh exact bindings (Fractions).

    Args:
      r: exact-rational unit triple (int/Fraction, sum squares EXACTLY 1).
      n: homogeneous qubit count 1..MAX_N.
      T: Fraction 0<T<=1/4 (common radius).
      us: list/tuple of u Fractions (each |u|<=T/2), one per atom.
      coeffs: list of (re,im) int/Fraction pairs, same length as us.
      beta: Fraction budget 0<beta<1; requires fresh b_cert<=beta else ValueError.
      den: rounding denominator provenance (stored; coeffs used exactly regardless).

    Returns frozen HomogeneousPlan. Caller must then use plan_copies for counts and
    run_homogeneous_streaming for data (exactly N fresh iid copies, no reuse).
    """
    r_frac = validate_rational_target(r)
    n = _require_n(n)
    Tf = _require_T(T)
    bf = _require_beta(beta)
    if _is_bool(den) or not isinstance(den, int) or den <= 0:
        raise ValueError(f"den must be positive int, got {den!r}")
    if not isinstance(us, (list, tuple)) or len(us) == 0:
        raise ValueError("us must be nonempty list/tuple")
    if len(us) > MAX_ATOMS:
        raise ValueError(f"too many atoms {len(us)}")
    if not isinstance(coeffs, (list, tuple)) or len(coeffs) != len(us):
        raise ValueError(f"coeffs length {len(coeffs) if isinstance(coeffs,(list,tuple)) else '?'} vs us {len(us)}")
    # Build atoms exactly.
    atoms = [atom_from_u(u, Tf) for u in us]
    cs = _validate_coeffs(coeffs)
    res = residual_of_coeffs(atoms, cs, n)
    b_cert = res["b"]
    if b_cert > bf:
        raise ValueError(
            f"fresh residual b={b_cert} ({float(b_cert):.6g}) exceeds budget beta={bf} "
            f"(infeasible grid/rounding; report honestly, no silent weakening)"
        )
    mass = mass_of_coeffs(cs)
    info = dominant_info(r_frac)
    lam = Fraction(n, 1) * info["s2"]
    vb = variance_bound(mass, lam, Tf)
    V = vb["V"]
    C_single = single_qubit_bound_C(info["s2"], Tf)
    us_t = tuple(_as_fraction_exact("u", u) for u in us)
    re_t = tuple(re for re, _ in cs)
    im_t = tuple(im for _, im in cs)
    return HomogeneousPlan(
        r=tuple(r_frac), n=int(n), T=Tf, beta=bf,
        us=us_t, coeffs_re=re_t, coeffs_im=im_t,
        b_cert=b_cert, mass=mass, lam=lam, V=V,
        ceil_exp=int(vb["ceil_exp"]), C_single=C_single,
        dom_idx=int(info["dom_idx"]), den=int(den),
    )


def plan_copies(plan: HomogeneousPlan, eps) -> dict:
    """Count-only planning: tol=eps-b>0, m=ceil(4V/tol^2), N=3m (exact).

    plan: frozen HomogeneousPlan. eps: int/Fraction with eps>b_cert.
    Canonically revalidates ALL stored bindings (V/b_cert/ceil_exp/C_single/den
    included) before any count calculation; altered derived values rejected.
    Returns dict with tol,m,N (ints/Fractions). No callbacks. Huge N returned
    (not failed) here; run_... enforces max_copies before callbacks.
    """
    if not isinstance(plan, HomogeneousPlan):
        raise ValueError("plan must be frozen HomogeneousPlan (dicts rejected)")
    # Canonical revalidation BEFORE trusting any stored V/b_cert (zero-copy
    # tamper rejected here, not just in streaming).
    canon = _canonical_revalidate(plan)
    b = canon["b"]
    V = canon["V"]
    ef = _require_eps(eps)
    if ef <= b:
        raise ValueError(f"eps={ef} must exceed b_cert={b} (tol>0 required)")
    tol = ef - b
    # m = ceil(4V/tol^2). V,tol Fractions => ratio Fraction.
    ratio = (Fraction(4, 1) * V) / (tol * tol)
    if ratio < 0:
        raise RuntimeError("ratio negative (unreachable)")
    if ratio == 0:
        m = 0 if V == 0 else 1
        # V==0 only if mass==0 (all coeffs zero) which gives b=1 for n>=0? Actually
        # mass 0 => b=1 (since sums 0 vs delta), so eps>b implies eps>1 impossible
        # (eps<=1). So V==0 unreachable for feasible plans; guard anyway.
        if V != 0:
            m = 1
    else:
        p, q = ratio.numerator, ratio.denominator
        m = (p + q - 1) // q
        if m < 1:
            m = 1
    N = 3 * m
    return {"tol": tol, "tol_str": str(tol), "m": int(m), "N": int(N),
            "V": V, "V_str": str(V), "b": b, "eps": ef}


def _canonical_revalidate(plan: HomogeneousPlan) -> dict:
    """Recompute canonical bindings from authoritative r/n/T/us/coeffs (exact).

    Returns dict with atoms, residual, mass, lam, V, C, dom. Raises ValueError on
    any mismatch/tamper (caller must call BEFORE any callback/RNG).
    """
    if not isinstance(plan, HomogeneousPlan):
        raise ValueError("plan must be frozen HomogeneousPlan")
    # Strict type checks (bool rejected).
    if not isinstance(plan.r, tuple) or len(plan.r) != 3:
        raise ValueError("plan.r must be 3-tuple")
    for v in plan.r:
        if not isinstance(v, Fraction):
            raise ValueError(f"plan.r entries must be Fractions, got {type(v).__name__}")
    if _is_bool(plan.n) or not isinstance(plan.n, int):
        raise ValueError("plan.n must be int")
    if not isinstance(plan.T, Fraction) or not isinstance(plan.beta, Fraction):
        raise ValueError("plan.T/beta must be Fractions")
    if not isinstance(plan.us, tuple) or not isinstance(plan.coeffs_re, tuple) or not isinstance(plan.coeffs_im, tuple):
        raise ValueError("plan.us/coeffs must be tuples (mutable lists rejected)")
    if len(plan.us) != len(plan.coeffs_re) or len(plan.us) != len(plan.coeffs_im):
        raise ValueError("plan us/coeffs length mismatch")
    if not isinstance(plan.b_cert, Fraction) or not isinstance(plan.mass, Fraction):
        raise ValueError("plan b/mass must be Fractions")
    if not isinstance(plan.lam, Fraction) or not isinstance(plan.V, Fraction):
        raise ValueError("plan lam/V must be Fractions")
    if not isinstance(plan.C_single, Fraction):
        raise ValueError(f"plan.C_single must be Fraction, got {type(plan.C_single).__name__}")
    if _is_bool(plan.ceil_exp) or not isinstance(plan.ceil_exp, int):
        raise ValueError(f"plan.ceil_exp must be int (bool rejected), got {plan.ceil_exp!r}")
    if _is_bool(plan.den) or not isinstance(plan.den, int) or plan.den <= 0:
        raise ValueError(f"plan.den must be positive int (bool rejected), got {plan.den!r}")
    if _is_bool(plan.dom_idx) or not isinstance(plan.dom_idx, int):
        raise ValueError("plan.dom_idx must be int")
    # Recompute from authoritative raws.
    r_frac = validate_rational_target(list(plan.r))
    n = _require_n(plan.n)
    Tf = _require_T(plan.T)
    bf = _require_beta(plan.beta)
    # us/coeffs immutable snapshots (deep copy semantics: tuples already immutable).
    us_list = list(plan.us)
    coeffs_list = list(zip(list(plan.coeffs_re), list(plan.coeffs_im)))
    for v in us_list:
        if not isinstance(v, Fraction):
            raise ValueError("plan.us entries must be Fractions")
    for re, im in coeffs_list:
        if not isinstance(re, Fraction) or not isinstance(im, Fraction):
            raise ValueError("plan coeffs must be Fractions")
    atoms = [atom_from_u(u, Tf) for u in us_list]
    res = residual_of_coeffs(atoms, coeffs_list, n)
    mass = mass_of_coeffs(coeffs_list)
    info = dominant_info(r_frac)
    lam = Fraction(n, 1) * info["s2"]
    vb = variance_bound(mass, lam, Tf)
    C_single = single_qubit_bound_C(info["s2"], Tf)
    # Compare to stored (reject understated/tampered).
    if res["b"] != plan.b_cert:
        raise ValueError(f"plan b_cert {plan.b_cert} != fresh {res['b']} (tamper/understatement rejected)")
    if mass != plan.mass:
        raise ValueError(f"plan mass {plan.mass} != fresh {mass}")
    if lam != plan.lam:
        raise ValueError(f"plan lam {plan.lam} != fresh {lam}")
    if vb["V"] != plan.V:
        raise ValueError(f"plan V {plan.V} != fresh {vb['V']}")
    if C_single != plan.C_single:
        raise ValueError(f"plan C_single {plan.C_single} != fresh {C_single} (tamper rejected)")
    if int(vb["ceil_exp"]) != int(plan.ceil_exp):
        raise ValueError(f"plan ceil_exp {plan.ceil_exp} != fresh {vb['ceil_exp']} (tamper rejected)")
    if int(info["dom_idx"]) != int(plan.dom_idx):
        raise ValueError(f"plan dom {plan.dom_idx} != fresh {info['dom_idx']}")
    if res["b"] > bf:
        raise ValueError(f"fresh b {res['b']} exceeds beta {bf}")
    if tuple(r_frac) != tuple(plan.r):
        raise ValueError("plan.r != canonical")
    return {"r": r_frac, "n": n, "T": Tf, "beta": bf, "atoms": atoms,
            "coeffs": coeffs_list, "b": res["b"], "mass": mass, "lam": lam,
            "V": vb["V"], "ceil_exp": vb["ceil_exp"], "C": C_single,
            "dom_idx": int(info["dom_idx"]), "info": info, "res": res}


def _sample_axes_for_copy(n: int, dom_idx: int, rng) -> list[str]:
    """Sample n independent axes with 1/8,1/8,3/4 via rng.randrange(8) (exact).

    Mapping: draw d in 0..7; 0->transverse[0], 1->transverse[1], 2-7->dominant.
    Requires rng.randrange. Returns list of X/Y/Z strings.
    """
    if not hasattr(rng, "randrange"):
        raise ValueError("rng must have .randrange() (axis sampling via randrange(8))")
    transverse = [i for i in range(3) if i != dom_idx]
    axes: list[str] = []
    for _ in range(n):
        d = rng.randrange(8)
        if _is_bool(d) or not isinstance(d, int):
            raise ValueError(f"rng.randrange(8) must return int, got {d!r}")
        if d < 0 or d >= 8:
            raise ValueError(f"rng.randrange(8) out of range: {d}")
        if d == 0:
            axes.append(PAULI_AXES[transverse[0]])
        elif d == 1:
            axes.append(PAULI_AXES[transverse[1]])
        else:
            axes.append(PAULI_AXES[dom_idx])
    return axes


def _per_copy_real_score(r_frac, dom_idx: int, atoms_z: list[tuple[Fraction, Fraction]],
                         coeffs: list[tuple[Fraction, Fraction]],
                         axes: list[str], outcomes: list[int]) -> Fraction:
    """Exact per-copy real score Re(sum_j c_j prod_q h^{(q)}_j) (Fraction)."""
    if len(axes) != len(outcomes):
        raise ValueError("axes/outcomes length mismatch")
    n = len(axes)
    # For each atom, product over qubits.
    total = Fraction(0, 1)
    for (z_re, z_im), (cr, ci) in zip(atoms_z, coeffs):
        Hr, Hi = Fraction(1, 1), Fraction(0, 1)
        for ax, o in zip(axes, outcomes):
            if ax not in ("X", "Y", "Z"):
                raise ValueError(f"invalid axis {ax!r}")
            if _is_bool(o) or not isinstance(o, int) or o not in (1, -1):
                # outcomes must be ints +-1 (bool/float rejected: 1.0==1 passes
                # `in` check, so explicit int-type gate required before casts).
                raise ValueError(f"outcomes must be +-1 ints (bool/float rejected), got {o!r}")
            h_re, h_im = _h_for_axis(r_frac, dom_idx, ax, int(o), z_re, z_im)
            nHr = Hr * h_re - Hi * h_im
            nHi = Hr * h_im + Hi * h_re
            Hr, Hi = nHr, nHi
        # Re(c*H) = cr*Hr - ci*Hi
        total += cr * Hr - ci * Hi
    return total


def _clip01_frac(x: Fraction) -> Fraction:
    """Clip Fraction to [0,1] (exact)."""
    if not isinstance(x, Fraction):
        raise ValueError(f"clip input must be Fraction, got {type(x).__name__}")
    if x < 0:
        return Fraction(0, 1)
    if x > 1:
        return Fraction(1, 1)
    return x


def _outward_float_interval(lo: Fraction, hi: Fraction) -> list[float]:
    """Outward-rounded floats [lo_f, hi_f] with proved containment lo_f<=lo<=hi<=hi_f.

    lo,hi Fractions in [0,1] (already clipped). Uses exact
    Fraction(float) comparison and math.nextafter stepping. Preserves exact
    endpoints separately (caller keeps interval_exact); these floats are the
    outward display/containment pair. Proved: loop adjusts until
    Fraction(lo_f)<=lo and Fraction(hi_f)>=hi; each step moves one ulp outward,
    float(lo) is nearest so finitely many steps suffice (bounded loop).
    """
    if not isinstance(lo, Fraction) or not isinstance(hi, Fraction):
        raise ValueError("outward interval endpoints must be Fractions")
    if lo > hi:
        raise ValueError(f"outward interval inverted exact [{lo},{hi}] (unreachable after clip01)")
    lo_f = float(lo)
    hi_f = float(hi)
    if not (math.isfinite(lo_f) and math.isfinite(hi_f)):
        raise RuntimeError(f"outward float nonfinite for [{lo},{hi}]")
    # Step outward until exact containment proved by Fraction(float) comparison.
    for _ in range(1000):
        if Fraction(lo_f) <= lo:
            break
        lo_f = math.nextafter(lo_f, -math.inf)
    else:
        raise RuntimeError("outward lo adjustment did not converge")
    for _ in range(1000):
        if Fraction(hi_f) >= hi:
            break
        hi_f = math.nextafter(hi_f, math.inf)
    else:
        raise RuntimeError("outward hi adjustment did not converge")
    if not (Fraction(lo_f) <= lo <= hi <= Fraction(hi_f)):
        raise RuntimeError("outward containment failed (unreachable)")
    if not (lo_f <= hi_f):
        raise RuntimeError("outward floats inverted (unreachable)")
    return [lo_f, hi_f]


def run_homogeneous_streaming(plan: HomogeneousPlan, measure_fn, rng, eps,
                              max_copies: int = MAX_COPIES) -> dict:
    """Stream N=3m copies via proxy complex scores + median-of-means, clip.

    plan must be frozen HomogeneousPlan. At entry, BEFORE any callback/RNG,
    revalidates target/atoms/coeffs/bindings via canonical reconstruction,
    rejecting tampered/understated plans with zero draws. Computes tol=eps-b,
    m=ceil(4V/tol^2), N=3m; if N>max_copies raises BEFORE callbacks (huge requests
    fail). Then delegates to fresh axis sampling (randrange(8)) + full-copy
    measure_fn(axes)->outcomes (correlated outcomes required) for exactly N copies,
    exact Fraction block means, median then clip to [0,1]. Returns estimate,
    intervals, exact rationals, counts, timings.
    """
    t0 = time.perf_counter()
    if not isinstance(plan, HomogeneousPlan):
        raise ValueError("plan must be frozen HomogeneousPlan from plan_homogeneous (dicts rejected)")
    if not callable(measure_fn):
        raise ValueError("measure_fn must be callable")
    if not hasattr(rng, "randrange"):
        raise ValueError("rng must have .randrange()")
    if _is_bool(max_copies) or not isinstance(max_copies, int) or max_copies <= 0:
        raise ValueError(f"max_copies must be positive int, got {max_copies!r}")
    ef = _require_eps(eps)
    # Canonical revalidation EXACTLY ONCE before any callback/RNG (timed).
    t_r0 = time.perf_counter()
    canon = _canonical_revalidate(plan)
    t_r1 = time.perf_counter()
    r_frac = canon["r"]
    n = canon["n"]
    dom_idx = canon["dom_idx"]
    atoms = canon["atoms"]
    coeffs = canon["coeffs"]
    b = canon["b"]
    V = canon["V"]
    if ef <= b:
        raise ValueError(f"eps={ef} must exceed b_cert={b}")
    tol = ef - b
    # m,N exact.
    ratio = (Fraction(4, 1) * V) / (tol * tol)
    if ratio == 0:
        # V==0 unreachable for feasible (see plan_copies); guard.
        m = 1 if V != 0 else 1
        if V == 0:
            raise ValueError("V==0 with feasible b (unreachable)")
    else:
        p, q = ratio.numerator, ratio.denominator
        m = (p + q - 1) // q
        if m < 1:
            m = 1
    N = 3 * m
    if N > max_copies:
        raise ValueError(
            f"required N={N} (m={m}, V={V}, tol={tol}) exceeds max_copies={max_copies} "
            f"(huge request fails before callbacks; use plan_copies count-only or looser eps)"
        )
    # Zero callbacks/RNG so far verified by construction (no rng/measure calls above).
    atoms_z = [(a["z_re"], a["z_im"]) for a in atoms]
    # Streaming: 3 blocks, m copies each, exact Fraction accumulation.
    block_means: list[Fraction] = []
    total_calls = 0
    for blk in range(3):
        s = Fraction(0, 1)
        for _ in range(m):
            axes = _sample_axes_for_copy(n, dom_idx, rng)
            # Pass a COPY to the callback; scoring below uses the original `axes`
            # so callback mutation cannot corrupt the recorded measurement basis.
            outcomes = measure_fn(list(axes))
            total_calls += 1
            if not isinstance(outcomes, (list, tuple)) or len(outcomes) != n:
                raise ValueError(f"measure_fn must return length-{n} outcomes, got {outcomes!r}")
            # Validate outcomes ints +-1 (bool/float rejected) before scoring.
            for o in outcomes:
                if _is_bool(o) or not isinstance(o, int) or o not in (1, -1):
                    raise ValueError(f"outcomes must be +-1 ints (bool/float rejected), got {o!r}")
            sc = _per_copy_real_score(r_frac, dom_idx, atoms_z, coeffs, axes, list(outcomes))
            s += sc
        block_means.append(s / m)
    # Median of 3 (exact sort).
    srt = sorted(block_means)
    med = srt[1]
    # Clip estimate to [0,1] rationally.
    if med < 0:
        clipped = Fraction(0, 1)
    elif med > 1:
        clipped = Fraction(1, 1)
    else:
        clipped = med
    t1 = time.perf_counter()
    # Note: total error budget is b (bias) + tol (sampling). Median estimates
    # mu=Tr(sigma D) within tol, mu within b of fidelity. So fidelity in
    # [med-tol-b, med+tol+b]. Clip BOTH endpoints via clip01 (med outside [0,1]
    # would otherwise invert a one-sided clip).
    R_total = tol + b
    lo_exact = _clip01_frac(med - R_total)
    hi_exact = _clip01_frac(med + R_total)
    if lo_exact > hi_exact:
        raise RuntimeError(f"clipped interval inverted [{lo_exact},{hi_exact}] (unreachable)")
    lo_f, hi_f = _outward_float_interval(lo_exact, hi_exact)
    return {
        "estimate": float(clipped),
        "estimate_exact": str(clipped),
        "median_exact": str(med),
        "block_means_exact": [str(v) for v in block_means],
        "interval_exact": [str(lo_exact), str(hi_exact)],
        "interval": [lo_f, hi_f],
        "tol_exact": str(tol), "b_exact": str(b), "R_total_exact": str(R_total),
        "m": int(m), "N": int(N), "n": int(n), "dom_idx": int(dom_idx),
        "V_exact": str(V), "mass_exact": str(canon["mass"] if "mass" in canon else plan.mass),
        "calls": int(total_calls),
        "revalidate_time_sec": t_r1 - t_r0,
        "stream_time_sec": t1 - t0,
        "eps": str(ef), "max_copies": int(max_copies),
    }


def demo_proposal(n: int = 1, den: int = ROUND_DEN_DEFAULT) -> dict:
    """Tiny deterministic proposal helper (float LP diagnostic + exact cert).

    Uses T=1/4, beta=1/40, grid u=[-1/10,0,1/10] (all |u|<=1/8). Solves float LP
    with tight beta/2, rounds to den, certifies fresh exact residual. Returns
    dict with atoms/coeffs/b/mass/V floats+exact, solver info. Raises honestly on
    infeasibility (no silent weakening). For verifier/demo only; not a theorem.
    """
    from fractions import Fraction as _F

    T = Fraction(1, 4)
    beta = Fraction(1, 40)
    us = [_F(-1, 10), _F(0, 1), _F(1, 10)]
    atoms = [atom_from_u(u, T) for u in us]
    z_float = [(float(a["z_re"]), float(a["z_im"])) for a in atoms]
    beta_tight = float(beta) / 2.0
    sol = solve_finite_lp_float(z_float, int(n), beta_tight)
    rnd = round_coeffs_float(sol["a_float"], sol["b_float"], den)
    coeffs = rnd["coeffs"]
    res = residual_of_coeffs(atoms, coeffs, int(n))
    mass = mass_of_coeffs(coeffs)
    return {"T": T, "beta": beta, "us": us, "atoms": atoms,
            "solver": sol, "rounded": rnd, "coeffs": coeffs,
            "residual": res, "mass": mass, "n": int(n), "den": den}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bounded homogeneous-fidelity demo")
    parser.add_argument("--demo", action="store_true", help="run tiny n=1 loose-eps demo (exact-law, no sampling)")
    args = parser.parse_args()
    if args.demo:
        # Deterministic tiny demo: n=1, T=1/4, beta=1/40, loose eps=1/2, count-only
        # plus exact moment checks (no sampling, no huge loops).
        from fractions import Fraction as _F

        r = (_F(3, 5), _F(0, 1), _F(4, 5))
        print(f"target r={tuple(str(v) for v in r)}")
        info = dominant_info(r)
        print(f"dom={info['dom_axis']} a={info['a']} s2={info['s2']}")
        prop = demo_proposal(1)
        print(f"LP objective={prop['solver']['objective']:.6g} "
              f"viol={prop['solver']['max_violation']:.3g}")
        print(f"rounded b={prop['residual']['b']} mass={prop['mass']}")
        plan = plan_homogeneous(list(r), 1, _F(1, 4),
                                prop["us"], prop["coeffs"], _F(1, 40))
        print(f"plan b={plan.b_cert} mass={plan.mass} V={plan.V} lam={plan.lam}")
        counts = plan_copies(plan, _F(1, 2))
        print(f"loose eps=1/2: tol={counts['tol']} m={counts['m']} N={counts['N']}")
        # Exact single-qubit moments for first atom.
        mom = single_qubit_moments(list(r), prop["atoms"][0])
        chk = check_psd_bound(mom["M2"], mom["C"])
        print(f"M1==Az: {matrices_equal(mom['M1'], mom['Az'])} PSD ok: {chk['ok']} C={mom['C']}")
        sys.exit(0)
    else:
        parser.print_help()
        sys.exit(2)
