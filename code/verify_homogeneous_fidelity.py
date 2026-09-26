"""Independent deterministic validation for homogeneous-fidelity prototype.

Frozen small scope: exact-rational target validation (signed/tied/Pauli/invalid);
rational-circle atom membership/angle sufficient bound; independent exact first/second
POVM matrices (own reimplementation, NOT production) for n<=3 including mixed/entangled
sigma via explicit Born enumeration; dense eigvalsh support for all orientations
including large tilt (diagnostic, never certificate); exact rational residual/mass/V
plus 2x2 PSD minors (exact certificates, no floating tolerance); finite-grid LP
diagnostic (SciPy HiGHS float proposal) vs FRESH exact residual for n=1..4, T=1/4,
beta=1/40 with meaningful nonconstant accepted scores and honest infeasibility/large-weight
reporting; weak/constant fake rejection; tamper rejection before data (zero callbacks);
huge-copy fail-before-callbacks; tiny loose-eps sampling demo with honest Born backend
(simulation-labelled, never fake iid labelled physical); invalid inputs.

Run: python code/verify_homogeneous_fidelity.py
Writes homogeneous_fidelity_checks.json alongside. Exits nonzero on failure.
Requires NumPy/SciPy in .venv for LP+eigvalsh diagnostics only; exact certificates are
stdlib Fractions. No network.
"""

from __future__ import annotations

import dataclasses
import itertools
import json
import math
import platform
import random
import time
import tracemalloc
from fractions import Fraction
from pathlib import Path

import homogeneous_fidelity as hf

BITS_NOTE = "exact Fractions, no float certificate"
SEED = 20260908
TOL_EIG = 1e-9
TOL_BORN = 1e-9
BETA = Fraction(1, 40)
T_FIX = Fraction(1, 4)
DEN = 4096

# ---------------- own independent exact implementation (NOT production) -----

def own_as_frac(v) -> Fraction:
    if isinstance(v, bool):
        raise ValueError("bool")
    if isinstance(v, int):
        return Fraction(v, 1)
    if isinstance(v, Fraction):
        return v
    raise ValueError(f"need int/Fraction, got {type(v).__name__}")


def own_validate_target(r):
    if not isinstance(r, (list, tuple)) or len(r) != 3:
        raise ValueError("len")
    fs = [own_as_frac(v) for v in r]
    if fs[0]*fs[0]+fs[1]*fs[1]+fs[2]*fs[2] != 1:
        raise ValueError("not unit")
    return tuple(fs)


def own_dominant(r):
    abses = [abs(v) for v in r]
    a = max(abses)
    idx = min(i for i, av in enumerate(abses) if av == a)
    s2 = Fraction(1, 1) - a*a
    return {"idx": idx, "a": a, "s2": s2}


def own_atom(u, T):
    uf = own_as_frac(u)
    Tf = own_as_frac(T)
    if not (Fraction(0, 1) < Tf <= Fraction(1, 4)):
        raise ValueError("T")
    if abs(uf) > Tf/2:
        raise ValueError("|u|")
    scale = Fraction(1, 1) - Tf*Tf
    den = Fraction(1, 1) + uf*uf
    zr = scale*((Fraction(1, 1)-uf*uf)/den)
    zi = scale*((Fraction(2, 1)*uf)/den)
    assert zr*zr+zi*zi == scale*scale
    return {"u": uf, "T": Tf, "scale": scale, "z_re": zr, "z_im": zi}


def own_h(r, dom_idx, axis, o, zr, zi):
    t_re = Fraction(1, 1)-zr
    t_im = Fraction(0, 1)-zi
    ax_idx = {"X": 0, "Y": 1, "Z": 2}[axis]
    of = Fraction(o, 1)
    if ax_idx == dom_idx:
        c = r[ax_idx]
        re = Fraction(1, 1)-Fraction(2, 3)*t_re+Fraction(2, 3)*c*of*t_re
        im = Fraction(0, 1)-Fraction(2, 3)*t_im+Fraction(2, 3)*c*of*t_im
        return (re, im)
    else:
        ra = r[ax_idx]
        re = Fraction(1, 1)+Fraction(4, 1)*t_re*ra*of
        im = Fraction(0, 1)+Fraction(4, 1)*t_im*ra*of
        return (re, im)


def own_cpow(zr, zi, k):
    re, im = Fraction(1, 1), Fraction(0, 1)
    for _ in range(k):
        nre = re*zr-im*zi
        nim = re*zi+im*zr
        re, im = nre, nim
    return (re, im)


def own_residual(zs, coeffs, n):
    diffs = []
    for k in range(n+1):
        tot = Fraction(0, 1)
        for (cr, ci), (zr, zi) in zip(coeffs, zs):
            pr, pi = own_cpow(zr, zi, k)
            tot += cr*pr-ci*pi
        target = Fraction(1, 1) if k == 0 else Fraction(0, 1)
        diffs.append(abs(tot-target))
    return max(diffs), diffs


def own_M1_M2_Az(r, dom_idx, zr, zi):
    # Pauli exact with i as (re,im).
    F0, F1, Fm1 = Fraction(0, 1), Fraction(1, 1), Fraction(-1, 1)
    PM = {
        "I": [[(F1, F0), (F0, F0)], [(F0, F0), (F1, F0)]],
        "X": [[(F0, F0), (F1, F0)], [(F1, F0), (F0, F0)]],
        "Y": [[(F0, F0), (F0, Fm1)], [(F0, F1), (F0, F0)]],
        "Z": [[(F1, F0), (F0, F0)], [(F0, F0), (Fm1, F0)]],
    }
    def proj(ax, o):
        I, P = PM["I"], PM[ax]
        of = Fraction(o, 1)
        out = [[None, None], [None, None]]
        for i in range(2):
            for j in range(2):
                out[i][j] = ((I[i][j][0]+of*P[i][j][0])/2, (I[i][j][1]+of*P[i][j][1])/2)
        return out
    M1 = [[(F0, F0), (F0, F0)], [(F0, F0), (F0, F0)]]
    M2 = [[(F0, F0), (F0, F0)], [(F0, F0), (F0, F0)]]
    for ax in ("X", "Y", "Z"):
        p = Fraction(3, 4) if {"X": 0, "Y": 1, "Z": 2}[ax] == dom_idx else Fraction(1, 8)
        for o in (1, -1):
            hr, hi = own_h(r, dom_idx, ax, o, zr, zi)
            Pi = proj(ax, o)
            for i in range(2):
                for j in range(2):
                    pr, pi = Pi[i][j]
                    cr = pr*hr-pi*hi
                    ci = pr*hi+pi*hr
                    M1[i][j] = (M1[i][j][0]+p*cr, M1[i][j][1]+p*ci)
            w = hr*hr+hi*hi
            for i in range(2):
                for j in range(2):
                    pr, pi = Pi[i][j]
                    M2[i][j] = (M2[i][j][0]+p*pr*w, M2[i][j][1]+p*pi*w)
    # Az
    R = [[(F0, F0), (F0, F0)], [(F0, F0), (F0, F0)]]
    for coeff, key in ((r[0], "X"), (r[1], "Y"), (r[2], "Z")):
        P = PM[key]
        for i in range(2):
            for j in range(2):
                R[i][j] = (R[i][j][0]+coeff*P[i][j][0], R[i][j][1]+coeff*P[i][j][1])
    I = PM["I"]
    tau_p = [[((I[i][j][0]+R[i][j][0])/2, (I[i][j][1]+R[i][j][1])/2) for j in range(2)] for i in range(2)]
    tau_m = [[((I[i][j][0]-R[i][j][0])/2, (I[i][j][1]-R[i][j][1])/2) for j in range(2)] for i in range(2)]
    Az = [[None, None], [None, None]]
    for i in range(2):
        for j in range(2):
            mr, mi = tau_m[i][j]
            Az[i][j] = (tau_p[i][j][0]+zr*mr-zi*mi, tau_p[i][j][1]+zr*mi+zi*mr)
    return M1, M2, Az


# ---------------- float dense helpers (diagnostic only) -------------------

def float_pauli():
    I = [[1+0j, 0j], [0j, 1+0j]]
    X = [[0j, 1+0j], [1+0j, 0j]]
    Y = [[0j, -1j], [1j, 0j]]
    Z = [[1+0j, 0j], [0j, -1+0j]]
    return {"I": I, "X": X, "Y": Y, "Z": Z}


def kron(a, b):
    n, m = len(a), len(a[0])
    p, q = len(b), len(b[0])
    out = [[0j]*(m*q) for _ in range(n*p)]
    for i in range(n):
        for j in range(m):
            aij = a[i][j]
            if aij == 0:
                continue
            for kk in range(p):
                for ll in range(q):
                    out[i*p+kk][j*q+ll] = aij*b[kk][ll]
    return out


def mmul(a, b):
    n = len(a)
    out = [[0j]*n for _ in range(n)]
    for i in range(n):
        for k_ in range(n):
            aik = a[i][k_]
            if aik == 0:
                continue
            for j in range(n):
                out[i][j] += aik*b[k_][j]
    return out


def trace(a):
    return sum(a[i][i] for i in range(len(a)))


def tensor_list(mats):
    out = [[1+0j]]
    for m in mats:
        out = kron(out, m)
    return out


def complex_frac_to_float(pair) -> complex:
    return complex(float(pair[0]), float(pair[1]))


def mat_frac_to_float(M):
    return [[complex_frac_to_float(M[i][j]) for j in range(2)] for i in range(2)]


def bell_phiplus():
    import math as _m
    v = [1/_m.sqrt(2)+0j, 0j, 0j, 1/_m.sqrt(2)+0j]
    n = len(v)
    return [[v[i]*v[j].conjugate() for j in range(n)] for i in range(n)]


def ghz3():
    import math as _m
    dim = 8
    v = [0j]*dim
    v[0] = 1/_m.sqrt(2)
    v[7] = 1/_m.sqrt(2)
    return [[v[i]*v[j].conjugate() for j in range(dim)] for i in range(dim)]


def meas_proj_float(axis, o):
    PM = float_pauli()
    I, P = PM["I"], PM[axis]
    return [[(I[i][j]+o*P[i][j])/2.0 for j in range(2)] for i in range(2)]


def born_probs_float(sigma, axes):
    n = len(axes)
    dim = 1 << n
    probs: dict[tuple[int, ...], float] = {}
    total = 0.0
    for bits in itertools.product((1, -1), repeat=n):
        projs = [meas_proj_float(axes[j], bits[j]) for j in range(n)]
        P = tensor_list(projs)
        p = trace(mmul(sigma, P)).real
        if -1e-12 <= p < 0.0:
            p = 0.0
        if not math.isfinite(p) or p < -1e-9 or p > 1.0+1e-9:
            raise ValueError(f"Born {p}")
        probs[bits] = float(p)
        total += float(p)
    if abs(total-1.0) > 1e-9:
        raise ValueError(f"Born sum {total}")
    for k in list(probs.keys()):
        probs[k] /= total
    return probs


def make_honest_born_measure(sigma, n, backend_rng):
    """Honest simulation backend: explicit joint Born law per requested basis (SIMULATION-labelled)."""
    if not hasattr(backend_rng, "random"):
        raise ValueError("backend rng")
    def measure(axes):
        if len(axes) != n:
            raise ValueError("axes len")
        probs = born_probs_float(sigma, list(axes))
        u = float(backend_rng.random())
        acc = 0.0
        for bits in sorted(probs.keys(), key=lambda b: tuple(1 if x == 1 else 0 for x in b)):
            acc += probs[bits]
            if u < acc:
                return list(bits)
        return list(sorted(probs.keys())[-1])
    measure.is_simulation = True
    return measure


# ---------------- main validation -----------------------------------------

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
    print(f"validating homogeneous_fidelity with python {py_ver} numpy {numpy_ver} scipy {scipy_ver}")

    # ---- A. target validation signed/tied/Pauli/invalid ----
    try:
        good = [
            ((Fraction(0), Fraction(0), Fraction(1)), "PauliZ", 2, Fraction(1), Fraction(0)),
            ((Fraction(1), Fraction(0), Fraction(0)), "PauliX", 0, Fraction(1), Fraction(0)),
            ((Fraction(3, 5), Fraction(0), Fraction(4, 5)), "signed-345", 2, Fraction(4, 5), Fraction(9, 25)),
            ((Fraction(-3, 5), Fraction(0), Fraction(4, 5)), "neg-x", 2, Fraction(4, 5), Fraction(9, 25)),
            ((Fraction(2, 3), Fraction(2, 3), Fraction(1, 3)), "tie-223", 0, Fraction(2, 3), Fraction(5, 9)),
            ((Fraction(0), Fraction(3, 5), Fraction(-4, 5)), "neg-z", 2, Fraction(4, 5), Fraction(9, 25)),
        ]
        for r, label, exp_dom, exp_a, exp_s2 in good:
            try:
                rf = hf.validate_rational_target(list(r))
            except Exception as e:
                fail(f"target {label} rejected: {e}")
                continue
            info = hf.dominant_info(rf)
            if info["dom_idx"] != exp_dom or info["a"] != exp_a or info["s2"] != exp_s2:
                fail(f"target {label} dom {info} vs exp {(exp_dom,exp_a,exp_s2)}")
            # Own cross-check.
            orf = own_validate_target(list(r))
            if tuple(orf) != tuple(rf):
                fail(f"target {label} own mismatch")
            od = own_dominant(orf)
            if od["idx"] != exp_dom or od["a"] != exp_a or od["s2"] != exp_s2:
                fail(f"target {label} own dom {od}")
        # Pauli flag.
        if not hf.dominant_info(hf.validate_rational_target([0, 0, 1]))["is_pauli"]:
            fail("PauliZ not flagged Pauli")
        if hf.dominant_info(hf.validate_rational_target([Fraction(3,5),Fraction(0),Fraction(4,5)]))["is_pauli"]:
            fail("non-Pauli flagged Pauli")
        # Invalid: floats rejected, nonunit, zero, bool, wrong dims.
        def expect_err(fn, label):
            try:
                fn()
            except ValueError:
                return
            except Exception as e:
                fail(f"{label} wrong exc {type(e).__name__}: {e}")
                return
            fail(f"{label} did not raise")
        expect_err(lambda: hf.validate_rational_target([0.0, 0.0, 1.0]), "float target")
        expect_err(lambda: hf.validate_rational_target([Fraction(1,2),Fraction(1,2),Fraction(1,2)]), "nonunit")
        expect_err(lambda: hf.validate_rational_target([0, 0, 0]), "zero")
        expect_err(lambda: hf.validate_rational_target([True, Fraction(0), Fraction(1)]), "bool coord")
        expect_err(lambda: hf.validate_rational_target([(1,2), Fraction(0), Fraction(1)]), "tuple coord")
        expect_err(lambda: hf.validate_rational_target([]), "empty")
        expect_err(lambda: hf.atom_from_u(Fraction(1,2), T_FIX), "u bound")
        expect_err(lambda: hf.atom_from_u(Fraction(0), Fraction(1,2)), "T bound")
        expect_err(lambda: hf.atom_from_u(0.1, T_FIX), "float u")
        notes.append("targets incl. signed/tied/Pauli ok")
    except Exception as e:
        import traceback as _tb
        fail(f"target harness {_tb.format_exc()}")

    # ---- B. atom membership/angle ----
    try:
        for u in [Fraction(-1,8), Fraction(-1,10), Fraction(0), Fraction(1,16), Fraction(1,8)]:
            a = hf.atom_from_u(u, T_FIX)
            ck = hf.check_atom(a)
            if ck["z_re"]*ck["z_re"]+ck["z_im"]*ck["z_im"] != ck["scale"]*ck["scale"]:
                fail(f"atom circle {u}")
            if abs(ck["u"]) > ck["T"]/2:
                fail(f"atom angle {u}")
            # Own.
            oa = own_atom(u, T_FIX)
            if oa["z_re"] != ck["z_re"] or oa["z_im"] != ck["z_im"]:
                fail(f"atom own mismatch {u}")
            # Float angle diagnostic: |arg|<=T+1e-12.
            zf = complex(float(ck["z_re"]), float(ck["z_im"]))
            arg = abs(math.atan2(zf.imag, zf.real))
            if arg > float(T_FIX)+1e-12:
                fail(f"atom arg {u}: {arg} > T")
        notes.append("atoms circle+angle ok")
    except Exception as e:
        import traceback as _tb2
        fail(f"atom harness {_tb2.format_exc()}")

    # ---- C. exact first/second moments + PSD minors + eigvalsh + Born ----
    try:
        import numpy as _np
        test_rs = [
            (Fraction(0), Fraction(0), Fraction(1)),
            (Fraction(1), Fraction(0), Fraction(0)),
            (Fraction(3, 5), Fraction(0), Fraction(4, 5)),
            (Fraction(-3, 5), Fraction(0), Fraction(4, 5)),
            (Fraction(2, 3), Fraction(2, 3), Fraction(1, 3)),
            (Fraction(0), Fraction(3, 5), Fraction(-4, 5)),
        ]
        test_us = [Fraction(-1,10), Fraction(0), Fraction(1,10)]
        max_eig_gap = 0.0
        for r in test_rs:
            rf = hf.validate_rational_target(list(r))
            info = hf.dominant_info(rf)
            for u in test_us:
                atom = hf.atom_from_u(u, T_FIX)
                prod = hf.single_qubit_moments(list(r), atom)
                # Own independent.
                od = own_dominant(own_validate_target(list(r)))
                oa = own_atom(u, T_FIX)
                oM1, oM2, oAz = own_M1_M2_Az(tuple(rf), od["idx"], oa["z_re"], oa["z_im"])
                # Exact equality production vs own.
                if not hf.matrices_equal(prod["M1"], oM1):
                    fail(f"M1 mismatch r={r} u={u}")
                if not hf.matrices_equal(prod["M2"], oM2):
                    fail(f"M2 mismatch r={r} u={u}")
                if not hf.matrices_equal(prod["Az"], oAz):
                    fail(f"Az mismatch r={r} u={u}")
                # First moment exactly Az (theory Sec 1.1).
                if not hf.matrices_equal(prod["M1"], prod["Az"]):
                    fail(f"M1!=Az r={r} u={u}")
                # PSD minors exact (certificate, no float).
                chk = hf.check_psd_bound(prod["M2"], prod["C"])
                if not chk["ok"]:
                    fail(f"PSD minors failed r={r} u={u}: a={chk['a']} d={chk['d']} det={chk['det']}")
                # Eigvalsh diagnostic (support only).
                M2f = mat_frac_to_float(prod["M2"])
                arr = _np.array(M2f, dtype=complex)
                eig = _np.linalg.eigvalsh(arr)
                Cf = float(prod["C"])
                if float(_np.max(eig)) > Cf+TOL_EIG:
                    fail(f"eigvalsh r={r} u={u}: max {float(_np.max(eig))} > C {Cf}")
                max_eig_gap = max(max_eig_gap, Cf-float(_np.max(eig)))
        notes.append(f"moments exact+PSD+eig ok gap_min={max_eig_gap:.3g}")
        # Born enumeration for n<=3 incl. entangled sigma (float diagnostic).
        # n=1: sigma pure target, orth, mixed.
        # n=2: Bell Phi+ entangled; n=3: GHZ.
        PMf = float_pauli()
        def tau_pm_float(rhat, s):
            # rhat floats normalized (here rational->float, unit).
            Rx = [[rhat[0]*PMf["X"][i][j] for j in range(2)] for i in range(2)]
            Ry = [[rhat[1]*PMf["Y"][i][j] for j in range(2)] for i in range(2)]
            Rz = [[rhat[2]*PMf["Z"][i][j] for j in range(2)] for i in range(2)]
            R = [[Rx[i][j]+Ry[i][j]+Rz[i][j] for j in range(2)] for i in range(2)]
            I = PMf["I"]
            return [[(I[i][j]+s*R[i][j])/2.0 for j in range(2)] for i in range(2)]
        # Case n=2 Bell with homogeneous target (3/5,0,4/5) x2, atom u=1/10.
        r2 = (Fraction(3,5), Fraction(0), Fraction(4,5))
        rf2 = hf.validate_rational_target([r2[0], r2[1], r2[2]])
        atom2 = hf.atom_from_u(Fraction(1,10), T_FIX)
        zr2, zi2 = atom2["z_re"], atom2["z_im"]
        dom2 = hf.dominant_info(rf2)["dom_idx"]
        sigma_bell = bell_phiplus()
        # Expected H via enumeration (own h, float).
        # Enumerate axes (3^2=9) with probs, outcomes via Born.
        axes_list = list(itertools.product(("X","Y","Z"), repeat=2))
        exp_H = 0j
        exp_abs2 = 0.0
        for axes in axes_list:
            p_axes = 1.0
            for j, ax in enumerate(axes):
                # prob: longitudinal 3/4 else 1/8.
                is_long = ({"X":0,"Y":1,"Z":2}[ax] == dom2)
                p_axes *= 0.75 if is_long else 0.125
            probs = born_probs_float(sigma_bell, list(axes))
            for bits, p_out in probs.items():
                # H product (own exact -> float for diagnostic).
                H = 1+0j
                for j, (ax, o) in enumerate(zip(axes, bits)):
                    hr, hi = own_h(tuple(rf2), dom2, ax, o, zr2, zi2)
                    h = complex(float(hr), float(hi))
                    H *= h
                exp_H += p_axes*p_out*H
                exp_abs2 += p_axes*p_out*abs(H)**2
        # Dense Tr(sigma A^{tensor2}) and Tr(sigma M^{tensor2}) floats.
        rhat2 = (0.6, 0.0, 0.8)
        tpm = tau_pm_float(rhat2, 1)
        tmm = tau_pm_float(rhat2, -1)
        zc = complex(float(zr2), float(zi2))
        Az1 = [[tpm[i][j]+zc*tmm[i][j] for j in range(2)] for i in range(2)]
        Az2 = tensor_list([Az1, Az1])
        fid_exp = trace(mmul(sigma_bell, Az2))
        if abs(exp_H-fid_exp) > TOL_BORN:
            fail(f"Bell first moment: E={exp_H} vs Tr={fid_exp}")
        # Second moment vs M tensor (float).
        M2f = mat_frac_to_float(hf.single_qubit_moments([r2[0],r2[1],r2[2]], atom2)["M2"])
        M2t = tensor_list([M2f, M2f])
        second_tr = trace(mmul(sigma_bell, M2t)).real
        if abs(exp_abs2-second_tr) > 1e-7:
            fail(f"Bell second moment: E|H|^2={exp_abs2} vs Tr={second_tr}")
        # Bound: second_tr <= C^2 (since M<=C I, tensor <=C^2).
        Cf1 = float(hf.single_qubit_moments([r2[0],r2[1],r2[2]], atom2)["C"])
        if second_tr > Cf1*Cf1+TOL_EIG:
            fail(f"Bell second bound: {second_tr} > C^2={Cf1*Cf1}")
        notes.append(f"Born n=2 Bell E={exp_H.real:.4f}+{exp_H.imag:.4f}i Tr={fid_exp.real:.4f} |H|^2={exp_abs2:.4f} C^2={Cf1*Cf1:.4f}")
        # n=3 GHZ spot check with Pauli target (0,0,1) atom u=0 (z=15/16 real).
        r3 = (Fraction(0), Fraction(0), Fraction(1))
        atom3 = hf.atom_from_u(Fraction(0), T_FIX)
        zr3, zi3 = atom3["z_re"], atom3["z_im"]
        dom3 = 2
        sigma_ghz = ghz3()
        exp_H3 = 0j
        for axes in itertools.product(("X","Y","Z"), repeat=3):
            p_axes = 1.0
            for ax in axes:
                p_axes *= 0.75 if ax == "Z" else 0.125
            probs = born_probs_float(sigma_ghz, list(axes))
            for bits, p_out in probs.items():
                H = 1+0j
                for ax, o in zip(axes, bits):
                    hr, hi = own_h((Fraction(0),Fraction(0),Fraction(1)), dom3, ax, o, zr3, zi3)
                    H *= complex(float(hr), float(hi))
                exp_H3 += p_axes*p_out*H
        # Dense: Az for Pauli Z target with real z=15/16.
        Az_pauli = [[1+0j, 0j], [0j, complex(float(zr3), 0.0)]]
        # In Z basis tau_+=|0><0|, tau_-=|1><1|, Az=diag(1,z). Tensor3 diag.
        Az3 = tensor_list([Az_pauli, Az_pauli, Az_pauli])
        fid3 = trace(mmul(sigma_ghz, Az3))
        if abs(exp_H3-fid3) > 1e-7:
            fail(f"GHZ first moment: {exp_H3} vs {fid3}")
        notes.append(f"Born n=3 GHZ ok E={exp_H3.real:.4f}")
    except Exception as e:
        import traceback as _tb3
        fail(f"moments/Born harness {_tb3.format_exc()}")

    # ---- D. finite LP diagnostic vs exact residual n=1..4 ----
    lp_info: dict = {}
    try:
        us_grid = [Fraction(k, 40) for k in (-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5)]
        # All satisfy |u|<=1/8=T/2 (since 5/40=1/8).
        atoms_grid = [hf.atom_from_u(u, T_FIX) for u in us_grid]
        z_float = [(float(a["z_re"]), float(a["z_im"])) for a in atoms_grid]
        accepted: dict = {}
        for n in (1, 2, 3, 4):
            try:
                sol = hf.solve_finite_lp_float(z_float, n, float(BETA)/2.0)
            except Exception as e:
                fail(f"LP n={n} solver failed (honest report, no weakening): {e}")
                lp_info[f"n{n}"] = {"ok": False, "error": str(e)}
                continue
            rnd = hf.round_coeffs_float(sol["a_float"], sol["b_float"], DEN)
            coeffs = rnd["coeffs"]
            try:
                res = hf.residual_of_coeffs(atoms_grid, coeffs, n)
            except Exception as e:
                fail(f"LP n={n} residual failed: {e}")
                continue
            mass = hf.mass_of_coeffs(coeffs)
            # Nonconstant check: at least two distinct (re,im) or non-all-equal?
            distinct = len(set(coeffs))
            b = res["b"]
            ok = (b <= BETA)
            # Mass reporting (honest, no threshold).
            lp_info[f"n{n}"] = {"ok": bool(ok), "b": str(b), "mass": str(mass),
                                "mass_float": float(mass), "distinct": distinct,
                                "objective": sol["objective"], "viol": sol["max_violation"],
                                "maxdiff": rnd["maxdiff_str"]}
            if not ok:
                fail(f"LP n={n} rounded b={b} > beta={BETA} (honest infeasible/rounding failure, threshold NOT weakened)")
            else:
                # Meaningful nonconstant: distinct>=2 and not all mass on k=0?
                # Check coeffs nonconstant (at least two nonzero or distinct).
                nz = sum(1 for re, im in coeffs if re != 0 or im != 0)
                if distinct < 2 or nz < 2:
                    fail(f"LP n={n} vacuous constant (distinct={distinct} nz={nz})")
                else:
                    accepted[n] = {"b": b, "mass": mass, "coeffs": coeffs, "atoms": atoms_grid,
                                   "us": us_grid, "sol": sol}
                    notes.append(f"LP n={n} accepted b={b} mass={float(mass):.3g} distinct={distinct}")
        if 1 not in accepted:
            fail("LP meaningful accepted missing for n=1 (need nonconstant small-bias)")
        # Weak/constant fake cannot claim small bias.
        # Fake1: single-atom c=(1,0) rest zero (constant-like).
        try:
            fake_coeffs = [(Fraction(1), Fraction(0))] + [(Fraction(0), Fraction(0))]*(len(us_grid)-1)
            fres = hf.residual_of_coeffs(atoms_grid, fake_coeffs, 1)
            if fres["b"] <= BETA:
                fail(f"fake single-atom passed n=1 b={fres['b']} (must fail)")
            else:
                notes.append(f"fake single-atom correctly rejected n=1 b={fres['b']}")
            # Fake2: all zero.
            zero_coeffs = [(Fraction(0), Fraction(0))]*len(us_grid)
            zres = hf.residual_of_coeffs(atoms_grid, zero_coeffs, 1)
            if zres["b"] <= BETA:
                fail(f"fake zero passed b={zres['b']}")
            else:
                notes.append(f"fake zero correctly rejected b={zres['b']}")
            # Fake3: uniform small? e.g., all 1/11?
            uni = [(Fraction(1,11), Fraction(0))]*len(us_grid)
            ures = hf.residual_of_coeffs(atoms_grid, uni, 2)
            # For n=2, uniform likely fails; just check it does not silently pass with wrong threshold.
            # No fail if it passes honestly (unlikely); but verify residual computed exactly.
            notes.append(f"fake uniform n=2 b={ures['b']}")
            if ures["b"] <= BETA:
                # If uniform honestly passes, need to ensure it is nonconstant? It is constant across atoms
                # but still uses multiple atoms; check mass honest. Do not fail, just note.
                notes.append("fake uniform honestly passes (rare, mass honest)")
        except Exception as e:
            import traceback as _tbf
            fail(f"fake harness {_tbf.format_exc()}")
    except Exception as e:
        import traceback as _tb4
        fail(f"LP harness {_tb4.format_exc()}")

    # ---- E. tamper rejection + huge-copy fail before callbacks ----
    try:
        # Build valid n=1 plan from accepted LP (or fallback demo_proposal).
        if 1 in locals().get("accepted", {}) or "accepted" in dir():
            pass
        # Re-derive accepted for tamper base deterministically (n=1).
        us3 = [Fraction(-1,10), Fraction(0), Fraction(1,10)]
        atoms3 = [hf.atom_from_u(u, T_FIX) for u in us3]
        zf3 = [(float(a["z_re"]), float(a["z_im"])) for a in atoms3]
        sol3 = hf.solve_finite_lp_float(zf3, 1, float(BETA)/2.0)
        rnd3 = hf.round_coeffs_float(sol3["a_float"], sol3["b_float"], DEN)
        coeffs3 = rnd3["coeffs"]
        r_base = [Fraction(3,5), Fraction(0), Fraction(4,5)]
        plan_v = hf.plan_homogeneous(r_base, 1, T_FIX, us3, coeffs3, BETA)
        # Counters (SIMULATION-labelled honest backend not used here; MOCK interface probe
        # for tamper uses constant outcomes but labelled MOCK, never physical).
        def _mk_counters():
            calls = {"measure": 0, "rng": 0}
            def _measure(axes):
                calls["measure"] += 1
                return [1 for _ in axes]
            _measure.is_mock = True
            class _CRng:
                def __init__(self, base):
                    self.base = base
                def randrange(self, n):
                    calls["rng"] += 1
                    return self.base.randrange(n)
                def random(self):
                    calls["rng"] += 1
                    return self.base.random()
            return calls, _measure, _CRng(random.Random(1))
        def _expect_reject(plan, label):
            calls, meas, crng = _mk_counters()
            try:
                hf.run_homogeneous_streaming(plan, meas, crng, Fraction(1,2), max_copies=100000)
                fail(f"{label} not rejected")
            except ValueError:
                if calls["measure"] != 0 or calls["rng"] != 0:
                    fail(f"{label} consumed callbacks measure={calls['measure']} rng={calls['rng']}")
            except Exception as e:
                fail(f"{label} wrong exc {type(e).__name__}: {e}")
        # Understated b.
        if not (plan_v.b_cert > 0):
            # b may be small but >0; if exactly 0 (unlikely), skip understated test with 0?
            notes.append(f"tamper base b={plan_v.b_cert} (zero, understated test uses mass tamper)")
        else:
            _expect_reject(dataclasses.replace(plan_v, b_cert=Fraction(0,1)), "understated b")
        # Wrong coeffs (flip first re).
        bad_re = list(plan_v.coeffs_re)
        bad_re[0] = bad_re[0]+Fraction(1,1024)
        _expect_reject(dataclasses.replace(plan_v, coeffs_re=tuple(bad_re)), "wrong coeffs")
        # Wrong us (change first u).
        bad_us = list(plan_v.us)
        bad_us[0] = bad_us[0]+Fraction(1,40) if bad_us[0]+Fraction(1,40) <= T_FIX/2 else bad_us[0]-Fraction(1,40)
        # Need coeffs length match (same len) but atoms changed => residual mismatch => reject.
        # If new u violates bound, plan construction would fail earlier, but replace bypasses;
        # revalidation should reject (either bound or residual mismatch).
        try:
            _expect_reject(dataclasses.replace(plan_v, us=tuple(bad_us)), "wrong us")
        except Exception as e:
            fail(f"wrong us harness {e}")
        # Wrong target.
        _expect_reject(dataclasses.replace(plan_v, r=(Fraction(0),Fraction(0),Fraction(1))), "wrong r")
        # Wrong n.
        try:
            _expect_reject(dataclasses.replace(plan_v, n=2), "wrong n")
        except Exception as e:
            fail(f"wrong n harness {e}")
        # Dict plan rejected.
        try:
            calls, meas, crng = _mk_counters()
            try:
                hf.run_homogeneous_streaming({"n": 1}, meas, crng, Fraction(1,2))
                fail("dict plan not rejected")
            except ValueError:
                if calls["measure"] != 0 or calls["rng"] != 0:
                    fail("dict consumed callbacks")
        except Exception as e:
            fail(f"dict harness {e}")
        # Huge-copy fails before callbacks: eps just above b gives enormous m.
        try:
            tiny_tol = Fraction(1, 10000)
            huge_eps = plan_v.b_cert+tiny_tol
            # plan_copies count-only should return huge N (no fail here).
            counts = hf.plan_copies(plan_v, huge_eps)
            if counts["N"] <= 100000:
                fail(f"huge-copy setup not huge: N={counts['N']}")
            else:
                notes.append(f"huge-copy count-only N={counts['N']} (honest huge)")
                calls, meas, crng = _mk_counters()
                try:
                    hf.run_homogeneous_streaming(plan_v, meas, crng, huge_eps, max_copies=100000)
                    fail("huge-copy not rejected")
                except ValueError as ve:
                    if "max_copies" not in str(ve):
                        fail(f"huge-copy wrong ValueError: {ve}")
                    if calls["measure"] != 0 or calls["rng"] != 0:
                        fail(f"huge-copy consumed callbacks {calls}")
                    else:
                        notes.append("huge-copy correctly failed before callbacks")
        except Exception as e:
            import traceback as _tbh
            fail(f"huge harness {_tbh.format_exc()}")
        # Valid identical copy must run with loose eps and small max (tiny demo counts).
        try:
            p_copy = dataclasses.replace(plan_v)
            # Loose eps=1/2 gives moderate N (check).
            cnt = hf.plan_copies(p_copy, Fraction(1,2))
            if cnt["N"] > 100000:
                fail(f"valid copy N huge {cnt['N']} (loose demo infeasible)")
            else:
                calls, meas, crng = _mk_counters()
                out = hf.run_homogeneous_streaming(p_copy, meas, crng, Fraction(1,2), max_copies=100000)
                if out["N"] != cnt["N"]:
                    fail("valid copy N mismatch")
                if calls["measure"] != cnt["N"]:
                    fail(f"valid copy measure {calls['measure']} vs N {cnt['N']}")
                notes.append(f"valid copy ran N={cnt['N']} est={out['estimate']:.3f} (MOCK probe, not physical)")
        except Exception as e:
            import traceback as _tbv
            fail(f"valid copy rejected: {_tbv.format_exc()}")
        # ---- E2. count-only tamper: plan_copies must canonically revalidate ----
        try:
            def _expect_copies_reject(p, label):
                try:
                    hf.plan_copies(p, Fraction(1,2))
                except ValueError:
                    return
                except Exception as e:
                    fail(f"{label} wrong exc {type(e).__name__}: {e}")
                    return
                fail(f"{label} not rejected by plan_copies")
            # V=0 tamper previously returned zero copies; must now raise.
            _expect_copies_reject(dataclasses.replace(plan_v, V=Fraction(0,1)), "plan_copies V=0")
            # Altered V (doubled) must raise.
            _expect_copies_reject(dataclasses.replace(plan_v, V=plan_v.V*2), "plan_copies V doubled")
            # Understated/altered b_cert must raise (count-only, no callbacks involved).
            if plan_v.b_cert != Fraction(0,1):
                _expect_copies_reject(dataclasses.replace(plan_v, b_cert=Fraction(0,1)), "plan_copies b=0")
            else:
                _expect_copies_reject(dataclasses.replace(plan_v, b_cert=plan_v.b_cert+Fraction(1,100)), "plan_copies b altered")
            # ceil_exp / C_single derived bindings must be revalidated.
            _expect_copies_reject(dataclasses.replace(plan_v, ceil_exp=int(plan_v.ceil_exp)+1), "plan_copies ceil_exp")
            _expect_copies_reject(dataclasses.replace(plan_v, C_single=plan_v.C_single+Fraction(1,100)), "plan_copies C_single")
            # den is positive-int metadata per constructor; altered den must raise
            # (no invented coefficient-denominator requirement beyond constructor).
            _expect_copies_reject(dataclasses.replace(plan_v, den=0), "plan_copies den=0")
            _expect_copies_reject(dataclasses.replace(plan_v, den=True), "plan_copies den bool")
            # Valid plan still gives identical counts via canonical path.
            try:
                c1 = hf.plan_copies(plan_v, Fraction(1,2))
                c2 = hf.plan_copies(dataclasses.replace(plan_v), Fraction(1,2))
                if c1["N"] != c2["N"] or c1["V"] != c2["V"] or c1["b"] != c2["b"]:
                    fail("plan_copies valid mismatch")
                else:
                    notes.append(f"count-only tamper rejected (V/b/ceil/C/den), valid N={c1['N']}")
            except Exception as e:
                fail(f"plan_copies valid raised: {e}")
        except Exception as e:
            import traceback as _tbe2
            fail(f"count-only tamper harness {_tbe2.format_exc()}")
        # ---- E3. stream float/bool outcome rejection (1.0==1 passes `in`) ----
        try:
            def _float_measure(axes):
                return [1.0 for _ in axes]
            _float_measure.is_mock = True
            def _bool_measure(axes):
                return [True for _ in axes]
            _bool_measure.is_mock = True
            for _m, _lbl in ((_float_measure, "float 1.0"), (_bool_measure, "bool True")):
                _calls, _mm, _cr = _mk_counters()
                # Use valid plan with loose eps; stream must raise ValueError on bad outcomes.
                try:
                    hf.run_homogeneous_streaming(plan_v, _m, _cr, Fraction(1,2), max_copies=100000)
                    fail(f"stream {_lbl} not rejected")
                except ValueError as ve:
                    if "+-1 ints" not in str(ve) and "bool/float" not in str(ve):
                        fail(f"stream {_lbl} wrong ValueError: {ve}")
                    else:
                        notes.append(f"stream {_lbl} correctly rejected")
                except Exception as e:
                    fail(f"stream {_lbl} wrong exc {type(e).__name__}: {e}")
            # Direct per-copy scorer must also reject floats.
            try:
                hf._per_copy_real_score(tuple(plan_v.r), plan_v.dom_idx,
                                        [(a["z_re"], a["z_im"]) for a in atoms3],
                                        list(zip(plan_v.coeffs_re, plan_v.coeffs_im)),
                                        ["Z"], [1.0])
                fail("per-copy float not rejected")
            except ValueError:
                notes.append("per-copy float correctly rejected")
        except Exception as e:
            import traceback as _tbe3
            fail(f"outcome-type harness {_tbe3.format_exc()}")
        # ---- E4. MOCK axes-copy plumbing: callback mutation must not corrupt scoring ----
        try:
            def _clean(axes):
                return [1 for _ in axes]
            _clean.is_mock = True
            def _dirty(axes):
                # Mutate the passed list in place (plumbing probe only).
                for i in range(len(axes)):
                    axes[i] = "X"
                return [1 for _ in axes]
            _dirty.is_mock = True
            _r1, _r2 = random.Random(999), random.Random(999)
            _o_clean = hf.run_homogeneous_streaming(plan_v, _clean, _r1, Fraction(1,2), max_copies=100000)
            _o_dirty = hf.run_homogeneous_streaming(plan_v, _dirty, _r2, Fraction(1,2), max_copies=100000)
            if _o_clean["median_exact"] != _o_dirty["median_exact"]:
                fail(f"axes-mutation changed scoring clean={_o_clean['median_exact']} dirty={_o_dirty['median_exact']}")
            else:
                notes.append(f"axes-copy plumbing preserved (median={_o_clean['median_exact']}, MOCK probe not physical)")
        except Exception as e:
            import traceback as _tbe4
            fail(f"axes-copy harness {_tbe4.format_exc()}")
        # ---- E5. out-of-range median intervals: clip01 both ends + outward floats ----
        try:
            _T0 = Fraction(1,4)
            _u0 = _T0/2
            _at0 = hf.atom_from_u(_u0, _T0)
            _zr0, _zi0 = _at0["z_re"], _at0["z_im"]
            if _zi0 == 0:
                fail("E5 z_im zero (unreachable)")
            else:
                _c0 = [(Fraction(1,1), _zr0/_zi0)]
                _r345 = [Fraction(3,5), Fraction(0), Fraction(4,5)]
                _res0 = hf.residual_of_coeffs([_at0], _c0, 1)
                if _res0["b"] != Fraction(0,1):
                    fail(f"E5 residual {_res0['b']} != 0")
                else:
                    _plan0 = hf.plan_homogeneous(_r345, 1, _T0, [_u0], _c0, BETA)
                    _eps0 = Fraction(9,10)
                    _cnt0 = hf.plan_copies(_plan0, _eps0)
                    if _cnt0["N"] > 100000:
                        fail(f"E5 N huge {_cnt0['N']}")
                    else:
                        class _AlwaysX:
                            def randrange(self, n):
                                if n != 8:
                                    raise ValueError("randrange n")
                                return 0
                        def _plus(axes):
                            return [1 for _ in axes]
                        _plus.is_mock = True
                        def _minus(axes):
                            return [-1 for _ in axes]
                        _minus.is_mock = True
                        _outP = hf.run_homogeneous_streaming(_plan0, _plus, _AlwaysX(), _eps0, max_copies=100000)
                        _outM = hf.run_homogeneous_streaming(_plan0, _minus, _AlwaysX(), _eps0, max_copies=100000)
                        if _outP["median_exact"] != str(Fraction(17,5)):
                            fail(f"E5 high median {_outP['median_exact']} != 17/5")
                        if _outM["median_exact"] != str(Fraction(-7,5)):
                            fail(f"E5 low median {_outM['median_exact']} != -7/5")
                        if _outP["interval_exact"] != [str(Fraction(1,1)), str(Fraction(1,1))]:
                            fail(f"E5 high interval {_outP['interval_exact']} != [1,1]")
                        if _outM["interval_exact"] != [str(Fraction(0,1)), str(Fraction(0,1))]:
                            fail(f"E5 low interval {_outM['interval_exact']} != [0,0]")
                        # Outward containment for both (MOCK plumbing only, not physical).
                        for _o, _lbl in ((_outP, "high"), (_outM, "low")):
                            _loE, _hiE = Fraction(_o["interval_exact"][0]), Fraction(_o["interval_exact"][1])
                            if _loE > _hiE:
                                fail(f"E5 {_lbl} inverted exact {_o['interval_exact']}")
                            _loF, _hiF = _o["interval"][0], _o["interval"][1]
                            if not (_loF <= _hiF):
                                fail(f"E5 {_lbl} inverted floats {_o['interval']}")
                            if not (Fraction(_loF) <= _loE and _hiE <= Fraction(_hiF)):
                                fail(f"E5 {_lbl} outward failed floats={_o['interval']} exact={_o['interval_exact']}")
                        # Also check valid-copy outward containment.
                        def _cleanV(axes):
                            return [1 for _ in axes]
                        _cleanV.is_mock = True
                        _oV = hf.run_homogeneous_streaming(plan_v, _cleanV, random.Random(1), Fraction(1,2), max_copies=100000)
                        _vloE, _vhiE = Fraction(_oV["interval_exact"][0]), Fraction(_oV["interval_exact"][1])
                        _vloF, _vhiF = _oV["interval"][0], _oV["interval"][1]
                        if not (Fraction(_vloF) <= _vloE and _vhiE <= Fraction(_vhiF)):
                            fail(f"E5 valid outward failed { _oV['interval']} vs {_oV['interval_exact']}")
                        else:
                            notes.append("E5 high/low clip01+outward ok (medians 17/5,-7/5; MOCK plumbing only)")
        except Exception as e:
            import traceback as _tbe5
            fail(f"E5 harness {_tbe5.format_exc()}")
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
        r_ok = [Fraction(3,5), Fraction(0), Fraction(4,5)]
        us_ok = [Fraction(0)]
        # Need valid coeffs for n=1 with single atom? Single atom cannot achieve beta (residual large),
        # so plan_homogeneous will raise infeasible (ValueError) — use that to test k-mismatch? Instead
        # test with valid 3-atom plan base from Sec E (re-derive quickly).
        us3b = [Fraction(-1,10), Fraction(0), Fraction(1,10)]
        atoms3b = [hf.atom_from_u(u, T_FIX) for u in us3b]
        # Use previously rounded coeffs3 if available else solve again.
        try:
            _sol = hf.solve_finite_lp_float([(float(a["z_re"]),float(a["z_im"])) for a in atoms3b], 1, float(BETA)/2.0)
            _rnd = hf.round_coeffs_float(_sol["a_float"], _sol["b_float"], DEN)
            _coeffs = _rnd["coeffs"]
        except Exception:
            _coeffs = [(Fraction(1,2),Fraction(0)),(Fraction(0),Fraction(0)),(Fraction(1,2),Fraction(0))]
        expect_value_error(lambda: hf.plan_homogeneous([0.0,0.0,1.0], 1, T_FIX, us3b, _coeffs, BETA), "float r")
        expect_value_error(lambda: hf.plan_homogeneous([Fraction(1,2),Fraction(1,2),Fraction(1,2)], 1, T_FIX, us3b, _coeffs, BETA), "nonunit r")
        expect_value_error(lambda: hf.plan_homogeneous(r_ok, 0, T_FIX, us3b, _coeffs, BETA), "n0")
        expect_value_error(lambda: hf.plan_homogeneous(r_ok, True, T_FIX, us3b, _coeffs, BETA), "bool n")
        expect_value_error(lambda: hf.plan_homogeneous(r_ok, 1, Fraction(1,2), us3b, _coeffs, BETA), "T big")
        expect_value_error(lambda: hf.plan_homogeneous(r_ok, 1, 0.25, us3b, _coeffs, BETA), "float T")
        expect_value_error(lambda: hf.plan_homogeneous(r_ok, 1, T_FIX, [Fraction(1,2)], _coeffs, BETA), "us len mismatch")
        expect_value_error(lambda: hf.plan_homogeneous(r_ok, 1, T_FIX, us3b, [(Fraction(1),Fraction(0))], BETA), "coeff len")
        expect_value_error(lambda: hf.plan_homogeneous(r_ok, 1, T_FIX, us3b, [(0.5,0.0)]*3, BETA), "float coeff")
        # eps<=b must raise: use fresh valid plan's exact b_cert (BETA itself is budget, not
        # necessarily <=b; BETA>=b so eps=BETA is VALID and must NOT raise).
        try:
            _planF = hf.plan_homogeneous(r_ok, 1, T_FIX, us3b, _coeffs, BETA)
        except Exception as e:
            fail(f"SecF plan build failed: {e}")
            _planF = None
        if _planF is not None:
            expect_value_error(lambda: hf.plan_copies(_planF, _planF.b_cert), "eps==b")
            expect_value_error(lambda: hf.plan_copies(_planF, _planF.b_cert/2), "eps<b")
            # Sanity: eps=BETA is valid iff BETA>b (strict); if b==BETA exactly, eps=BETA raises
            # honestly (tol=0). Check consistency without failing.
            if _planF.b_cert < BETA:
                try:
                    _cnt = hf.plan_copies(_planF, BETA)
                    notes.append(f"SecF eps=BETA valid (b={_planF.b_cert}<BETA) N={_cnt['N']}")
                except ValueError as e:
                    fail(f"eps=BETA should be valid when b<BETA: {e}")
            else:
                # b==BETA exactly: eps=BETA must raise (tol=0).
                expect_value_error(lambda: hf.plan_copies(_planF, BETA), "eps==b==BETA")
            expect_value_error(lambda: hf.plan_copies(_planF, 0.5), "float eps")
        else:
            notes.append("SecF skipped eps tests (no valid plan; LP infeasible honestly)")
        expect_value_error(lambda: hf.variance_bound(Fraction(-1), Fraction(1), T_FIX), "neg mass")
    except Exception as e:
        import traceback as _tb6
        fail(f"invalid harness {_tb6.format_exc()}")

    # ---- G. tiny honest sampling demo (SIMULATION-labelled, loose eps) ----
    tiny_info: dict = {}
    try:
        # n=1 homogeneous (3/5,0,4/5), 3-atom plan, loose eps=1/2.
        r_tiny = [Fraction(3,5), Fraction(0), Fraction(4,5)]
        us_tiny = [Fraction(-1,10), Fraction(0), Fraction(1,10)]
        atoms_tiny = [hf.atom_from_u(u, T_FIX) for u in us_tiny]
        zf_tiny = [(float(a["z_re"]), float(a["z_im"])) for a in atoms_tiny]
        sol_tiny = hf.solve_finite_lp_float(zf_tiny, 1, float(BETA)/2.0)
        rnd_tiny = hf.round_coeffs_float(sol_tiny["a_float"], sol_tiny["b_float"], DEN)
        plan_tiny = hf.plan_homogeneous(r_tiny, 1, T_FIX, us_tiny, rnd_tiny["coeffs"], BETA)
        counts_tiny = hf.plan_copies(plan_tiny, Fraction(1,2))
        # Honest single-qubit sigma: pure target itself (fidelity 1) with explicit Born backend.
        # sigma density matrix floats for target (3/5,0,4/5): |psi><psi| with Bloch r.
        # |psi> = cos(t/2)|0>+sin(t/2)|1| with r=(sin t,0,cos t), t=arccos(0.8).
        import math as _m
        theta = _m.acos(0.8)
        c, s = _m.cos(theta/2), _m.sin(theta/2)
        psi = [c+0j, s+0j]
        sigma_pure = [[psi[i]*psi[j].conjugate() for j in range(2)] for i in range(2)]
        backend_rng = random.Random(777)
        design_rng = random.Random(1)
        measure_sim = make_honest_born_measure(sigma_pure, 1, backend_rng)
        # Exact-law first moment (no sampling): enumerate axes/outcomes.
        # Compute expected real score via Born (float diagnostic) vs fidelity 1 within b+tol.
        # Use own h for expectation (independent) plus production h equality already checked.
        exp_sim = 0.0
        for ax in ("X","Y","Z"):
            p_ax = 0.75 if ax == "Z" else 0.125  # dom Z
            probs = born_probs_float(sigma_pure, [ax])
            for bits, p_out in probs.items():
                o = bits[0]
                # Production per-copy score for n=1: Re sum c H (H=h).
                sc_frac = hf._per_copy_real_score(tuple(plan_tiny.r), plan_tiny.dom_idx,
                                                 [(a["z_re"],a["z_im"]) for a in atoms_tiny],
                                                 list(zip(plan_tiny.coeffs_re, plan_tiny.coeffs_im)),
                                                 [ax], [o])
                exp_sim += p_ax*p_out*float(sc_frac)
        fid_pure = 1.0
        b_f = float(plan_tiny.b_cert)
        if abs(exp_sim-fid_pure) > b_f+TOL_BORN:
            fail(f"tiny exact-law: E={exp_sim} fid={fid_pure} b={b_f}")
        else:
            notes.append(f"tiny exact-law E={exp_sim:.4f} fid=1 b={b_f:.4g}")
        # Sampling run with honest backend (SIMULATION, loose eps). N may be ~1e4; run it
        # only if N<=20000 to keep runtime bounded; else report count-only (honest huge).
        if counts_tiny["N"] > 20000:
            tiny_info = {"N": counts_tiny["N"], "ran": False,
                         "note": "N>20000, count-only (honest huge, no sampling)"}
            notes.append(f"tiny N={counts_tiny['N']} too large for sampling, count-only")
        else:
            out_tiny = hf.run_homogeneous_streaming(plan_tiny, measure_sim, design_rng,
                                                    Fraction(1,2), max_copies=100000)
            # Check estimate in [0,1] and interval covers fidelity 1? With loose eps=0.5,
            # interval radius = tol+b =0.5, so [est-0.5,est+0.5] clipped should contain 1
            # with >=1-5/32 (probabilistic; single run may rarely fail — do not fail hard
            # on coverage, only on range/counts/composition). Check range only.
            if not (0.0 <= out_tiny["estimate"] <= 1.0):
                fail("tiny estimate range")
            if out_tiny["N"] != counts_tiny["N"]:
                fail("tiny N mismatch")
            tiny_info = {"N": out_tiny["N"], "ran": True, "est": out_tiny["estimate"],
                         "interval": out_tiny["interval"], "expected": exp_sim}
            notes.append(f"tiny sampling est={out_tiny['estimate']:.3f} interval={out_tiny['interval']} (SIMULATION backend, loose eps)")
    except Exception as e:
        import traceback as _tb7
        fail(f"tiny harness {_tb7.format_exc()}")

    # ---- write JSON ----
    t_end = time.perf_counter()
    curr, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # Serialize lp_info with Fractions as strings (already).
    # Recompute accepted summary for JSON (n=1..4 honest).
    result = {
        "metadata": {
            "python": py_ver, "numpy": numpy_ver, "scipy": scipy_ver,
            "seed": SEED, "T": str(T_FIX), "beta": str(BETA), "den": DEN,
            "time_sec": t_end-t_start, "peak_bytes": peak,
            "status": "pass" if not failures else "fail",
            "failures": failures, "notes": notes,
        },
        "lp_grid": lp_info,
        "tiny_demo": tiny_info,
        "thresholds": {"T": str(T_FIX), "beta": str(BETA), "den": DEN,
                       "tol_born": TOL_BORN, "tol_eig": TOL_EIG},
        "conventions": {
            "target": "exact-rational UNIT int/Fraction only (floats rejected)",
            "atoms": "rational u,T with |u|<=T/2 sufficient angle bound, z derived",
            "residual": "max_k|Re sum c z^k-delta| exact Fractions",
            "mass": "sum(|re|+|im|) exact, V=mass^2 3^{ceil(9 lam T^2)}",
            "blocks": "3 blocks, m=ceil(4V/tol^2), median then clip, delta=1/3 (5/32<1/3)",
            "callbacks": "full-copy only, axes via randrange(8) 1/8,1/8,3/4",
            "float_LP": "diagnostic proposal only, fresh exact residual is certificate",
            "mock_label": "tamper probes use MOCK constant outcomes (not physical); honest demo uses SIMULATION Born backend",
        },
    }
    out_path = (Path(__file__).resolve().parents[1] / "results" / "homogeneous_fidelity_checks.json")
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(f"wrote {out_path} status={result['metadata']['status']} "
              f"failures={len(failures)} time={t_end-t_start:.2f}s")
    except Exception as e:
        fail(f"json write failed: {e}")
    if failures:
        print(f"{len(failures)} FAILURES")
        return 1
    print("all homogeneous-fidelity checks passed (bounded prototype, NOT acceptance)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
