"""Deterministic stdlib-only checks for the broader product-state boundary.

Generalizes verify_population_recovery.py (method reused, file NOT modified):
any fixed pure single-qubit Bloch vector r, R=r.X X+r.Y Y+r.Z Z, L=||r||_1,
a=||r||_inf, tau_pm=(I+-R)/2, homogeneous target tau_+^{tensor n} and
heterogeneous tensor_j tau^j_+ targets.

What is checked (n<=3, explicit complex matrices/tensor products):
  - Single-qubit weighted sign-corrected POVM E_s=(I+sR/L)/2 for diverse
    signed/zero/asymmetric r (explicit weighted sum vs closed form; positivity,
    normalization, [E_s,R]=0; zero coefficients never selected).
  - Homogeneous averaged-vs-BSC for arbitrary sigma (weighted axis average over
    all 3^n settings via explicit (I+oA)/2 projectors vs BSC_eta prediction
    from R-diagonal p_x=Tr[sigma E_x] with eta=(1-1/L)/2); honesty, fidelity
    match p_0, off-diagonal vanishing.
  - Pauli-eigenstate all-matching indicator equals fidelity (explicit).
  - Heterogeneous degradation to common 1/sqrt(3) plus exact stabilizer flag:
    explicit weighted Born + classical degradation convolution vs common-BSC
    unnormalized joint law; flag-multiplied score moments (bias 0 instance,
    second moment <= (3/2)^k) without postselection/division.
  - Lower simulation: per-axis Born means (1+(-1)^x r_i)/2 on diagonal mixtures
    plus classical base-BSC_a + per-request coin (mean r_i/a, incl. negative and
    zero) reproduction; a CONCRETE within-copy adaptive policy enumerated path
    by path (explicit projectors vs scalar-coin simulation).
  - Falsification: different axes do NOT give same BSC for asymmetric r
    (diagonal-setting spread large), while 111 spread ~0.
  - Noise-level distinction: upper eta=(1-1/L)/2 vs lower eta_*=(1-a)/2 differ
    generally, coincide for 111.
  - Overlap/operator identities: Tr(PQ)=prod(1+a_j)/2, Tr[(P-Q)^2]=2-2c,
    cubic (P-Q)^3=(1-c)(P-Q) explicitly; analytic bound sqrt(1-c)<=
    sqrt(sum(1-a_j)/2); all-n near-axis certificate via 1-cos(t)<=t^2/2
    (theta_n=c0/sqrt(n) gives ||P-Q||<=c0/2 for every n) vs fixed nonzero
    tilt (L>1,a<1 gaps).
  - Product-correlation boundary: local-score E[z_j]=f_j on product mixed
    inputs via explicit Born/matrix plus partial-trace marginals; correlated
    separable mixture 0.5++/0.5-- has fid 1/2 vs marginal product 1/4;
    target-product does not imply input-product; polynomial sufficient
    N=ceil(3n^2/(2eps^2) ln(2n/delta)) recorded for n=2.

These checks support reduction algebra only. They do not prove any asymptotic
sample-complexity theorem and are not novelty evidence.

Run: python code/verify_product_state_boundary.py
Writes product_state_boundary_checks.json alongside this script.
Stdlib only. No NumPy. No network. No upstream execution.
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
SQRT2 = math.sqrt(2.0)
SQRT3 = math.sqrt(3.0)
EPS0 = 1.0 / 20.0
DELTA0 = 1.0 / 3.0

I_M = [[1 + 0j, 0j], [0j, 1 + 0j]]
X_M = [[0j, 1 + 0j], [1 + 0j, 0j]]
Y_M = [[0j, -1j], [1j, 0j]]
Z_M = [[1 + 0j, 0j], [0j, -1 + 0j]]
PAULI = {"X": X_M, "Y": Y_M, "Z": Z_M}
PAULI_LIST = ("X", "Y", "Z")


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


def msub(a, b):
    return madd(a, b, 1.0, -1.0)


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


def max_abs_entry(a):
    return max(abs(v) for row in a for v in row)


def r_mat(r):
    rx, ry, rz = r
    return [
        [rx * X_M[i][j] + ry * Y_M[i][j] + rz * Z_M[i][j] for j in range(2)]
        for i in range(2)
    ]


def L_of(r):
    return abs(r[0]) + abs(r[1]) + abs(r[2])


def a_of(r):
    return max(abs(r[0]), abs(r[1]), abs(r[2]))


def tau_pm(r, sign):
    R = r_mat(r)
    return [[(I_M[i][j] + sign * R[i][j]) / 2.0 for j in range(2)] for i in range(2)]


def pi_plus(r):
    """Nearest Pauli eigenstate projector (max |coord|, matching sign)."""
    coords = [r[0], r[1], r[2]]
    idx = max(range(3), key=lambda k: abs(coords[k]))
    ax = PAULI_LIST[idx]
    s = 1.0 if coords[idx] >= 0 else -1.0
    A = PAULI[ax]
    return [[(I_M[i][j] + s * A[i][j]) / 2.0 for j in range(2)] for i in range(2)], ax, s


def plus_eigvec(r):
    rx, ry, rz = r
    if abs(rz - 1.0) < 1e-12 and abs(rx) + abs(ry) < 1e-12:
        return [1 + 0j, 0j]
    vx = -(rx - 1j * ry)
    vz = rz - 1.0
    n = math.sqrt(abs(vx) ** 2 + abs(vz) ** 2)
    return [vx / n + 0j, vz / n + 0j]


def minus_eigvec(r):
    rx, ry, rz = r
    if abs(rz + 1.0) < 1e-12 and abs(rx) + abs(ry) < 1e-12:
        return [1 + 0j, 0j]
    vx = -(rx - 1j * ry)
    vz = rz + 1.0
    n = math.sqrt(abs(vx) ** 2 + abs(vz) ** 2)
    return [vx / n + 0j, vz / n + 0j]


def eig2_hermitian(a, b, d):
    """Eigenvalues of [[a,b],[conj(b),d]] with a,d real-ish Hermitian."""
    tr = a.real + d.real
    diff = (a.real - d.real) / 2.0
    disc = math.sqrt(diff * diff + abs(b) ** 2)
    return tr / 2.0 + disc, tr / 2.0 - disc


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


def proj_tensor(setting, outcomes):
    mats = []
    for ax, s in zip(setting, outcomes):
        a = PAULI[ax]
        mats.append([[(I_M[i][j] + s * a[i][j]) / 2.0 for j in range(2)] for i in range(2)])
    return tensor_list(mats)


def r_projector_tensor(xbits, rs):
    return tensor_list([tau_pm(r, 1 if b == 0 else -1) for b, r in zip(xbits, rs)])


def target_projector(rs):
    return tensor_list([tau_pm(r, 1) for r in rs])


def sampling_weights(rs):
    """Per-qubit axis sampling probs |r_i|/L (zero -> 0, never selected)."""
    out = []
    for r in rs:
        L = L_of(r)
        out.append({"X": abs(r[0]) / L, "Y": abs(r[1]) / L, "Z": abs(r[2]) / L})
    return out


def sign_of_coord(r, ax):
    idx = {"X": 0, "Y": 1, "Z": 2}[ax]
    v = r[idx]
    if v > 0:
        return 1
    if v < 0:
        return -1
    return 0


def bsc_w(s, x, eta):
    same = (s == (1 if x == 0 else -1))
    return (1.0 - eta) if same else eta


def born_dist(sigma, setting):
    n = len(setting)
    out = {}
    for s in all_outcomes(n):
        out[s] = trace(mmul(sigma, proj_tensor(setting, s)))
    return out


def make_states(n, rs):
    """Deterministic test family; target/orth/R-cat use per-qubit R eigenvectors."""
    dim = 2 ** n
    states = {}
    vec0 = [0j] * dim
    vec0[0] = 1 + 0j
    states["comp_zero"] = ket_to_dm(vec0)
    vec1 = [0j] * dim
    vec1[dim - 1] = 1 + 0j
    states["comp_one"] = ket_to_dm(vec1)

    def kron_vec(a, b):
        return [x * y for x in a for y in b]

    vv = [1 + 0j]
    for _ in range(n):
        vv = kron_vec(vv, [1 / math.sqrt(2) + 0j, 1 / math.sqrt(2) + 0j])
    states["plus_product"] = ket_to_dm(vv)
    # Target / orthogonal densities directly from projectors (pure products).
    states["target_pure"] = target_projector(rs)
    states["orthogonal_pure"] = tensor_list([tau_pm(r, -1) for r in rs])
    vg = [0j] * dim
    vg[0] = 1 / math.sqrt(2)
    vg[dim - 1] = 1 / math.sqrt(2)
    states["ghz"] = ket_to_dm(vg)
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
    states["maximally_mixed"] = mscale(eye(dim), 1.0 / dim)
    states["target_orth_mix"] = madd(states["target_pure"], states["orthogonal_pure"], 0.6, 0.4)
    states["ghz_mixed"] = madd(states["ghz"], states["maximally_mixed"], 0.5, 0.5)
    # Coherent R-cat: tensor _plus eigenvectors superposed with tensor _minus.
    vp = [1 + 0j]
    vo = [1 + 0j]
    for r in rs:
        vp = kron_vec(vp, plus_eigvec(r))
        vo = kron_vec(vo, minus_eigvec(r))
    vs = [(vp[i] + vo[i]) / math.sqrt(2) for i in range(dim)]
    nrm = math.sqrt(sum(abs(c) ** 2 for c in vs))
    vs = [c / nrm for c in vs]
    states["r_cat"] = ket_to_dm(vs)
    return states


# ---------------- checks ----------------

def check_single_qubit_povm(rs_list):
    res = {"max_povm_err": 0.0, "max_norm_err": 0.0, "max_pos_viol": 0.0,
           "max_comm_err": 0.0, "details": []}
    for r in rs_list:
        L = L_of(r)
        R = r_mat(r)
        p = {"X": abs(r[0]) / L, "Y": abs(r[1]) / L, "Z": abs(r[2]) / L}
        for s in (1, -1):
            Ew = zeros(2)
            for ax in PAULI_LIST:
                if p[ax] == 0:
                    continue
                sg = sign_of_coord(r, ax)
                A = PAULI[ax]
                term = [[(I_M[i][j] + s * sg * A[i][j]) / 2.0 for j in range(2)] for i in range(2)]
                Ew = madd(Ew, term, 1.0, p[ax])
            # (I + sR/L)/2 explicit:
            Eform = [[(I_M[i][j] + s * R[i][j] / L) / 2.0 for j in range(2)] for i in range(2)]
            res["max_povm_err"] = max(res["max_povm_err"], max_abs_entry(msub(Ew, Eform)))
        # normalization E_+ + E_- = I
        Ep = [[(I_M[i][j] + R[i][j] / L) / 2.0 for j in range(2)] for i in range(2)]
        Em = [[(I_M[i][j] - R[i][j] / L) / 2.0 for j in range(2)] for i in range(2)]
        res["max_norm_err"] = max(res["max_norm_err"], max_abs_entry(msub(madd(Ep, Em), I_M)))
        for E in (Ep, Em):
            ev1, ev2 = eig2_hermitian(E[0][0], E[0][1], E[1][1])
            for ev in (ev1, ev2):
                if ev < -res["max_pos_viol"]:
                    res["max_pos_viol"] = -ev
                if ev > 1 + res["max_pos_viol"] and ev - 1 > res["max_pos_viol"]:
                    res["max_pos_viol"] = max(res["max_pos_viol"], ev - 1.0)
            # commutator [E,R]
            C = msub(mmul(E, R), mmul(R, E))
            res["max_comm_err"] = max(res["max_comm_err"], max_abs_entry(C))
        # zero-coord never selected
        for ax, idx in (("X", 0), ("Y", 1), ("Z", 2)):
            if r[idx] == 0:
                assert p[ax] == 0
            else:
                assert p[ax] > 0
        # legality |r_i/a| <= 1
        a = a_of(r)
        for v in r:
            assert abs(v) / a <= 1.0 + 1e-12
        res["details"].append({"r": list(r), "L": L, "a": a,
                               "eta_up": (1 - 1 / L) / 2, "eta_low": (1 - a) / 2})
    return res


def check_homogeneous(n, r):
    rs = [r] * n
    L = L_of(r)
    eta = (1 - 1 / L) / 2.0
    settings = all_settings(n)
    outcomes = all_outcomes(n)
    xs = all_x(n)
    states = make_states(n, rs)
    wtab = sampling_weights(rs)
    # setting weight
    def setting_weight(st):
        w = 1.0
        for j, ax in enumerate(st):
            w *= wtab[j][ax]
        return w

    ex_list = [r_projector_tensor(x, rs) for x in xs]
    target = target_projector(rs)
    res = {"n": n, "r": list(r), "L": L, "a": a_of(r), "eta": eta,
           "max_avg_vs_bsc": 0.0, "max_p_sum_err": 0.0, "max_p_neg": 0.0,
           "max_fid_err": 0.0, "max_offdiag": 0.0, "max_norm_err": 0.0,
           "max_imag": 0.0}
    for name, sigma in states.items():
        p_c = [trace(mmul(sigma, e)) for e in ex_list]
        res["max_imag"] = max(res["max_imag"], max(abs(v.imag) for v in p_c))
        raw = [v.real for v in p_c]
        res["max_p_sum_err"] = max(res["max_p_sum_err"], abs(sum(raw) - 1.0))
        res["max_p_neg"] = max(res["max_p_neg"], max(0.0, max(-v for v in raw)))
        fid = trace(mmul(sigma, target)).real
        res["max_fid_err"] = max(res["max_fid_err"],
                                 abs(raw[xs.index(tuple([0] * n))] - fid))
        # weighted averaged Q(s) via explicit projectors + sign correction
        q_avg = {s: 0j for s in outcomes}
        for st in settings:
            w = setting_weight(st)
            if w == 0:
                continue
            bd = born_dist(sigma, st)
            signs = [sign_of_coord(rs[j], st[j]) for j in range(n)]
            for o, prob in bd.items():
                s = tuple(signs[j] * o[j] for j in range(n))
                q_avg[s] += prob * w
            res["max_norm_err"] = max(res["max_norm_err"],
                                      abs(sum(v.real for v in bd.values()) - 1.0))
            res["max_imag"] = max(res["max_imag"],
                                  max(abs(v.imag) for v in bd.values()))
        # BSC prediction from raw p
        md = 0.0
        for s in outcomes:
            pred = 0.0
            for rv, x in zip(raw, xs):
                bw = 1.0
                for sj, xj in zip(s, x):
                    bw *= bsc_w(sj, xj, eta)
                pred += rv * bw
            md = max(md, abs(q_avg[s].real - pred))
            res["max_imag"] = max(res["max_imag"], abs(q_avg[s].imag))
        res["max_avg_vs_bsc"] = max(res["max_avg_vs_bsc"], md)
        # offdiag vanish
        sigma_diag = zeros(2 ** n)
        for rv, e in zip(raw, ex_list):
            sigma_diag = madd(sigma_diag, e, 1.0, rv)
        qd = {s: 0j for s in outcomes}
        for st in settings:
            w = setting_weight(st)
            if w == 0:
                continue
            bd = born_dist(sigma_diag, st)
            signs = [sign_of_coord(rs[j], st[j]) for j in range(n)]
            for o, prob in bd.items():
                s = tuple(signs[j] * o[j] for j in range(n))
                qd[s] += prob * w
        res["max_offdiag"] = max(res["max_offdiag"],
                                 max(abs(q_avg[s].real - qd[s].real) for s in outcomes))
    # diagonal identical-across-settings spread is NOT expected for asymmetric r;
    # recorded separately in falsification check. Here just record honesty done.
    return res


def check_pauli_indicator(n):
    # r = +Z eigenstate
    rs = [(0.0, 0.0, 1.0)] * n
    states = make_states(n, rs)
    xs = all_x(n)
    ex_list = [r_projector_tensor(x, rs) for x in xs]
    target = target_projector(rs)
    st_allz = tuple(["Z"] * n)
    want = tuple([1] * n)
    max_err = 0.0
    for sigma in states.values():
        bd = born_dist(sigma, st_allz)
        pall = bd[want].real
        p0 = trace(mmul(sigma, target)).real
        max_err = max(max_err, abs(pall - p0))
    # Hoeffding N for eps0/delta0 on [0,1] indicator (n-independent)
    N_star = math.ceil(math.log(2.0 / DELTA0) / (2 * EPS0 * EPS0))
    return {"n": n, "max_indicator_err": max_err, "N_star": N_star}


def check_heterogeneous():
    rA = (1 / SQRT3, 1 / SQRT3, 1 / SQRT3)
    rB = (1 / SQRT2, 1 / SQRT2, 0.0)
    rC = (0.0, 0.0, 1.0)
    rs = [rA, rB, rC]
    n = 3
    K_idx = [0, 1]
    S_idx = [2]
    k = 2
    eta_common = (1 - 1 / SQRT3) / 2.0
    settings = all_settings(n)
    outcomes = all_outcomes(n)
    xs = all_x(n)
    states = make_states(n, rs)
    wtab = sampling_weights(rs)
    ex_list = [r_projector_tensor(x, rs) for x in xs]

    def setting_weight(st):
        w = 1.0
        for j, ax in enumerate(st):
            w *= wtab[j][ax]
        return w

    # degradation means m_j = L_j/sqrt3 for j in K
    Lj = [L_of(rs[j]) for j in K_idx]
    mj = [L / SQRT3 for L in Lj]
    assert all(m <= 1.0 + 1e-12 for m in mj)

    def g_score(tup):
        v = 1.0
        for t in tup:
            v *= (1.0 + SQRT3 * t) / 2.0
        return v

    res = {"rs": [list(r) for r in rs], "eta_common": eta_common,
           "max_deg_err": 0.0, "max_flag_bias": 0.0, "max_second": 0.0,
           "second_bound": (1.5) ** k, "max_imag": 0.0}
    tout = list(itertools.product((1, -1), repeat=k))
    sout_S = list(itertools.product((1, -1), repeat=len(S_idx)))
    for name, sigma in states.items():
        p_c = [trace(mmul(sigma, e)) for e in ex_list]
        raw = [v.real for v in p_c]
        res["max_imag"] = max(res["max_imag"], max(abs(v.imag) for v in p_c))
        # sign-corrected averaged Q(s) explicit
        Q = {s: 0j for s in outcomes}
        for st in settings:
            w = setting_weight(st)
            if w == 0:
                continue
            bd = born_dist(sigma, st)
            signs = [sign_of_coord(rs[j], st[j]) for j in range(n)]
            # skip settings hitting zero-coord axes (weight 0 already)
            if any(sg == 0 for sg in signs):
                continue
            for o, prob in bd.items():
                s = tuple(signs[j] * o[j] for j in range(n))
                Q[s] += prob * w
        # degrade K bits classically
        Qdeg = {}
        for tK in tout:
            for sS in sout_S:
                Qdeg[(tK, sS)] = 0j
        for s, prob in Q.items():
            sK = tuple(s[j] for j in K_idx)
            sS = tuple(s[j] for j in S_idx)
            for tK in tout:
                w = 1.0
                for jj, j in enumerate(K_idx):
                    m = mj[jj]
                    w *= ((1 + m) / 2 if tK[jj] == sK[jj] else (1 - m) / 2)
                Qdeg[(tK, sS)] += prob * w
        # prediction from p: sum_x p_x prod_K BSC_common prod_S delta
        for tK in tout:
            for sS in sout_S:
                pred = 0.0
                for rv, x in zip(raw, xs):
                    xK = tuple(x[j] for j in K_idx)
                    xS = tuple(x[j] for j in S_idx)
                    want_S = tuple(1 if b == 0 else -1 for b in xS)
                    if sS != want_S:
                        continue
                    bw = 1.0
                    for tj, xj in zip(tK, xK):
                        bw *= bsc_w(tj, xj, eta_common)
                    pred += rv * bw
                res["max_deg_err"] = max(res["max_deg_err"],
                                         abs(Qdeg[(tK, sS)].real - pred))
                res["max_imag"] = max(res["max_imag"], abs(Qdeg[(tK, sS)].imag))
        # flag moments with concrete g
        p0 = raw[xs.index((0, 0, 0))]
        EgF = 0.0
        Eg2F = 0.0
        for (tK, sS), prob in Qdeg.items():
            F = 1.0 if sS == tuple([1] * len(S_idx)) else 0.0
            g = g_score(tK)
            EgF += prob.real * g * F
            Eg2F += prob.real * (g * F) ** 2
        res["max_flag_bias"] = max(res["max_flag_bias"], abs(EgF - p0))
        res["max_second"] = max(res["max_second"], Eg2F)
    assert res["max_second"] <= res["second_bound"] + 1e-9
    # legality of common-signal lower coins |r_ji/a_*| <= 1
    a_star = max(a_of(rs[j]) for j in K_idx)
    for j in K_idx:
        for v in rs[j]:
            assert abs(v) / a_star <= 1.0 + 1e-12
    res["a_star"] = a_star
    res["eta_star"] = (1 - a_star) / 2.0
    return res


def check_lower_adaptive():
    # concrete policy on n=2 with signed+zero r = (-1,1,0)/sqrt2
    r = (-1 / SQRT2, 1 / SQRT2, 0.0)
    a = a_of(r)
    eta_star = (1 - a) / 2.0
    n = 2
    rs = [r, r]
    xs = all_x(n)
    ex_list = [r_projector_tensor(x, rs) for x in xs]
    gen = lcg(SEED + 77)
    raw_w = [next(gen) + 0.1 for _ in xs]
    tot = sum(raw_w)
    p = [v / tot for v in raw_w]
    sigma_p = zeros(4)
    for w, e in zip(p, ex_list):
        sigma_p = madd(sigma_p, e, 1.0, w)

    def coord(ax):
        return {"X": r[0], "Y": r[1], "Z": r[2]}[ax]

    # per-axis Born vs scalar formula (explicit matrices vs coords)
    max_mean_err = 0.0
    for st in all_settings(n):
        bd = born_dist(sigma_p, st)
        for o in all_outcomes(n):
            pred = 0.0
            for w, x in zip(p, xs):
                f = 1.0
                for j in range(n):
                    rc = coord(st[j])
                    bit = (-1) ** x[j]
                    f *= (1 + o[j] * bit * rc) / 2.0
                pred += w * f
            max_mean_err = max(max_mean_err, abs(bd[o].real - pred))
    # adaptive policy: q0 X; if o0==+1 then q1 Y else q1 Z
    transcripts = list(itertools.product((1, -1), repeat=2))
    q_quant = {}
    for (o0, o1) in transcripts:
        a1 = "Y" if o0 == 1 else "Z"
        proj = proj_tensor(("X", a1), (o0, o1))
        q_quant[(o0, o1)] = trace(mmul(sigma_p, proj)).real
    assert abs(sum(q_quant.values()) - 1.0) < 1e-9
    # simulation via base BSC_a + coins
    def bsc_a_prob(y, x):
        return bsc_w(y, x, eta_star)

    def coin_prob(z, mean):
        # z in +-1, mean in [-1,1]
        return (1 + z * mean) / 2.0

    q_sim = {}
    rX, rY, rZ = r[0], r[1], r[2]
    for (o0, o1) in transcripts:
        a1 = "Y" if o0 == 1 else "Z"
        r0 = rX
        r1 = rY if a1 == "Y" else rZ
        m0 = r0 / a
        m1 = r1 / a
        tot_p = 0.0
        for w, x in zip(p, xs):
            x0, x1 = x
            inner = 0.0
            for y0 in (1, -1):
                for y1 in (1, -1):
                    bw = bsc_a_prob(y0, x0) * bsc_a_prob(y1, x1)
                    cw = coin_prob(o0 * y0, m0) * coin_prob(o1 * y1, m1)
                    inner += bw * cw
            tot_p += w * inner
        q_sim[(o0, o1)] = tot_p
    max_adapt_err = max(abs(q_quant[t] - q_sim[t]) for t in transcripts)
    return {"r": list(r), "a": a, "eta_star": eta_star, "p": p,
            "max_mean_err": max_mean_err, "max_adapt_err": max_adapt_err,
            "q_quant": [q_quant[t] for t in transcripts],
            "q_sim": [q_sim[t] for t in transcripts]}


def check_falsification_and_noise():
    # diagonal mixture n=1: asymmetric r axes genuinely differ
    r_asym = (2 / 3, 1 / 3, 2 / 3)
    r111 = (1 / SQRT3, 1 / SQRT3, 1 / SQRT3)
    out = {}
    for label, r in (("asym", r_asym), ("111", r111)):
        rs = [r]
        xs = all_x(1)
        ex_list = [r_projector_tensor(x, rs) for x in xs]
        gen = lcg(SEED + 11)
        w0 = [next(gen) + 0.1 for _ in xs]
        tot = sum(w0)
        p = [v / tot for v in w0]
        sigma_p = zeros(2)
        for w, e in zip(p, ex_list):
            sigma_p = madd(sigma_p, e, 1.0, w)
        dists = [born_dist(sigma_p, (ax,)) for ax in PAULI_LIST]
        outcomes = all_outcomes(1)
        spread = 0.0
        for s in outcomes:
            vals = [d[s].real for d in dists]
            spread = max(spread, max(vals) - min(vals))
        L = L_of(r)
        a = a_of(r)
        out[label] = {"spread": spread, "L": L, "a": a,
                      "eta_up": (1 - 1 / L) / 2, "eta_low": (1 - a) / 2}
    out["noise_diff_asym"] = abs(out["asym"]["eta_up"] - out["asym"]["eta_low"])
    out["noise_diff_111"] = abs(out["111"]["eta_up"] - out["111"]["eta_low"])
    return out


def check_overlap_identities():
    # heterogeneous instance + tilt instance
    rA = (1 / SQRT3, 1 / SQRT3, 1 / SQRT3)
    rB = (1 / SQRT2, 1 / SQRT2, 0.0)
    rC = (0.0, 0.0, 1.0)
    th = 0.04 / math.sqrt(3.0)
    rT = (math.sin(th), 0.0, math.cos(th))
    cases = {"hetero": [rA, rB, rC], "tilt3": [rT, rT, rT]}
    res = {}
    for name, rs in cases.items():
        n = len(rs)
        P = target_projector(rs)
        Qm = tensor_list([pi_plus(r)[0] for r in rs])
        aj = [a_of(r) for r in rs]
        c_form = 1.0
        for a in aj:
            c_form *= (1 + a) / 2.0
        c_trace = trace(mmul(P, Qm)).real
        D = msub(P, Qm)
        trD = trace(D).real
        D2 = mmul(D, D)
        trD2 = trace(D2).real
        D3 = mmul(D2, D)
        E = madd(D3, D, 1.0, -(1 - c_form))
        cubic = max_abs_entry(E)
        op = math.sqrt(max(0.0, 1 - c_form))
        analytic = math.sqrt(sum((1 - a) / 2.0 for a in aj))
        res[name] = {"c_form": c_form, "c_trace": c_trace,
                     "c_err": abs(c_form - c_trace), "trD": abs(trD),
                     "trD2_err": abs(trD2 - (2 - 2 * c_form)), "cubic": cubic,
                     "op": op, "analytic": analytic,
                     "bound_holds": analytic + 1e-12 >= op,
                     "a_list": aj, "L_list": [L_of(r) for r in rs]}
    # near-axis sufficient check for tilt3 at eps0
    t = res["tilt3"]
    t["eps_half"] = EPS0 / 2
    t["sufficient_holds"] = t["op"] <= EPS0 / 2 + 1e-12
    t["all_nonstabilizer"] = all(
        abs(r[0]) > 1e-12 and L_of(r) > 1 + 1e-12 and a_of(r) < 1 - 1e-12
        for r in cases["tilt3"]
    )
    # fixed nonzero tilt gaps
    rF = (math.sin(0.5), 0.0, math.cos(0.5))
    t["fixed_L"] = L_of(rF)
    t["fixed_a"] = a_of(rF)
    t["fixed_hard"] = (t["fixed_L"] > 1.01 and t["fixed_a"] < 0.99)
    # all-n analytic near-axis certificate for theta_n=c0/sqrt(n), c0=0.04:
    # 1-cos(t)<=t^2/2 gives ||P-Q||<=c0/2 for every n (not just n=3).
    c0 = 0.04
    all_n_ok = True
    all_n_worst = 0.0
    for n in (1, 2, 3, 4, 8, 16, 64, 256, 1024):
        thn = c0 / math.sqrt(n)
        an = math.cos(thn)
        op_n = math.sqrt(max(0.0, 1 - ((1 + an) / 2.0) ** n))
        cert_n = math.sqrt(n * (1 - an) / 2.0)
        ana_n = c0 / 2.0
        # cert via 1-cos<=t^2/2: n*(1-cos)/2 <= c0^2/4
        if not (cert_n <= ana_n + 1e-12 and op_n <= cert_n + 1e-12
                and op_n <= EPS0 / 2 + 1e-12):
            all_n_ok = False
        all_n_worst = max(all_n_worst, op_n)
    t["all_n_c0"] = c0
    t["all_n_ok"] = all_n_ok
    t["all_n_worst_op"] = all_n_worst
    return res


def _partial_trace_first(sigma):
    # 2-qubit kron(first, second) ordering: idx=r0*2+r1.
    red = [[0j, 0j], [0j, 0j]]
    for r0 in range(2):
        for c0 in range(2):
            s = 0j
            for k in range(2):
                s += sigma[r0 * 2 + k][c0 * 2 + k]
            red[r0][c0] = s
    return red


def _partial_trace_second(sigma):
    red = [[0j, 0j], [0j, 0j]]
    for r1 in range(2):
        for c1 in range(2):
            s = 0j
            for k in range(2):
                s += sigma[k * 2 + r1][k * 2 + c1]
            red[r1][c1] = s
    return red


def _local_score_expectation(single, r):
    # Explicit Born/matrix check: E[z]=(1+L E[s])/2 with axis sampling.
    # E[s]=sum_i p_i sign(r_i) Tr[single P_i] via (I+-P)/2 projectors.
    L = L_of(r)
    f_direct = trace(mmul(single, tau_pm(r, 1))).real
    exp_s = 0.0
    for ax, idx in (("X", 0), ("Y", 1), ("Z", 2)):
        p_ax = abs(r[idx]) / L
        if p_ax == 0:
            continue
        sg = sign_of_coord(r, ax)
        A = PAULI[ax]
        p_plus = trace(mmul(single, [[(I_M[i][j] + A[i][j]) / 2.0
                                      for j in range(2)] for i in range(2)])).real
        p_minus = trace(mmul(single, [[(I_M[i][j] - A[i][j]) / 2.0
                                       for j in range(2)] for i in range(2)])).real
        exp_o = p_plus - p_minus
        exp_s += p_ax * sg * exp_o
    exp_z = (1.0 + L * exp_s) / 2.0
    return f_direct, exp_z


def check_product_correlation():
    # Sec 4: product-input promise vs separable-correlated inputs.
    # Target-product promise does NOT imply input-product promise.
    r = (1 / SQRT3, 1 / SQRT3, 1 / SQRT3)
    L = L_of(r)
    tau_p = tau_pm(r, 1)
    tau_m = tau_pm(r, -1)
    target = kron(tau_p, tau_p)
    tauI = kron(tau_p, I_M)
    Itau = kron(I_M, tau_p)
    res = {"r": list(r), "L": L, "max_local_err": 0.0,
           "max_product_fact_err": 0.0, "cases": {}}
    # Product mixed cases (input-product promise holds).
    s1_diag = madd(tau_p, tau_m, 0.7, 0.3)
    s2_diag = madd(tau_p, tau_m, 0.2, 0.8)
    plus1 = [[0.5 + 0j, 0.5 + 0j], [0.5 + 0j, 0.5 + 0j]]
    comp0 = [[1 + 0j, 0j], [0j, 0j]]
    mix = mscale(eye(2), 0.5)
    prod_cases = {
        "diag_mix": (s1_diag, s2_diag, 0.7, 0.2, 0.14),
        "coherent_product": (comp0, plus1, None, None, None),
        "maxmixed_product": (mix, tau_p, 0.5, 1.0, 0.5),
    }
    for name, (s1, s2, f1_want, f2_want, fid_want) in prod_cases.items():
        sigma = kron(s1, s2)
        f1 = trace(mmul(sigma, tauI)).real
        f2 = trace(mmul(sigma, Itau)).real
        fid = trace(mmul(sigma, target)).real
        f1d, e1 = _local_score_expectation(s1, r)
        f2d, e2 = _local_score_expectation(s2, r)
        # reduced-state route (same numbers, independent partial trace)
        r1 = _partial_trace_first(sigma)
        r2 = _partial_trace_second(sigma)
        f1r = trace(mmul(r1, tau_p)).real
        f2r = trace(mmul(r2, tau_p)).real
        _, e1r = _local_score_expectation(r1, r)
        _, e2r = _local_score_expectation(r2, r)
        res["max_local_err"] = max(res["max_local_err"], abs(f1d - e1),
                                   abs(f2d - e2), abs(f1r - e1r),
                                   abs(f2r - e2r), abs(f1 - f1d),
                                   abs(f2 - f2d))
        res["max_product_fact_err"] = max(res["max_product_fact_err"],
                                          abs(fid - f1 * f2))
        entry = {"f1": f1, "f2": f2, "fid": fid, "prod": f1 * f2,
                 "e1": e1, "e2": e2, "fact_err": abs(fid - f1 * f2),
                 "local_err": max(abs(f1d - e1), abs(f2d - e2))}
        if f1_want is not None:
            entry["want_err"] = max(abs(f1 - f1_want), abs(f2 - f2_want),
                                    abs(fid - fid_want))
            res["max_local_err"] = max(res["max_local_err"], entry["want_err"])
        res["cases"][name] = entry
    # Correlated separable counterexample: mixture of two products.
    sigma_corr = madd(kron(tau_p, tau_p), kron(tau_m, tau_m), 0.5, 0.5)
    f1c = trace(mmul(sigma_corr, tauI)).real
    f2c = trace(mmul(sigma_corr, Itau)).real
    fidc = trace(mmul(sigma_corr, target)).real
    prodc = f1c * f2c
    rc1 = _partial_trace_first(sigma_corr)
    rc2 = _partial_trace_second(sigma_corr)
    _, e1c = _local_score_expectation(rc1, r)
    _, e2c = _local_score_expectation(rc2, r)
    res["max_local_err"] = max(res["max_local_err"], abs(f1c - 0.5),
                               abs(f2c - 0.5), abs(e1c - 0.5), abs(e2c - 0.5))
    res["cases"]["correlated_mixture"] = {
        "f1": f1c, "f2": f2c, "fid": fidc, "prod": prodc,
        "e1": e1c, "e2": e2c,
        "gap_fid_vs_prod": abs(fidc - prodc),
        "want_fid": 0.5, "want_prod": 0.25,
        "fid_err": abs(fidc - 0.5), "prod_err": abs(prodc - 0.25),
    }
    # Sufficient polynomial bound instance for product promise (Sec 4.2):
    # N>=3n^2/(2eps^2) ln(2n/delta) at n=2, eps0=1/20, delta=1/3.
    n2 = 2
    N_suff = math.ceil(3 * n2 * n2 / (2 * EPS0 * EPS0)
                       * math.log(2 * n2 / DELTA0))
    res["N_suff_product_n2"] = N_suff
    res["N_suff_formula"] = "ceil(3n^2/(2eps^2) ln(2n/delta))"
    return res


def main():
    rs_povm = [
        (1 / SQRT3, 1 / SQRT3, 1 / SQRT3),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (1 / SQRT2, 1 / SQRT2, 0.0),
        (-1 / SQRT2, 1 / SQRT2, 0.0),
        (2 / 3, 1 / 3, 2 / 3),
        (-1 / 3, 2 / 3, 2 / 3),
        (0.6, 0.0, 0.8),
        (math.sin(0.5), 0.0, math.cos(0.5)),
    ]
    povm = check_single_qubit_povm(rs_povm)
    homog_rs = [
        (1 / SQRT3, 1 / SQRT3, 1 / SQRT3),
        (1 / SQRT2, 1 / SQRT2, 0.0),
        (-1 / SQRT2, 1 / SQRT2, 0.0),
        (2 / 3, 1 / 3, 2 / 3),
    ]
    per_n = []
    for n in NS:
        for r in homog_rs:
            # limit n=3 runtime: all four still fine (explicit); keep all.
            per_n.append(check_homogeneous(n, r))
    pauli = [check_pauli_indicator(n) for n in NS]
    het = check_heterogeneous()
    low = check_lower_adaptive()
    fals = check_falsification_and_noise()
    ov = check_overlap_identities()
    corr = check_product_correlation()

    overall = max([
        povm["max_povm_err"], povm["max_norm_err"], povm["max_pos_viol"], povm["max_comm_err"],
        max(r["max_avg_vs_bsc"] for r in per_n),
        max(r["max_p_sum_err"] for r in per_n),
        max(r["max_p_neg"] for r in per_n),
        max(r["max_fid_err"] for r in per_n),
        max(r["max_offdiag"] for r in per_n),
        max(r["max_norm_err"] for r in per_n),
        max(r["max_imag"] for r in per_n),
        max(r["max_indicator_err"] for r in pauli),
        het["max_deg_err"], het["max_flag_bias"], het["max_imag"],
        low["max_mean_err"], low["max_adapt_err"],
        ov["hetero"]["c_err"], ov["hetero"]["trD"], ov["hetero"]["trD2_err"],
        ov["hetero"]["cubic"],
        ov["tilt3"]["c_err"], ov["tilt3"]["trD"], ov["tilt3"]["trD2_err"],
        ov["tilt3"]["cubic"],
        corr["max_local_err"], corr["max_product_fact_err"],
        corr["cases"]["correlated_mixture"]["fid_err"],
        corr["cases"]["correlated_mixture"]["prod_err"],
    ])
    # falsification must show spread large for asym and ~0 for 111 (not part of max)
    # Asym spread 0.03 is >7 orders above TOL: raw axes genuinely differ.
    fals_ok = (fals["asym"]["spread"] > 0.01) and (fals["111"]["spread"] <= 1e-9)
    noise_ok = (fals["noise_diff_asym"] > 1e-3) and (fals["noise_diff_111"] <= 1e-12)
    overlap_ok = (
        ov["hetero"]["bound_holds"] and ov["tilt3"]["bound_holds"]
        and ov["tilt3"]["sufficient_holds"] and ov["tilt3"]["all_nonstabilizer"]
        and ov["tilt3"]["fixed_hard"] and ov["tilt3"]["all_n_ok"]
    )
    corr_ok = (
        corr["cases"]["correlated_mixture"]["gap_fid_vs_prod"] > 0.24
        and corr["cases"]["correlated_mixture"]["fid_err"] <= TOL
        and corr["cases"]["correlated_mixture"]["prod_err"] <= TOL
    )
    pauli_N_const = len({r["N_star"] for r in pauli}) == 1
    status = "PASS" if (overall <= TOL and fals_ok and noise_ok and overlap_ok and corr_ok and pauli_N_const) else "FAIL"
    payload = {
        "status": status,
        "scope": "Finite algebra support only (n<=3 explicit matrices); not an asymptotic-theorem or novelty check.",
        "seed": SEED,
        "tolerance": TOL,
        "python": platform.python_version(),
        "eps0": EPS0,
        "delta0": DELTA0,
        "overall_max_error": overall,
        "single_qubit_povm": povm,
        "homogeneous": per_n,
        "pauli_indicator": pauli,
        "heterogeneous": het,
        "lower_adaptive": low,
        "falsification_noise": fals,
        "overlap": ov,
        "product_correlation": corr,
        "gates": {
            "falsification_spread_asym": fals["asym"]["spread"],
            "falsification_spread_111": fals["111"]["spread"],
            "falsification_ok": fals_ok,
            "noise_diff_asym": fals["noise_diff_asym"],
            "noise_diff_111": fals["noise_diff_111"],
            "noise_ok": noise_ok,
            "overlap_ok": overlap_ok,
            "corr_ok": corr_ok,
            "corr_gap": corr["cases"]["correlated_mixture"]["gap_fid_vs_prod"],
            "pauli_N_const": pauli_N_const,
        },
        "notes": [
            "Born probabilities use explicit (I+sA)/2 projectors and kron tensor products; BSC predictions use scalar R-diagonal weights.",
            "Weighted sign correction multiplies outcome by sign(r_i) with axis sampled w.p. |r_i|/L; zero coords have weight 0 and are never selected.",
            "Lower coins have mean r_i/a (negative biases to -1, zero gives fair coin); base BSC_a pre-sampled, revealed in algorithm-chosen order.",
            "Adaptive check enumerates a concrete within-copy policy (X then Y/Z by first outcome) path by path.",
            "Falsification spread shows raw per-axis channels differ for asymmetric r; averaged mixture is the BSC.",
            "Overlap cubic identity plus HS norm imply rank-two spectrum {+s,-s,0} with s=sqrt(1-c); analytic bound is Weierstrass.",
            "All-n near-axis certificate uses 1-cos(t)<=t^2/2 giving ||P-Q||<=c0/2 for every n.",
            "Product-correlation: local scores E[z_j]=f_j on product mixed inputs via explicit Born; correlated mixture has fid 1/2 vs marginal product 1/4.",
            "Target-product promise does not imply input-product promise; separable-correlated gap is promise distinction, not entanglement witness.",
            "No sampling randomness; deterministic states plus LCG diagonal p.",
        ],
    }
    dest = (Path(__file__).resolve().parents[1] / "results" / "product_state_boundary_checks.json")
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if status != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
