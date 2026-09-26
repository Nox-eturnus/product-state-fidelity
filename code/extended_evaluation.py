"""Extended Monte Carlo evaluation.

Fixed comparison scope:
all 8 methods, both hom_111/hom_near targets, n=1,2,4,6, N=4096,
corr07 input, eps=0.05, delta=1/3: 64 rows, 10000 trials each.
The comparison scope is fixed below; all specified rows are retained.

Reuses fixed law/scoring/certificate routines from
finite_size_evaluation.py + product_state_estimator.py +
product_state_geometry.py with the same (k,N,channel) scores as frozen.
No refitting of methods, no change to frozen files. Genuine joint
correlated-mixture law preserved (0.7 target + 0.3 all-opposite,
single latent draw per copy via frozen corr07 mixture logic).
Does NOT call frozen run_evaluation test-grid seam as reportable and
does NOT mutate production globals (local caches only; frozen design
hash revalidated at entry; source hashes recorded).

Sampling: separate deterministic master seed 20260921 (recorded).
Per-row exact ID {tid}|n{n}|corr07|{method}|N4096. Per-repetition seed
int.from_bytes(SHA256(f'{master}|{tid_full}|corr07|{method}|4096|{rep}').digest()[:16],'big'),
rep 0..9999, each rep its own numpy Generator + one multinomial(4096,p).
Full 128-bit seeds with no 32-bit modulus; uniqueness within each row is
asserted and the run fails on any within-row duplicate (previously identified
collision in old hom_near n2 lp_half addressed). Distributionally equivalent to iid full-copy sampling. Raw means saved
for every row. 95% binomial intervals: Wilson (z=1.96) plus exact
one-sided zero-event upper 1-0.05**(1/R). Every selected row and any
failure retained. Trials cannot infer worst-case guarantee (stated).
Historical 100-trial 1360-row design kept distinct (different file,
seed, reps, scope).

Run: python code/extended_evaluation.py --run
Writes extended_results.json. Requires NumPy and SciPy.

"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
import time
import tracemalloc
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DESIGN_PATH = ROOT.parent / "data" / "finite_size_design.json"
FROZEN_RESULTS_PATH = ROOT.parent / "results" / "finite_size_results.json"
RESULTS_PATH = ROOT.parent / "results" / "extended_results.json"

EPS = 0.05
DELTA = 1.0 / 3.0
N_FROZEN = 4096
REPS_NEW = 10000
SEED_NEW = 20260921
N_LIST = [1, 2, 4, 6]
TIDS = ["hom_111", "hom_near"]
METHODS = ["lp_common", "lp_half", "inv_common", "inv_half", "native", "shadow", "naive", "geometry"]
STATE = "corr07"
BITS = 80
DEN = 256
SELECTOR_BETA_FRAC = Fraction(1, 40)
FROZEN_FILE_HASH = "c1afc09ecb83c62775dcddf1ae8a23e926dd5904d0d10c1b1be46ddc7ba7cbb8"
ZERO_EVENT_UPPER = float(1.0 - 0.05 ** (1.0 / REPS_NEW))

def file_sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def design_canonical_hash(d: dict) -> str:
    c = json.dumps(d, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(c).hexdigest()

def seed_int_for(master: int, tid_full: str, state: str, method: str, N: int, rep: int) -> int:
    key = f"{master}|{tid_full}|{state}|{method}|{N}|{rep}"
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:16], "big")

def ensure_finite(x: float, label: str) -> float:
    f = float(x)
    if not math.isfinite(f):
        raise ValueError(f"nonfinite {label}: {x!r}")
    return f

def wilson_interval(k: int, n: int, z: float = 1.96) -> list:
    if n <= 0:
        raise ValueError("n must be positive")
    if k < 0 or k > n:
        raise ValueError("k out of range")
    p = k / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    inner = p * (1 - p) / n + z * z / (4 * n * n)
    if inner < 0:
        inner = 0.0
    half = z * math.sqrt(inner) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return [ensure_finite(lo, "wilson_lo"), ensure_finite(hi, "wilson_hi")]

def law_stats(scores: list, probs: list, f_true: float) -> dict:
    exp = sum(s * p for s, p in zip(scores, probs))
    exp2 = sum((s ** 2) * p for s, p in zip(scores, probs))
    return {"expectation": ensure_finite(exp, "exp"),
            "second_moment": ensure_finite(exp2, "e2"),
            "bias": ensure_finite(exp - f_true, "bias"),
            "min_score": ensure_finite(min(scores), "min"),
            "max_score": ensure_finite(max(scores), "max")}

def sample_row_stats_new(scores: list, probs: list, N: int, tid_full: str, method: str, f_true: float, radius) -> dict:
    import numpy as np
    if not isinstance(N, int) or isinstance(N, bool) or N <= 0:
        raise ValueError(f"N must be positive int, got {N!r}")
    if len(scores) == 0 or len(scores) != len(probs):
        raise ValueError("scores/probs mismatch")
    scores_arr = np.array(scores, dtype=np.float64)
    probs_arr = np.array(probs, dtype=np.float64)
    probs_arr = probs_arr / probs_arr.sum()
    raw_means: list = []
    seeds: list = []
    for rep in range(REPS_NEW):
        s_rep = seed_int_for(SEED_NEW, tid_full, STATE, method, N, rep)
        seeds.append(int(s_rep))
        rng = np.random.default_rng(int(s_rep))
        try:
            counts = rng.multinomial(N, probs_arr)
        except Exception as e:
            raise RuntimeError(f"multinomial failed N={N} C={len(scores)} rep={rep}: {e}") from e
        m = float((counts.astype(np.float64) @ scores_arr) / float(N))
        raw_means.append(ensure_finite(m, "rep_mean"))
    if len(set(seeds)) != len(seeds):
        raise RuntimeError(f"duplicate 128-bit seeds within row {tid_full}|{method}: {len(seeds)-len(set(seeds))} collisions")
    means = np.array(raw_means, dtype=np.float64)
    clipped = np.clip(means, 0.0, 1.0)
    errs = clipped - f_true
    abs_errs = np.abs(errs)
    rmse = float(np.sqrt(np.mean(errs ** 2)))
    p50 = float(np.quantile(abs_errs, 0.5))
    p90 = float(np.quantile(abs_errs, 0.9))
    pmax = float(np.max(abs_errs))
    exceed = int(np.sum(abs_errs > EPS))
    exceed_frac = exceed / REPS_NEW
    exceed_wilson = wilson_interval(exceed, REPS_NEW)
    exceed_zero_upper = ensure_finite(ZERO_EVENT_UPPER, "zero_up") if exceed == 0 else None
    if radius is None:
        cover_frac = None
        cover_wilson = None
        cover_count = None
        cover_zero_upper = None
    else:
        lo = np.clip(means - radius, 0.0, 1.0)
        hi = np.clip(means + radius, 0.0, 1.0)
        covered = np.logical_and(lo <= f_true + 1e-12, f_true - 1e-12 <= hi)
        cover_count = int(np.sum(covered))
        cover_frac = cover_count / REPS_NEW
        cover_wilson = wilson_interval(cover_count, REPS_NEW)
        n_fail_cover = REPS_NEW - cover_count
        cover_zero_upper = ensure_finite(ZERO_EVENT_UPPER, "zero_up") if n_fail_cover == 0 else None
    seed_blob = ",".join(str(s) for s in seeds).encode("utf-8")
    seed_digest = hashlib.sha256(seed_blob).hexdigest()
    mean_blob = ",".join(repr(float(v)) for v in raw_means).encode("utf-8")
    mean_digest = hashlib.sha256(mean_blob).hexdigest()
    return {"rmse": ensure_finite(rmse, "rmse"),
            "p50": ensure_finite(p50, "p50"),
            "p90": ensure_finite(p90, "p90"),
            "pmax": ensure_finite(pmax, "pmax"),
            "exceed_count": exceed,
            "exceed_frac": ensure_finite(exceed_frac, "exceed"),
            "exceed_wilson": exceed_wilson,
            "exceed_zero_event_upper": exceed_zero_upper,
            "cover_count": cover_count,
            "cover_frac": (ensure_finite(cover_frac, "cover") if cover_frac is not None else None),
            "cover_wilson": cover_wilson,
            "cover_zero_event_upper": cover_zero_upper,
            "rep_means_raw": [ensure_finite(float(v), "rep_raw") for v in raw_means],
            "rep_seeds_digest": seed_digest,
            "rep_means_digest": mean_digest,
            "seed_unique_within_row": True,
            "rng_note": f"per-rep int.from_bytes(SHA256({SEED_NEW}|{{tid_full}}|corr07|{{method}}|4096|rep).digest()[:16],'big'), reps 0..{REPS_NEW-1}, each own Generator + one multinomial; full 128-bit, no modulus, within-row uniqueness asserted"}

def run_extended_evaluation() -> dict:
    t_all0 = time.perf_counter()
    tracemalloc.start()
    import product_state_estimator as est
    import product_state_geometry as geo
    import finite_size_evaluation as fse
    try:
        import numpy
        import scipy
        numpy_ver = numpy.__version__
        scipy_ver = scipy.__version__
    except Exception:
        numpy_ver = "missing"
        scipy_ver = "missing"
    py_ver = platform.python_version()
    design = json.loads(DESIGN_PATH.read_text(encoding="utf-8"))
    file_hash = file_sha256(DESIGN_PATH)
    if file_hash != FROZEN_FILE_HASH:
        raise RuntimeError(f"frozen design hash mismatch: got {file_hash} want {FROZEN_FILE_HASH}")
    canon_hash = design_canonical_hash(design)
    targets_per_n = design["targets_per_n"]
    src_hashes: dict = {}
    for fn in ("product_state_estimator.py", "product_state_geometry.py", "homogeneous_fidelity.py",
               "finite_size_evaluation.py", "extended_evaluation.py", "verify_extended_evaluation.py"):
        p = ROOT / fn
        try:
            src_hashes[fn] = file_sha256(p)
        except Exception as e:
            src_hashes[fn] = f"missing:{e}"
    try:
        frozen_record = json.loads(FROZEN_RESULTS_PATH.read_text(encoding="utf-8"))
        frozen_hashes = frozen_record.get("distribution_source_hashes", frozen_record.get("source_hashes", {}))
    except Exception:
        frozen_hashes = {}
    lp_cache: dict = {}
    lp_cache_details: dict = {}
    lp_cache_stats = {"solves": 0, "hits": 0, "total_solve_sec": 0.0}
    einfo = fse.eta_info_common()
    eta_mid = einfo["eta_mid"]
    max_eta_err = einfo["max_eta_error"]
    inv_cache: dict = {}
    inv_prep: dict = {}
    for k in (0, 1, 2, 4, 6):
        for ch in (est.CHANNEL_COMMON, est.CHANNEL_HALF):
            t0 = time.perf_counter()
            if k == 0:
                inv_cache[(k, ch)] = {"g_frac": [Fraction(1, 1)], "g_float": [1.0],
                                      "G": Fraction(1, 1), "bias_exact": Fraction(0, 1)}
            else:
                if ch == est.CHANNEL_HALF:
                    W = fse.build_W_half_frac(k)
                else:
                    W = fse.build_W_mid_frac(k, eta_mid)
                g_frac = fse.solve_inverse_frac(W)
                G = max(abs(v) for v in g_frac)
                b_ex = Fraction(0, 1) if ch == est.CHANNEL_HALF else Fraction(2, 1) * G * k * max_eta_err
                inv_cache[(k, ch)] = {"g_frac": g_frac, "g_float": [float(v) for v in g_frac],
                                      "G": G, "bias_exact": b_ex}
            t1 = time.perf_counter()
            inv_prep[f"k{k}_{ch}"] = {"prep_time_sec": ensure_finite(t1 - t0, "inv_prep"),
                                      "G": str(inv_cache[(k, ch)]["G"]),
                                      "bias_exact": str(inv_cache[(k, ch)]["bias_exact"])}
    native_global: dict = {}
    shadow_global: dict = {}
    for n in N_LIST:
        for tid in TIDS:
            rs_raw_g = targets_per_n[str(n)][tid]
            rs_list_g = fse._validate_rs_list(rs_raw_g)
            try:
                cls_g = fse.classify_via_production(rs_list_g)
                import product_state_estimator as _est2
                _cls2 = _est2.classify_target(rs_list_g)
                rs_hat_g = [tuple(float(v) for v in r) for r in _cls2["rs_hat"]]
            except Exception:
                rs_hat_g = [tuple(float(v) for v in r) for r in rs_list_g]
            nlo, nhi = fse.native_global_range(rs_hat_g)
            slo, shi = fse.shadow_global_range(rs_hat_g)
            native_global[(n, tid)] = (nlo, nhi)
            shadow_global[(n, tid)] = (slo, shi)
    geometry_prep: dict = {}
    rows: list = []
    n_fail = 0
    total_planned = len(N_LIST) * len(TIDS) * len(METHODS)
    for n in N_LIST:
        for tid in TIDS:
            rs_raw = targets_per_n[str(n)][tid]
            rs_list = fse._validate_rs_list(rs_raw)
            try:
                cls = est.classify_target(rs_list)
            except Exception as e:
                for m0 in METHODS:
                    rows.append({"row_id": f"{tid}|n{n}|{STATE}|{m0}|N{N_FROZEN}", "target_id": tid, "n": n,
                                 "state": STATE, "method": m0, "N": N_FROZEN, "status": "error",
                                 "reason": f"classify failed: {type(e).__name__}: {e}"[:500]})
                    n_fail += 1
                continue
            k_orig = int(cls["k"])
            rs_hat = [tuple(float(v) for v in r) for r in cls["rs_hat"]]
            K_idx = list(cls["K_idx"])
            S_idx = list(cls["S_idx"])
            stab_axis = list(cls["stab_axis"])
            stab_want = list(cls["stab_want"])
            naive_axes, naive_wants = fse.nearest_axes_for_rs(rs_hat)
            t_geo0 = time.perf_counter()
            try:
                sel = geo.select_discards(rs_list, SELECTOR_BETA_FRAC, BITS)
                D_geo = tuple(sel["chosen_D"])
                proxy_hat_geo = [tuple(float(v) for v in r) for r in sel["proxy"]]
                cls_proxy0 = est.classify_target(list(sel["proxy"]))
                k_proxy0 = int(cls_proxy0["k"])
                d_hi_frac0 = sel["d_hi"] if isinstance(sel["d_hi"], Fraction) else Fraction(sel["d_hi"])
            except Exception as e:
                sel = None
                D_geo = None
                proxy_hat_geo = None
                k_proxy0 = None
                d_hi_frac0 = None
                geo_sel_error = f"{type(e).__name__}: {e}"
            else:
                geo_sel_error = None
            t_geo1 = time.perf_counter()
            geometry_prep[f"n{n}_{tid}"] = {"select_time_sec": ensure_finite(t_geo1 - t_geo0, "geo_prep"),
                                            "D": sorted(list(D_geo)) if D_geo is not None else None,
                                            "d_hi": str(d_hi_frac0) if d_hi_frac0 is not None else None,
                                            "k_proxy": k_proxy0, "k_original": k_orig,
                                            "error": geo_sel_error}
            try:
                f_true = fse.f_true_for(n, rs_hat, STATE)
            except Exception as e:
                for m0 in METHODS:
                    rows.append({"row_id": f"{tid}|n{n}|{STATE}|{m0}|N{N_FROZEN}", "target_id": tid, "n": n,
                                 "k_original": k_orig, "N": N_FROZEN, "nN": n * N_FROZEN, "state": STATE,
                                 "method": m0, "status": "error", "reason": f"f_true failed: {e}"[:500]})
                    n_fail += 1
                continue
            for method in METHODS:
                t_row0 = time.perf_counter()
                tid_full = f"{tid}|n{n}"
                row_id = f"{tid}|n{n}|{STATE}|{method}|N{N_FROZEN}"
                row: dict = {"row_id": row_id, "target_id": tid, "n": n, "k_original": k_orig,
                             "N": N_FROZEN, "nN": n * N_FROZEN, "state": STATE, "method": method,
                             "f_true": ensure_finite(f_true, "f_true")}
                try:
                    if method in ("lp_common", "lp_half"):
                        ch = est.CHANNEL_COMMON if method == "lp_common" else est.CHANNEL_HALF
                        rec = fse.solve_lp_cached(lp_cache, k_orig, N_FROZEN, ch, details=lp_cache_details, stats=lp_cache_stats)
                        b_cert = Fraction(rec["b_cert"])
                        G_cert = Fraction(rec["G_cert"])
                        g_float = [float(Fraction(s)) for s in rec["g_round"]]
                        if k_orig == 0:
                            R_frac = est._bernoulli_radius_frac(N_FROZEN, DELTA)
                        else:
                            R_frac = est._hoeffding_radius_frac(b_cert, G_cert, N_FROZEN, DELTA)
                        R_float = float(est._float_upper(R_frac))
                        if k_orig == 0:
                            need, reason = est.sufficient_copies_bernoulli(Fraction(0, 1), DELTA, EPS)
                        else:
                            need, reason = est.sufficient_copies_conservative(b_cert, G_cert, DELTA, EPS)
                        scores, probs = fse.lp_family_law_product(rs_hat, rs_hat, K_idx, S_idx, stab_axis, stab_want, g_float, ch, STATE)
                        stats = law_stats(scores, probs, f_true)
                        samp = sample_row_stats_new(scores, probs, N_FROZEN, tid_full, method, f_true, R_float)
                        row.update({"status": "ok", "channel": ch, "k": k_orig,
                                    "g_round": list(rec["g_round"]),
                                    "b_cert": str(b_cert), "G_cert": str(G_cert),
                                    "radius": ensure_finite(R_float, "R"), "radius_exact": str(R_frac),
                                    "certificate_type": "production-certified",
                                    "sufficient_copies": need, "sufficient_reason": reason,
                                    "law_entries": len(scores),
                                    "expectation": stats["expectation"], "second_moment": stats["second_moment"],
                                    "unclipped_bias": stats["bias"],
                                    "range_lo": stats["min_score"], "range_hi": stats["max_score"]})
                        row.update(samp)
                        row["certified_eps"] = bool(R_float <= EPS)
                    elif method in ("inv_common", "inv_half"):
                        ch = est.CHANNEL_COMMON if method == "inv_common" else est.CHANNEL_HALF
                        inv = inv_cache[(k_orig, ch)]
                        g_float = list(inv["g_float"])
                        G_frac = inv["G"]
                        b_exact_inv = inv["bias_exact"]
                        if k_orig == 0:
                            b_exact_inv = Fraction(0, 1)
                            R_frac = est.bernoulli_radius_exact(N_FROZEN, DELTA)
                            R_float = est._float_upper(R_frac)
                            R_frac_s = str(R_frac)
                            lo_r, hi_r = 0.0, 1.0
                            range_exact_s = "[0,1]"
                        else:
                            if ch == est.CHANNEL_HALF:
                                W_chk = fse.build_W_half_frac(k_orig)
                            else:
                                W_chk = fse.build_W_mid_frac(k_orig, eta_mid)
                            for _h in range(k_orig + 1):
                                _s = sum(W_chk[_w][_h] * inv["g_frac"][_w] for _w in range(k_orig + 1))
                                _want = Fraction(1, 1) if _h == 0 else Fraction(0, 1)
                                if _s != _want:
                                    raise RuntimeError(f"inverse residual failed k={k_orig}")
                            if max(abs(_v) for _v in inv["g_frac"]) != G_frac:
                                raise RuntimeError("inverse range mismatch")
                            R_frac = est.hoeffding_radius_exact(b_exact_inv, G_frac, N_FROZEN, DELTA)
                            R_float = est._float_upper(R_frac)
                            R_frac_s = str(R_frac)
                            lo_r = est._float_lower(Fraction(0, 1) - G_frac)
                            hi_r = est._float_upper(G_frac)
                            range_exact_s = f"[-{G_frac},{G_frac}]"
                        if k_orig == 0:
                            need, reason = est.sufficient_copies_bernoulli(Fraction(0, 1), DELTA, EPS)
                        else:
                            need, reason = est.sufficient_copies_conservative(b_exact_inv, G_frac, DELTA, EPS)
                        scores, probs = fse.lp_family_law_product(rs_hat, rs_hat, K_idx, S_idx, stab_axis, stab_want, g_float, ch, STATE)
                        stats = law_stats(scores, probs, f_true)
                        samp = sample_row_stats_new(scores, probs, N_FROZEN, tid_full, method, f_true, R_float)
                        row.update({"status": "ok", "channel": ch, "k": k_orig,
                                    "g_inv": [str(v) for v in inv["g_frac"]],
                                    "G_inv": str(G_frac), "bias_exact": str(b_exact_inv),
                                    "radius": ensure_finite(R_float, "R"), "radius_exact": R_frac_s,
                                    "certificate_type": "rigorous-exact-rational",
                                    "range_exact": range_exact_s,
                                    "sufficient_copies": need, "sufficient_reason": reason,
                                    "law_entries": len(scores),
                                    "expectation": stats["expectation"], "second_moment": stats["second_moment"],
                                    "unclipped_bias": stats["bias"]})
                        row.update(samp)
                        row["certified_eps"] = bool(R_float <= EPS)
                    elif method == "native":
                        scores, probs = fse.native_law_product(rs_hat, K_idx, S_idx, stab_axis, stab_want, STATE)
                        stats = law_stats(scores, probs, f_true)
                        glo_lo, glo_hi = native_global[(n, tid)]
                        R_float = fse.radius_global_numerical(glo_hi, glo_lo, N_FROZEN)
                        need, reason = fse.sufficient_copies_global_numerical(glo_hi, glo_lo, Fraction(0, 1))
                        samp = sample_row_stats_new(scores, probs, N_FROZEN, tid_full, method, f_true, R_float)
                        row.update({"status": "ok", "k": k_orig,
                                    "range_lo": ensure_finite(glo_lo, "lo"), "range_hi": ensure_finite(glo_hi, "hi"),
                                    "range_global": True, "range_target_only": True,
                                    "certificate_type": "numerical-Hoeffding",
                                    "radius": ensure_finite(R_float, "R"),
                                    "sufficient_copies": need, "sufficient_reason": reason,
                                    "law_entries": len(scores),
                                    "expectation": stats["expectation"], "second_moment": stats["second_moment"],
                                    "unclipped_bias": stats["bias"]})
                        row.update(samp)
                        row["certified_eps"] = bool(R_float <= EPS)
                    elif method == "shadow":
                        scores, probs = fse.shadow_law_product(rs_hat, STATE)
                        if len(scores) > 100000:
                            raise RuntimeError(f"dense law {len(scores)} exceeds ceiling")
                        stats = law_stats(scores, probs, f_true)
                        glo_lo, glo_hi = shadow_global[(n, tid)]
                        R_float = fse.radius_global_numerical(glo_hi, glo_lo, N_FROZEN)
                        need, reason = fse.sufficient_copies_global_numerical(glo_hi, glo_lo, Fraction(0, 1))
                        samp = sample_row_stats_new(scores, probs, N_FROZEN, tid_full, method, f_true, R_float)
                        row.update({"status": "ok", "k": k_orig,
                                    "range_lo": ensure_finite(glo_lo, "lo"), "range_hi": ensure_finite(glo_hi, "hi"),
                                    "range_global": True, "range_target_only": True,
                                    "certificate_type": "numerical-Hoeffding",
                                    "radius": ensure_finite(R_float, "R"),
                                    "sufficient_copies": need, "sufficient_reason": reason,
                                    "law_entries": len(scores),
                                    "expectation": stats["expectation"], "second_moment": stats["second_moment"],
                                    "unclipped_bias": stats["bias"]})
                        row.update(samp)
                        row["certified_eps"] = bool(R_float <= EPS)
                    elif method == "naive":
                        scores, probs = fse.naive_law_product(rs_hat, naive_axes, naive_wants, STATE)
                        stats = law_stats(scores, probs, f_true)
                        samp = sample_row_stats_new(scores, probs, N_FROZEN, tid_full, method, f_true, None)
                        row.update({"status": "ok", "k": k_orig,
                                    "range_lo": 0.0, "range_hi": 1.0,
                                    "certificate_type": "no-fidelity-interval",
                                    "radius": None,
                                    "law_entries": len(scores),
                                    "expectation": stats["expectation"], "second_moment": stats["second_moment"],
                                    "unclipped_bias": stats["bias"],
                                    "naive_axes": list(naive_axes), "naive_wants": list(naive_wants)})
                        row.update(samp)
                        row["cover_frac"] = None
                        row["cover_wilson"] = None
                        row["cover_count"] = None
                        row["cover_zero_event_upper"] = None
                        row["certified_eps"] = False
                    elif method == "geometry":
                        if geo_sel_error is not None:
                            raise RuntimeError(f"selector failed: {geo_sel_error}")
                        assert D_geo is not None and proxy_hat_geo is not None and k_proxy0 is not None and d_hi_frac0 is not None
                        rec_in = fse.solve_lp_cached(lp_cache, k_proxy0, N_FROZEN, est.CHANNEL_COMMON, details=lp_cache_details, stats=lp_cache_stats)
                        b_in = Fraction(rec_in["b_cert"])
                        G_in = Fraction(rec_in["G_cert"])
                        g_float = [float(Fraction(s)) for s in rec_in["g_round"]]
                        b_tot = b_in + d_hi_frac0
                        if k_proxy0 == 0:
                            R_in_frac = est._bernoulli_radius_frac(N_FROZEN, DELTA)
                        else:
                            R_in_frac = est._hoeffding_radius_frac(b_in, G_in, N_FROZEN, DELTA)
                        R_tot_frac = R_in_frac + d_hi_frac0
                        R_tot_float = float(est._float_upper(R_tot_frac))
                        if b_tot >= Fraction(EPS):
                            need, reason = None, f"b_total={float(b_tot):.6g} >= eps={EPS}"
                        else:
                            if k_proxy0 == 0:
                                need, reason = est.sufficient_copies_bernoulli(b_tot, DELTA, EPS)
                            else:
                                need, reason = est.sufficient_copies_conservative(b_tot, G_in, DELTA, EPS)
                        cls_proxy_full = est.classify_target(list(sel["proxy"]))
                        K_proxy = list(cls_proxy_full["K_idx"])
                        S_proxy = list(cls_proxy_full["S_idx"])
                        stab_ax_p = list(cls_proxy_full["stab_axis"])
                        stab_w_p = list(cls_proxy_full["stab_want"])
                        scores, probs = fse.lp_family_law_product(rs_hat, proxy_hat_geo, K_proxy, S_proxy, stab_ax_p, stab_w_p, g_float, est.CHANNEL_COMMON, STATE)
                        stats = law_stats(scores, probs, f_true)
                        samp = sample_row_stats_new(scores, probs, N_FROZEN, tid_full, method, f_true, R_tot_float)
                        row.update({"status": "ok", "channel": est.CHANNEL_COMMON,
                                    "k_original": k_orig, "k_proxy": k_proxy0,
                                    "D": sorted(list(D_geo)),
                                    "d_hi": str(d_hi_frac0),
                                    "b_inner": str(b_in), "G_inner": str(G_in), "b_total": str(b_tot),
                                    "certificate_type": "production-certified",
                                    "radius": ensure_finite(R_tot_float, "Rtot"), "radius_exact": str(R_tot_frac),
                                    "sufficient_copies": need, "sufficient_reason": reason,
                                    "law_entries": len(scores),
                                    "expectation": stats["expectation"], "second_moment": stats["second_moment"],
                                    "unclipped_bias": stats["bias"]})
                        row.update(samp)
                        row["certified_eps"] = bool(R_tot_float <= EPS)
                    else:
                        raise ValueError(f"unknown method {method}")
                    row["reps"] = REPS_NEW
                    row["row_time_sec"] = ensure_finite(time.perf_counter() - t_row0, "row_time")
                except Exception as e:
                    import traceback
                    row["status"] = "error"
                    row["reason"] = f"{type(e).__name__}: {e}"[:800]
                    if "law_entries" not in row:
                        row["law_entries"] = None
                    n_fail += 1
                rows.append(row)
                print(f"row {row_id} status={row.get('status')} law={row.get('law_entries')} exceed={row.get('exceed_count')} cover={row.get('cover_count')}", flush=True)
    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    t_all1 = time.perf_counter()
    n_ok = sum(1 for r in rows if r.get("status") == "ok")
    results = {
        "label": "recorded scientific evaluation",
        "design_version": design.get("design_version", ""),
        "design_file_sha256": file_hash,
        "design_canonical_sha256": canon_hash,
        "frozen_source_hashes": frozen_hashes,
        "eps": EPS, "eps_exact": "1/20",
        "delta": DELTA, "delta_exact": "1/3",
        "master_seed": SEED_NEW, "reps": REPS_NEW,
        "N": N_FROZEN, "n_list": list(N_LIST), "targets": list(TIDS),
        "methods": list(METHODS), "state": STATE,
        "scope": "EXACTLY displayed comparison: 8 methods x 2 targets(hom_111/hom_near) x 4 sizes(n=1,2,4,6) x N=4096 x corr07 x eps=0.05 delta=1/3 = 64 rows, 10000 trials each. Predeclared, not selected after results. No refit, no frozen change, genuine joint mixture law. Historical 100-trial 1360-row design distinct.",
        "test_grid_note": "direct reconstruction via estimator routines; frozen run_evaluation test-grid seam NOT used as reportable; production globals unchanged (local caches only)",
        "worst_case_note": "trials are diagnostic examples at tested states; finite coverage cannot infer worst-case guarantee",
        "zero_event_upper": ensure_finite(ZERO_EVENT_UPPER, "zero_up"),
        "zero_event_note": "exact one-sided 95% upper for zero events: 1-0.05**(1/R) with R=10000",
        "wilson_note": "95% Wilson intervals (z=1.96) for exceedance and coverage fractions",
        "python_version": py_ver, "numpy_version": numpy_ver, "scipy_version": scipy_ver,
        "source_hashes": src_hashes,
        "precision": {"bits": BITS, "den": DEN, "note": "exact Fractions for LP/inverse/geometry certificates; numerical Hoeffding for native/shadow; float64 sampling proxies; strict JSON"},
        "distributional_note": "per-rep int.from_bytes(SHA256(master|tid_full|corr07|method|4096|rep).digest()[:16],'big'), each own Generator + one multinomial(4096,p); full 128-bit with no modulus, within-row uniqueness asserted; same law as sum of N iid full-copy scores; sampling AFTER scores frozen",
        "simulation_caveat": "aggregated-law diagnostics, NOT measured runtime of N operational full-copy calls; N and nN distinguished",
        "command": "python code/extended_evaluation.py --run",
        "primary_rows": rows,
        "primary_planned": total_planned,
        "primary_ok": n_ok,
        "primary_errors": n_fail,
        "native_global_ranges": {f"n{n}_{tid}": list(native_global[(n, tid)]) for (n, tid) in native_global},
        "shadow_global_ranges": {f"n{n}_{tid}": list(shadow_global[(n, tid)]) for (n, tid) in shadow_global},
        "lp_cache_details": lp_cache_details,
        "lp_cache_stats": lp_cache_stats,
        "inverse_prep": inv_prep,
        "geometry_prep": geometry_prep,
        "resources": {"total_time_sec": ensure_finite(t_all1 - t_all0, "total"),
                      "tracemalloc_current_bytes": cur, "tracemalloc_peak_bytes": peak,
                      "tracemalloc_note": "Python allocations only; HiGHS native excluded"},
    }
    return results

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Extended evaluation (64 rows, 10000 reps)")
    ap.add_argument("--run", action="store_true", help="execute and write extended_results.json")
    args = ap.parse_args(argv)
    if not args.run:
        print("use --run to execute (writes extended_results.json)")
        return 2
    results = run_extended_evaluation()
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, sort_keys=True, allow_nan=False, default=str)
    saved = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    print(f"wrote {RESULTS_PATH} ({len(saved.get('primary_rows',[]))} rows)")
    n_err = sum(1 for r in saved.get("primary_rows", []) if r.get("status") != "ok")
    print(f"ok={saved.get('primary_ok')} errors={n_err}")
    if n_err > 0:
        print(f"FAILURE: primary errors={n_err} (files saved; exiting nonzero)")
        return 1
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
