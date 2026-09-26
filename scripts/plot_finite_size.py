"""Regenerate the measurement schematic and finite-size comparison from stored results."""
import argparse
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
SELECT_N = [1, 2, 4, 6]
SELECT_N_COPY = 4096
SELECT_TARGETS = ['hom_near', 'hom_111']
SELECT_STATE = 'corr07'
SELECT_METHODS = ['lp_common', 'geometry', 'native', 'shadow']
EXPECTED_FIG2_ROWS = 32
import math

def select_fig2_rows(results: dict) -> list:
    rows = results.get("primary_rows", [])
    sel = [
        r for r in rows
        if r.get("N") == SELECT_N_COPY
        and r.get("state") == SELECT_STATE
        and r.get("n") in SELECT_N
        and r.get("target_id") in SELECT_TARGETS
        and r.get("method") in SELECT_METHODS
    ]
    if len(sel) != EXPECTED_FIG2_ROWS:
        raise ValueError("figure2 selection cardinality %d != %d" % (len(sel), EXPECTED_FIG2_ROWS))
    have = sorted([(r.get("target_id"), r.get("n"), r.get("method")) for r in sel])
    want = sorted([(t, n, m) for t in SELECT_TARGETS for n in SELECT_N for m in SELECT_METHODS])
    if have != want:
        raise ValueError("figure2 selection keys mismatch")
    for r in sel:
        for f in ("radius", "p90", "f_true", "reps"):
            if r.get(f) is None:
                raise ValueError("figure2 row missing %s: %s" % (f, (r.get("target_id"), r.get("n"), r.get("method"))))
        try:
            radius_val = float(r["radius"])
            p90_val = float(r["p90"])
            f_true_val = float(r["f_true"])
        except Exception:
            raise ValueError("figure2 row nonfinite radius/p90/f_true")
        if not (math.isfinite(radius_val) and math.isfinite(p90_val) and math.isfinite(f_true_val)):
            raise ValueError("figure2 row nonfinite radius/p90/f_true")
        if radius_val < 0 or p90_val < 0:
            raise ValueError("figure2 row negative radius/p90")
        if not (0.0 <= f_true_val <= 1.0):
            raise ValueError("figure2 row f_true outside [0,1]")
        if r.get("status") != "ok":
            raise ValueError("figure2 row status not ok")
        if r.get("reps") != 100:
            raise ValueError("figure2 row reps != 100")
    order_m = {m: i for i, m in enumerate(SELECT_METHODS)}
    order_t = {t: i for i, t in enumerate(SELECT_TARGETS)}
    sel = sorted(sel, key=lambda x: (order_t[x["target_id"]], x["n"], order_m[x["method"]]))
    return sel

def make_fig1(path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 9
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.9)
    ax.axis("off")
    boxes = [
        (0.3, 1.2, 1.9, 1.3, "Unknown copies\n\u03C3 (measured)", "#E6EFE9", "#2A7A5B"),
        (2.8, 1.2, 2.2, 1.3, "One local X/Y/Z\nper qubit, full\noutcome string", "#E6EFE9", "#2A7A5B"),
        (5.6, 1.2, 1.9, 1.3, "Classical scores\n+ median", "#F4EEE3", "#8A6A2B"),
        (8.1, 1.2, 1.6, 1.3, "Fidelity\nestimate", "#ECECEC", "#333333"),
    ]
    for (x, y, w, h, label, face, edge) in boxes:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                             facecolor=face, edgecolor=edge, linewidth=1.4)
        ax.add_patch(box)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9, color="black", linespacing=1.3)
    for (x0, x1) in [(2.2, 2.8), (5.0, 5.6), (7.5, 8.1)]:
        ax.annotate("", xy=(x1, 1.85), xytext=(x0, 1.85),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.2, shrinkA=1, shrinkB=1))
    # Known target supplies classical design and score information (dashed, not measured)
    ax.text(6.55, 3.05, "Known product target P\n(design and score info)", ha="center", va="center",
            fontsize=9, color="black",
            bbox=dict(facecolor="#E8EEF4", edgecolor="#1F4A7A", boxstyle="round,pad=0.3", linewidth=1.0))
    ax.annotate("", xy=(6.55, 2.5), xytext=(6.55, 2.78),
                arrowprops=dict(arrowstyle="->", color="#1F4A7A", lw=1.2, linestyle="dashed",
                                shrinkA=1, shrinkB=1))
    ax.text(5.0, 3.55, "Local Pauli fidelity protocol", ha="center", va="center",
            fontsize=11, fontweight="bold", color="black")
    # Main law with proper mathtext subscript and Greek
    ax.text(5.0, 0.55, r"$\log(1+N_{\mathrm{minimax}})=\Theta(1+\lambda^{1/3}),\ \lambda=n(1-a^2)$",
            ha="center", va="center", fontsize=10, color="black",
            bbox=dict(facecolor="white", edgecolor="#1F4A7A", boxstyle="round,pad=0.3", linewidth=1.0))
    ax.text(5.0, 0.15, "Each copy keeps all n outcome bits. Complex arithmetic is classical postprocessing.",
            ha="center", va="center", fontsize=9, color="#333333")
    fig.tight_layout(pad=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)

def make_fig2(sel: list, eps_ref: float, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 9
    plt.rcParams["axes.titlesize"] = 10
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    styles = {
        "lp_common": {"color": "#1F4A7A", "marker": "o", "label": "LP common (cert.)"},
        "geometry": {"color": "#2A7A5B", "marker": "s", "label": "Geometry (cert.)"},
        "native": {"color": "#8A6A2B", "marker": "^", "label": "Native (num.)"},
        "shadow": {"color": "#5A5A5A", "marker": "D", "label": "Shadow (num.)"},
    }
    by_target: dict = {t: {m: {} for m in SELECT_METHODS} for t in SELECT_TARGETS}
    for r in sel:
        by_target[r["target_id"]][r["method"]][int(r["n"])] = (float(r["radius"]), float(r["p90"]))
    xs = list(SELECT_N)
    fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0), sharex=True, sharey="row")
    fig.patch.set_facecolor("white")
    for c, tid in enumerate(SELECT_TARGETS):
        ax_top = axes[0][c]
        for m in SELECT_METHODS:
            ys = [by_target[tid][m][n][0] for n in xs]
            ax_top.plot(xs, ys, marker=styles[m]["marker"], color=styles[m]["color"],
                        label=styles[m]["label"], linewidth=1.6, markersize=5)
        ax_top.set_ylabel("radius" if c == 0 else "")
        ax_top.set_yscale("log")
        ax_top.grid(True, which="both", color="#CCCCCC", linewidth=0.6, alpha=0.5)
        try:
            if eps_ref is not None and float(eps_ref) > 0:
                ax_top.axhline(float(eps_ref), color="black", linestyle="--", linewidth=1.0, alpha=0.8)
        except Exception:
            pass
        ax_bottom = axes[1][c]
        for m in SELECT_METHODS:
            ys = [by_target[tid][m][n][1] for n in xs]
            ax_bottom.plot(xs, ys, marker=styles[m]["marker"], color=styles[m]["color"],
                           linewidth=1.6, markersize=5)
        ax_bottom.set_ylabel("p90 abs error" if c == 0 else "")
        ax_bottom.set_xlabel("n")
        ax_bottom.set_xticks(xs)
        ax_bottom.set_xticklabels([str(x) for x in xs])
        ax_bottom.grid(True, color="#CCCCCC", linewidth=0.6, alpha=0.5)
        try:
            if eps_ref is not None and float(eps_ref) > 0:
                ax_bottom.axhline(float(eps_ref), color="black", linestyle="--", linewidth=1.0, alpha=0.8)
        except Exception:
            pass
    axes[0][0].set_title("Near-axis target", fontsize=10, pad=8)
    axes[0][1].set_title("Equal-coordinate target", fontsize=10, pad=8)
    axes[1][0].set_title("Near-axis target", fontsize=10, pad=8)
    axes[1][1].set_title("Equal-coordinate target", fontsize=10, pad=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98),
               ncol=4, frameon=True, facecolor="white", edgecolor="#333333")
    fig.tight_layout(rect=(0, 0.02, 1, 0.88), pad=0.7)
    fig.subplots_adjust(top=0.85)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=300, bbox_inches="tight")
    plt.close(fig)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output-dir', type=Path, default=ROOT/'figures')
    args = ap.parse_args()
    rows = select_fig2_rows(json.loads((ROOT/'results/finite_size_results.json').read_text(encoding='utf-8')))
    saved = json.loads((ROOT/'figures/figure_data.json').read_text(encoding='utf-8'))['figure2']['rows']
    for row in rows:
        ref = next(r for r in saved if (r['n'],r['method'],r['target_id']) == (row['n'],row['method'],row['target_id']))
        assert ref['p90_raw'] == row['p90'] and ref['radius_raw'] == row['radius']
    make_fig1(args.output_dir/'measurement_scheme.png')
    make_fig2(rows, 0.05, args.output_dir/'finite_size_comparison.png')
    print('Validated 32 plot rows and rendered two figures.')

if __name__ == '__main__': main()
