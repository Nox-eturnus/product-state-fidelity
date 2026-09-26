"""Plot extended Monte Carlo diagnostics from recorded results."""

import json, sys
from pathlib import Path
import argparse
ROOT = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser(description='Regenerate the extended comparison from all stored rows.')
ap.add_argument('--output-dir', type=Path, default=ROOT/'figures')
args = ap.parse_args()
out_path = args.output_dir/'extended_comparison.png' 
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
payload = json.loads((ROOT/'results/extended_results.json').read_text(encoding='utf-8'))['primary_rows']
assert len(payload) == 64
saved = json.loads((ROOT/'figures/extended_figure_data.json').read_text(encoding='utf-8'))['selected_rows']
for ref in saved:
    row = next(r for r in payload if r['row_id'] == ref['row_id'])
    assert row['p90'] == ref['p90'] and row['radius'] == ref['radius']
by = {t: {m: {} for m in ["lp_common","geometry","native","shadow"]} for t in ["hom_near","hom_111"]}
for r in payload:
    if r["method"] not in ("lp_common","geometry","native","shadow"):
        continue
    if r["target_id"] not in ("hom_near","hom_111"):
        continue
    if r["radius"] is None:
        continue
    by[r["target_id"]][r["method"]][int(r["n"])] = (float(r["radius"]), float(r["p90"]))
xs = [1,2,4,6]
styles = {
    "lp_common": {"color": "#1F4A7A", "marker": "o", "label": "LP common (cert.)", "ms": 5, "fill": "full", "ls": "-", "lw": 1.6},
    "geometry": {"color": "#2A7A5B", "marker": "s", "label": "Geometry (cert.)", "ms": 8, "fill": "none", "ls": "--", "lw": 1.6},
    "native": {"color": "#8A6A2B", "marker": "^", "label": "Native (num.)", "ms": 6, "fill": "full", "ls": ":", "lw": 1.6},
    "shadow": {"color": "#5A5A5A", "marker": "D", "label": "Shadow (num.)", "ms": 6, "fill": "full", "ls": "-.", "lw": 1.6},
}
order = ["lp_common","native","shadow","geometry"]
fig, axes = plt.subplots(2, 2, figsize=(6.5, 5.0), sharex=True, sharey="row")
fig.patch.set_facecolor("white")
for c, tid in enumerate(["hom_near","hom_111"]):
    ax_top = axes[0][c]
    for m in order:
        ys = [by[tid][m][n][0] for n in xs]
        kw = dict(marker=styles[m]["marker"], color=styles[m]["color"], label=styles[m]["label"], linewidth=styles[m]["lw"], markersize=styles[m]["ms"], linestyle=styles[m]["ls"])
        if styles[m]["fill"] == "none":
            kw["markerfacecolor"] = "none"
            kw["markeredgewidth"] = 2.0
        ax_top.plot(xs, ys, **kw)
    ax_top.set_ylabel("radius" if c==0 else "")
    ax_top.set_yscale("log")
    ax_top.grid(True, which="both", color="#CCCCCC", linewidth=0.6, alpha=0.5)
    ax_top.axhline(0.05, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
    ax_bottom = axes[1][c]
    for m in order:
        ys = [by[tid][m][n][1] for n in xs]
        kw = dict(marker=styles[m]["marker"], color=styles[m]["color"], linewidth=styles[m]["lw"], markersize=styles[m]["ms"], linestyle=styles[m]["ls"])
        if styles[m]["fill"] == "none":
            kw["markerfacecolor"] = "none"
            kw["markeredgewidth"] = 2.0
        ax_bottom.plot(xs, ys, **kw)
    ax_bottom.set_ylabel("p90 abs error" if c==0 else "")
    ax_bottom.set_xlabel("n")
    ax_bottom.set_xticks(xs)
    ax_bottom.set_xticklabels([str(x) for x in xs])
    ax_bottom.grid(True, color="#CCCCCC", linewidth=0.6, alpha=0.5)
    ax_bottom.axhline(0.05, color="black", linestyle="--", linewidth=1.0, alpha=0.8)
axes[0][0].set_title("Near-axis target", fontsize=10, pad=8)
axes[0][1].set_title("Equal-coordinate target", fontsize=10, pad=8)
axes[1][0].set_title("Near-axis target", fontsize=10, pad=8)
axes[1][1].set_title("Equal-coordinate target", fontsize=10, pad=8)
handles, labels = axes[0][0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.98), ncol=4, frameon=True, facecolor="white", edgecolor="#333333")
fig.tight_layout(rect=(0, 0.02, 1, 0.88), pad=0.7)
fig.subplots_adjust(top=0.85, wspace=0.25, hspace=0.35)
out_path.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(str(out_path), dpi=300, bbox_inches="tight")
print("wrote %s" % out_path)
