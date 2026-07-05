#!/usr/bin/env python3
# Solver runtime comparison bar chart.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, os

SURFACE = "#ffffff"
INK = "#1a1a1a"; INK2 = "#5a6068"; GRID = "#e2e6ea"
meshes = ["990k", "4.9M", "15M", "35.6M"]
series = [
    ("brae (Blackwell GPU)",       [22, 108, 340, 877],   "#76B900"),
    ("OpenFOAM (20 Grace cores)",  [16, 107, 354, 990],   "#3E86C9"),
    ("OpenFOAM + AMGX (same GPU)", [80, 573, 1696, 4111], "#CE7E2A"),
    ("OpenFOAM + PETSc-GPU (same GPU)", [89, 602, 1790, None], "#9A63C9"),
    ("SPUMA (OpenFOAM-GPU port, same GPU)", [92, 623, 1359, None], "#D1495B"),
]

plt.rcParams.update({
    "font.family": "DejaVu Sans", "text.color": INK,
    "axes.edgecolor": GRID, "xtick.color": INK, "ytick.color": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})
fig, ax = plt.subplots(figsize=(11.5, 6.6), dpi=200)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)

x = np.arange(len(meshes)); n = len(series); w = 0.15; off = (np.arange(n) - (n-1)/2) * (w + 0.010)
for si, (name, vals, col) in enumerate(series):
    for gi, v in enumerate(vals):
        xp = x[gi] + off[si]
        if v is None:
            ax.text(xp, 12.5, "n/a", ha="center", va="bottom", color=INK2, fontsize=8)
            continue
        ax.bar(xp, v, width=w, bottom=10, color=col, linewidth=0, zorder=3,
               label=name if gi == 0 else None)
        ax.text(xp, v*1.04, f"{v}", ha="center", va="bottom", color=INK,
                fontsize=8.2, fontweight="bold" if si == 0 else "normal", zorder=4)

ax.set_yscale("log")
ax.set_ylim(10, 7000)
ax.set_yticks([10, 100, 1000])
ax.set_yticklabels(["10", "100", "1000"])
ax.set_xticks(x); ax.set_xticklabels(meshes, color=INK, fontsize=11)
ax.tick_params(axis="x", length=0)
ax.set_ylabel("Total wall time for 100 SIMPLE iterations  (seconds, log scale)", color=INK2, fontsize=10.5)
ax.set_xlabel("Mesh size (cells)", color=INK2, fontsize=10.5, labelpad=8)

for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
ax.set_axisbelow(True)

# title and subtitle
fig.text(0.065, 0.955, "Solver runtime comparison", fontsize=20, fontweight="bold", color=INK, va="top")
fig.text(0.065, 0.885, "Lower is better.", fontsize=11.5, fontweight="bold", color=INK, va="top")

ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.0), ncol=2, frameon=False,
          fontsize=9.3, labelcolor=INK, handlelength=1.1, handleheight=1.1,
          columnspacing=1.6, borderaxespad=0.4)

fig.text(0.065, 0.02,
         "brae is ~4-5x faster than every GPU alternative here (AMGX, PETSc, SPUMA) on the same GPU.",
         fontsize=8.6, color=INK2, va="bottom")

plt.subplots_adjust(left=0.065, right=0.975, top=0.83, bottom=0.13)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "solver_runtime_comparison.png")
fig.savefig(out, facecolor=SURFACE)
print("wrote", out)
