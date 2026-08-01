#!/usr/bin/env python3
"""
Replay a solver trace as a video: the mesh, the matrix, and the multigrid hierarchy, animated
one internal step at a time.

Everything it draws comes from the CSVs in <case>/trace/ -- nothing is computed here. Point it
at a different case directory, or hand-edit the CSVs, and the video follows.

    python3 replay.py demo/amgpcg/case20 --solver brae      -o brae.mp4
    python3 replay.py demo/amgpcg/case20 --compare          -o compare.mp4
    python3 replay.py demo/amgpcg/case20 --solver brae --gif -o brae.gif

Panels, left to right:
    mesh      the 4x5 cells, filled by the value the current step is acting on, with the
              aggregate boundaries of the ACTIVE grid drawn on top -- so restriction visibly
              merges cells and prolongation visibly splits them again
    matrix    the operator of the ACTIVE grid as a heatmap; it shrinks 20x20 -> 10x10 -> 5x5
              as the cycle descends and grows back on the way up
    cycle     the V-cycle schematic, current position marked
    residual  all three solvers' convergence, with a marker at the current iteration

Requires only numpy + matplotlib. MP4 needs ffmpeg; if it is missing, pass --gif.
"""

import argparse
import collections
import csv
import os
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle


SOLVER_LABEL = {
    "of_gamg": "OpenFOAM GAMG",
    "amgx": "AMGX  (PCG + AGGREGATION)",
    "brae": "brae  AMG-PCG",
    "brae_cheb": "brae AMG-PCG (Chebyshev)",
    "brae_gs": "brae AMG-PCG (multicolor GS)",
    "brae_soc": "brae AMG-PCG (strength filter)",
}

SOLVER_COLOUR = {
    "of_gamg": "#e4572e",
    "amgx": "#76b900",
    "brae": "#3f8efc",
}

PHASE_COLOUR = {
    "krylov_start": "#8d99ae",
    "presmooth": "#f4a259",
    "residual": "#8d99ae",
    "restrict": "#e4572e",
    "coarse_solve": "#9b5de5",
    "prolong": "#3f8efc",
    "postsmooth": "#f4a259",
    "krylov_update": "#43aa8b",
}

PHASE_TEXT = {
    "krylov_start": "iteration begins",
    "presmooth": "pre-smoothing sweep",
    "residual": "compute residual  r = b - A x",
    "restrict": "RESTRICT  r down to the coarse grid",
    "coarse_solve": "solve the coarsest grid",
    "prolong": "PROLONG  the correction back up",
    "postsmooth": "post-smoothing sweep",
    "krylov_update": "Krylov update  x += alpha p",
}


# ---------------------------------------------------------------------------------------- #
# Loading


def load_mesh(trace):
    cells = {}

    with open(os.path.join(trace, "mesh.csv")) as handle:
        for row in csv.DictReader(handle):
            cells[int(row["cell"])] = (float(row["x"]), float(row["y"]))

    n = len(cells)
    xs = np.array([cells[c][0] for c in range(n)])
    ys = np.array([cells[c][1] for c in range(n)])

    # Recover the structured (i, j) index of the block mesh from the unique coordinates.
    ux = np.unique(np.round(xs, 9))
    uy = np.unique(np.round(ys, 9))
    col = np.searchsorted(ux, np.round(xs, 9))
    row = np.searchsorted(uy, np.round(ys, 9))

    return n, col, row, len(ux), len(uy)


def load_steps(trace):
    steps = collections.defaultdict(list)

    with open(os.path.join(trace, "steps.csv")) as handle:
        for row in csv.DictReader(handle):
            steps[row["solver"]].append({
                "step": int(row["step"]),
                "iter": int(row["iter"]),
                "phase": row["phase"],
                "grid": int(row["grid"]),
                "ngrid": int(row["ngrid"]),
                "detail": row["detail"],
                "resnorm": float(row["resnorm"]) if row["resnorm"] else float("nan"),
            })

    return steps


def load_state(trace):
    state = collections.defaultdict(dict)

    with open(os.path.join(trace, "state.csv")) as handle:
        for row in csv.DictReader(handle):
            key = (row["solver"], int(row["step"]))
            state[key][int(row["cell"])] = (float(row["x"]), float(row["r"]))

    packed = {}

    for key, cells in state.items():
        m = len(cells)
        packed[key] = (
            np.array([cells[c][0] for c in range(m)]),
            np.array([cells[c][1] for c in range(m)]),
        )

    return packed


def load_grids(trace):
    entries = collections.defaultdict(list)

    with open(os.path.join(trace, "grids.csv")) as handle:
        for row in csv.DictReader(handle):
            entries[(row["solver"], int(row["grid"]))].append(
                (int(row["row"]), int(row["col"]), float(row["value"]))
            )

    grids = {}

    for key, triples in entries.items():
        size = max(max(r, c) for r, c, _ in triples) + 1
        A = np.zeros((size, size))

        for r, c, v in triples:
            A[r, c] = v

        grids[key] = A

    return grids


def load_aggregates(trace):
    """(solver, grid) -> array mapping each finest cell to its cell index on that grid."""
    entries = collections.defaultdict(dict)

    with open(os.path.join(trace, "aggregates.csv")) as handle:
        for row in csv.DictReader(handle):
            entries[(row["solver"], int(row["grid"]))][int(row["finest_cell"])] = \
                int(row["grid_cell"])

    return {
        key: np.array([value[c] for c in range(len(value))], dtype=int)
        for key, value in entries.items()
    }


# ---------------------------------------------------------------------------------------- #
# Drawing


def diverging_norm(values):
    limit = np.abs(values).max()

    if limit <= 0 or not np.isfinite(limit):
        return Normalize(vmin=-1, vmax=1)

    return TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)


def format_value(value, scale):
    """Readable in-cell label whatever the magnitude: fixed point near 1, mantissa when tiny."""
    if scale >= 0.01:
        return f"{value:.2f}"

    if scale <= 0 or not np.isfinite(scale):
        return "0"

    exponent = int(np.floor(np.log10(scale)))
    return f"{value / 10.0 ** exponent:.1f}"


def draw_mesh_panel(ax, col, row, ncol, nrow, values, agg, title, cmap, norm):
    ax.clear()
    scale = float(np.abs(values).max())
    ax.set_title(title, fontsize=11, pad=8)
    ax.set_xlim(-0.5, ncol - 0.5)
    ax.set_ylim(-0.5, nrow - 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    for cell in range(len(col)):
        ax.add_patch(Rectangle(
            (col[cell] - 0.5, row[cell] - 0.5), 1, 1,
            facecolor=cmap(norm(values[cell])), edgecolor="#00000018", linewidth=0.5,
        ))

        if len(col) <= 40:
            ax.text(col[cell], row[cell], format_value(values[cell], scale),
                    ha="center", va="center", fontsize=6.5, color="#1b1b1b")

    # Aggregate boundaries of the active grid: draw a thick edge wherever two adjacent cells
    # belong to different aggregates. This is what makes restriction visible.
    lookup = {(col[c], row[c]): c for c in range(len(col))}

    for cell in range(len(col)):
        i, j = col[cell], row[cell]

        for di, dj, x0, y0, x1, y1 in (
            (1, 0, i + 0.5, j - 0.5, i + 0.5, j + 0.5),
            (0, 1, i - 0.5, j + 0.5, i + 0.5, j + 0.5),
        ):
            other = lookup.get((i + di, j + dj))

            if other is None:
                continue

            if agg[cell] != agg[other]:
                ax.plot([x0, x1], [y0, y1], color="#111111", linewidth=2.0,
                        solid_capstyle="round", zorder=5)

    if 0 < scale < 0.01:
        exponent = int(np.floor(np.log10(scale)))
        ax.set_xlabel(f"cell labels in units of 1e{exponent}", fontsize=8, color="#55606b")
    else:
        ax.set_xlabel(f"max |value| = {scale:.3g}", fontsize=8, color="#55606b")

    for spine in ax.spines.values():
        spine.set_edgecolor("#cccccc")


def draw_matrix_panel(ax, A, title):
    ax.clear()
    ax.set_title(title, fontsize=11, pad=8)

    limit = np.abs(A).max() or 1.0
    ax.imshow(A, cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])

    size = A.shape[0]

    if size <= 24:
        for i in range(size):
            for j in range(size):
                if A[i, j] != 0.0:
                    ax.text(j, i, f"{A[i, j]:.0f}", ha="center", va="center",
                            fontsize=5.5, color="#111111")

    ax.set_xlabel(f"{size} x {size},  {int(np.count_nonzero(A))} nonzeros", fontsize=8)


def draw_cycle_panel(ax, n_levels, grid, phase, ngrid):
    ax.clear()
    ax.set_title("V-cycle position", fontsize=11, pad=8)
    ax.set_xlim(-0.6, 2 * n_levels - 1.4)
    ax.set_ylim(-n_levels + 0.4, 0.7)
    ax.axis("off")

    down = [(k, -k) for k in range(n_levels)]
    up = [(2 * n_levels - 2 - k, -k) for k in range(n_levels)]
    path = down + up[::-1]

    ax.plot([p[0] for p in path], [p[1] for p in path],
            color="#cfd6dd", linewidth=2.5, zorder=1)

    descending = phase in ("presmooth", "residual", "restrict", "krylov_start")
    active = (grid, -grid) if descending else (2 * n_levels - 2 - grid, -grid)

    for px, py in path:
        ax.scatter([px], [py], s=90, color="#ffffff", edgecolor="#9aa5b1",
                   linewidth=1.5, zorder=2)

    ax.scatter([active[0]], [active[1]], s=210,
               color=PHASE_COLOUR.get(phase, "#333333"),
               edgecolor="#111111", linewidth=1.5, zorder=3)

    for k in range(n_levels):
        ax.text(-0.45, -k, f"g{k}", ha="right", va="center", fontsize=8, color="#55606b")

    ax.text(active[0], active[1] + 0.32, f"{ngrid} cells", ha="center", va="bottom",
            fontsize=8.5, color="#111111", zorder=4)


def draw_residual_panel(ax, histories, solver, iteration):
    ax.clear()
    ax.set_title("convergence  |b - A x| / |b|", fontsize=11, pad=8)
    ax.set_yscale("log")
    ax.set_xlabel("cycle", fontsize=9)
    ax.grid(True, which="both", alpha=0.15, linewidth=0.6)

    for name, history in histories.items():
        ax.plot(range(len(history)), history,
                color=SOLVER_COLOUR.get(name, "#888888"),
                linewidth=2.2 if name == solver else 1.2,
                alpha=1.0 if name == solver else 0.45,
                label=SOLVER_LABEL.get(name, name))

    history = histories.get(solver)

    if history is not None and iteration < len(history):
        ax.scatter([iteration], [history[iteration]], s=70, zorder=5,
                   color=SOLVER_COLOUR.get(solver, "#333333"), edgecolor="#111111")

    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.9)
    ax.set_ylim(1e-14, 5.0)


# ---------------------------------------------------------------------------------------- #


def build_histories(trace):
    """Per-solver convergence, straight from compare.csv if present, else from steps.csv."""
    path = os.path.join(trace, "compare.csv")
    histories = collections.defaultdict(dict)

    if os.path.exists(path):
        with open(path) as handle:
            for row in csv.DictReader(handle):
                histories[row["solver"]][int(row["cycle"])] = float(row["resnorm"])
    else:
        with open(os.path.join(trace, "steps.csv")) as handle:
            for row in csv.DictReader(handle):
                if row["phase"] == "krylov_update" and row["resnorm"]:
                    histories[row["solver"]][int(row["iter"])] = float(row["resnorm"])

        for entry in histories.values():
            entry.setdefault(0, 1.0)

    return {
        name: [entry[k] for k in sorted(entry)]
        for name, entry in histories.items()
        if name in SOLVER_COLOUR
    }


def render(case, solver, out_path, fps, dpi, use_gif, field):
    trace = os.path.join(case, "trace")
    n, col, row, ncol, nrow = load_mesh(trace)
    steps = load_steps(trace)
    state = load_state(trace)
    grids = load_grids(trace)
    aggregates = load_aggregates(trace)
    histories = build_histories(trace)

    if solver not in steps:
        print(f"no trace for solver '{solver}'; have: {sorted(steps)}", file=sys.stderr)
        return 1

    frames = steps[solver]
    n_levels = max(f["grid"] for f in frames) + 1

    # One colour scale PER GRID, held fixed across the whole run. A single global scale would
    # wash the coarse levels out: what lives on g0 is the solution (O(1)) while what lives on
    # g1+ is a correction to it, orders of magnitude smaller. Per-grid scaling keeps both
    # readable, and holding each fixed across the run means brightness changes still mean
    # something within a level.
    norms = {}

    for f in frames:
        values = state[(solver, f["step"])][0 if field == "x" else 1]
        limit = np.abs(values).max()
        norms[f["grid"]] = max(norms.get(f["grid"], 0.0), limit)

    norms = {grid: diverging_norm(np.array([-limit, limit]))
             for grid, limit in norms.items()}

    cmap = LinearSegmentedColormap.from_list(
        "brae", ["#2166ac", "#f7f7f7", "#b2182b"]
    )

    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    fig.patch.set_facecolor("#ffffff")

    grid_spec = fig.add_gridspec(
        2, 3,
        width_ratios=[1.0, 1.0, 1.15], height_ratios=[1.0, 0.85],
        left=0.035, right=0.975, top=0.88, bottom=0.05, wspace=0.16, hspace=0.24,
    )

    ax_mesh = fig.add_subplot(grid_spec[:, 0])
    ax_matrix = fig.add_subplot(grid_spec[:, 1])
    ax_cycle = fig.add_subplot(grid_spec[0, 2])
    ax_res = fig.add_subplot(grid_spec[1, 2])

    title = fig.text(0.035, 0.985, "", fontsize=19, va="top", fontweight="bold")
    subtitle = fig.text(0.035, 0.945, "", fontsize=12.5, va="top", color="#3d4852")
    banner = fig.text(0.60, 0.985, "", fontsize=11.5, va="top", color="#ffffff",
                      bbox=dict(boxstyle="round,pad=0.45", facecolor="#333333", edgecolor="none"))

    def update(index):
        frame = frames[index]
        step = frame["step"]
        grid = frame["grid"]
        phase = frame["phase"]

        x_vec, r_vec = state[(solver, step)]
        values = x_vec if field == "x" else r_vec

        # Spread the active grid's vector back over the finest cells for the mesh panel.
        agg = aggregates[(solver, grid)]
        painted = values[agg]

        draw_mesh_panel(
            ax_mesh, col, row, ncol, nrow, painted, agg,
            f"mesh, grid g{grid}  ({frame['ngrid']} cells)   field = {field}",
            cmap, norms[grid],
        )

        A_g = grids.get((solver, grid))

        if A_g is not None:
            draw_matrix_panel(ax_matrix, A_g, f"operator  A{'' if grid == 0 else f'_{grid}'}")

        draw_cycle_panel(ax_cycle, n_levels, grid, phase, frame["ngrid"])
        draw_residual_panel(ax_res, histories, solver, frame["iter"])

        title.set_text(SOLVER_LABEL.get(solver, solver))
        subtitle.set_text(
            f"cycle {frame['iter']}     "
            f"residual {frame['resnorm']:.3e}     "
            f"frame {index + 1}/{len(frames)}"
        )
        phase_text = PHASE_TEXT.get(phase, phase)
        detail = frame["detail"]
        banner.set_text(phase_text if detail in ("", phase_text) else f"{phase_text}  --  {detail}")
        banner.get_bbox_patch().set_facecolor(PHASE_COLOUR.get(phase, "#333333"))

        return []

    anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps,
                                   blit=False)

    if use_gif:
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    else:
        try:
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass

        anim.save(out_path, writer=animation.FFMpegWriter(
            fps=fps, bitrate=6000, codec="libx264",
            extra_args=["-pix_fmt", "yuv420p"],
        ))

    plt.close(fig)
    print(f"wrote {out_path}  ({len(frames)} frames @ {fps} fps "
          f"= {len(frames) / fps:.1f} s)")
    return 0


def render_compare(case, out_path, fps, dpi, use_gif, field):
    """Three solvers side by side, advancing through their own steps in lockstep."""
    trace = os.path.join(case, "trace")
    n, col, row, ncol, nrow = load_mesh(trace)
    steps = load_steps(trace)
    state = load_state(trace)
    aggregates = load_aggregates(trace)
    histories = build_histories(trace)

    solvers = [s for s in ("of_gamg", "amgx", "brae") if s in steps]
    n_frames = max(len(steps[s]) for s in solvers)

    all_values = np.concatenate([
        state[(s, f["step"])][0 if field == "x" else 1]
        for s in solvers for f in steps[s]
    ])
    norm = diverging_norm(all_values)
    cmap = LinearSegmentedColormap.from_list("brae", ["#2166ac", "#f7f7f7", "#b2182b"])

    fig = plt.figure(figsize=(16, 9), dpi=dpi)
    grid_spec = fig.add_gridspec(
        2, 3, height_ratios=[1.25, 1.0],
        left=0.04, right=0.97, top=0.82, bottom=0.07, wspace=0.12, hspace=0.35,
    )

    mesh_axes = [fig.add_subplot(grid_spec[0, i]) for i in range(len(solvers))]
    ax_res = fig.add_subplot(grid_spec[1, :])

    title = fig.text(0.04, 0.995, "Same 20 cells, same matrix, three solvers",
                     fontsize=20, va="top", fontweight="bold")
    subtitle = fig.text(0.04, 0.94, "", fontsize=12, va="top", color="#3d4852")

    def update(index):
        for ax, solver in zip(mesh_axes, solvers):
            frames = steps[solver]
            frame = frames[min(index, len(frames) - 1)]
            x_vec, r_vec = state[(solver, frame["step"])]
            values = x_vec if field == "x" else r_vec
            agg = aggregates[(solver, frame["grid"])]

            draw_mesh_panel(
                ax, col, row, ncol, nrow, values[agg], agg,
                f"{SOLVER_LABEL.get(solver, solver)}\n"
                f"cycle {frame['iter']}  |  g{frame['grid']} ({frame['ngrid']} cells)  |  "
                f"{PHASE_TEXT.get(frame['phase'], frame['phase'])}",
                cmap, norm,
            )

        draw_residual_panel(ax_res, histories, "brae", steps["brae"][
            min(index, len(steps["brae"]) - 1)]["iter"])

        subtitle.set_text(f"frame {index + 1}/{n_frames}   -- "
                          f"each solver stepping through its own cycle internals")
        return []

    anim = animation.FuncAnimation(fig, update, frames=n_frames, interval=1000 / fps,
                                   blit=False)

    if use_gif:
        anim.save(out_path, writer=animation.PillowWriter(fps=fps))
    else:
        try:
            import imageio_ffmpeg
            matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass

        anim.save(out_path, writer=animation.FFMpegWriter(
            fps=fps, bitrate=6000, codec="libx264",
            extra_args=["-pix_fmt", "yuv420p"],
        ))

    plt.close(fig)
    print(f"wrote {out_path}  ({n_frames} frames @ {fps} fps = {n_frames / fps:.1f} s)")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", help="case directory containing trace/")
    parser.add_argument("--solver", default="brae", help="of_gamg | amgx | brae")
    parser.add_argument("--compare", action="store_true", help="all three side by side")
    parser.add_argument("--field", default="x", choices=["x", "r"],
                        help="paint the solution (x) or the residual (r)")
    parser.add_argument("-o", "--output", default=None)
    parser.add_argument("--fps", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=120)
    parser.add_argument("--gif", action="store_true", help="write a GIF instead of MP4")
    args = parser.parse_args()

    suffix = ".gif" if args.gif else ".mp4"
    default_name = "compare" if args.compare else args.solver
    out_path = args.output or os.path.join(args.case, f"{default_name}{suffix}")

    if args.compare:
        return render_compare(args.case, out_path, args.fps, args.dpi, args.gif, args.field)

    return render(args.case, args.solver, out_path, args.fps, args.dpi, args.gif, args.field)


if __name__ == "__main__":
    sys.exit(main())
