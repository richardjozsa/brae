#!/usr/bin/env python3
"""
Expand each solver's iteration into its INTERNAL steps, so the replay can show the process
being applied to the matrix rather than one number per iteration.

Emits two tidy CSVs into <case>/trace/:

    steps.csv    one row per animation frame
                 solver,step,iter,phase,grid,ngrid,detail,resnorm
    state.csv    one row per (frame, cell) -- the vector the frame is acting on
                 solver,step,grid,cell,x,r

`phase` is one of:
    krylov_start   the CG iteration begins: r is the current fine residual
    presmooth      one pre-smoothing sweep on `grid`
    residual       r = b - A x computed on `grid`
    restrict       r restricted from `grid` down to `grid+1` (b of the coarse level)
    coarse_solve   the coarsest grid is solved
    prolong        the coarse correction is interpolated from `grid+1` up to `grid`
    postsmooth     one post-smoothing sweep on `grid`
    krylov_update  the CG update x += alpha p, r -= alpha A p

HOW REAL IS THIS
    The mesh, the fine matrix A, the right-hand side b, and the agglomeration hierarchies for
    OpenFOAM GAMG and brae are all dumped from the real codes (dumpAmgTrace, trace_brae).
    The cycle bodies here are reference implementations of the three documented algorithms,
    which is what makes per-sweep state observable at all -- neither OpenFOAM nor AMGX exposes
    its intra-cycle vectors. Each one is checked against the real solver's own per-iteration
    trace (*_cycles.csv) and the max deviation is printed; see the "validation" block in the
    output. AMGX's aggregation is not retrievable through its C API, so its hierarchy is the
    documented SIZE_2 greedy pairwise selector rebuilt here -- flagged `reconstructed` in the
    output and in README.md. OpenFOAM's and brae's hierarchies are the genuine dumps.

Usage:  python3 make_steps.py demo/amgpcg/case20
"""

import argparse
import collections
import csv
import os
import sys

import numpy as np


OMEGA_BRAE = 0.8      # device_amg.cu:37   damped-Jacobi factor
NPRE_BRAE = 1         # device_amg.cu:38
NPOST_BRAE = 1        # device_amg.cu:38

OMEGA_AMGX = 0.75     # bench/amgxFoam/amgxSolver.C:210  relaxation_factor
NPRE_AMGX = 2         # amg:presweeps
NPOST_AMGX = 2        # amg:postsweeps

NPRE_OF = 0           # system/fvSolution  nPreSweeps
NPOST_OF = 2          # system/fvSolution  nPostSweeps
NFINEST_OF = 2        # system/fvSolution  nFinestSweeps


# ---------------------------------------------------------------------------------------- #
# Reading the dumped system


def read_ldu(path):
    """The LDU triple exactly as OpenFOAM and brae store it."""
    diag = {}
    upper = {}
    lower = {}
    owner = {}
    neighbour = {}

    with open(path) as handle:
        for row in csv.DictReader(handle):
            index = int(row["index"])
            value = float(row["value"])

            if row["kind"] == "diag":
                diag[index] = value
            else:
                owner[index] = int(row["owner"])
                neighbour[index] = int(row["neighbour"])

                if row["kind"] == "upper":
                    upper[index] = value
                else:
                    lower[index] = value

    n = len(diag)
    nf = len(owner)

    return (
        np.array([diag[c] for c in range(n)]),
        np.array([upper[f] for f in range(nf)]),
        np.array([lower[f] for f in range(nf)]),
        np.array([owner[f] for f in range(nf)], dtype=int),
        np.array([neighbour[f] for f in range(nf)], dtype=int),
    )


def dense(diag, upper, lower, owner, neighbour):
    n = len(diag)
    A = np.diag(diag).astype(float)

    for f in range(len(owner)):
        A[owner[f], neighbour[f]] = upper[f]
        A[neighbour[f], owner[f]] = lower[f]

    return A


def read_rhs(path, n):
    b = np.zeros(n)

    with open(path) as handle:
        for row in csv.DictReader(handle):
            b[int(row["cell"])] = float(row["b"])

    return b


def read_hierarchy(path):
    """level -> restriction map (fine cell of that level -> coarse cell)."""
    maps = collections.defaultdict(dict)

    with open(path) as handle:
        for row in csv.DictReader(handle):
            fine = int(row["fine_cell"])

            if fine < 0:
                continue

            maps[int(row["level"])][fine] = int(row["coarse_cell"])

    out = []

    for level in sorted(maps):
        entry = maps[level]
        out.append(np.array([entry[c] for c in range(len(entry))], dtype=int))

    return out


# ---------------------------------------------------------------------------------------- #
# Hierarchy construction (only used for AMGX, whose aggregation the C API does not expose)


def pairwise_aggregate(owner, neighbour, weights, n):
    """
    The greedy pairwise selector both OpenFOAM's faceAreaPair and AMGX's SIZE_2 implement:
    sweep cells in order, merge each unmapped cell with its unmapped neighbour across the
    strongest incident face, attach leftovers to an existing aggregate.
    """
    incident = collections.defaultdict(list)

    for f in range(len(owner)):
        incident[owner[f]].append(f)
        incident[neighbour[f]].append(f)

    cell_map = np.full(n, -1, dtype=int)
    n_coarse = 0

    for c in range(n):
        if cell_map[c] >= 0:
            continue

        best_face = -1
        best_weight = -1.0

        for f in incident[c]:
            other = neighbour[f] if owner[f] == c else owner[f]

            if cell_map[other] < 0 and weights[f] > best_weight:
                best_weight = weights[f]
                best_face = f

        if best_face >= 0:
            other = neighbour[best_face] if owner[best_face] == c else owner[best_face]
            cell_map[c] = n_coarse
            cell_map[other] = n_coarse
            n_coarse += 1
            continue

        attached = -1
        best_weight = -1.0

        for f in incident[c]:
            other = neighbour[f] if owner[f] == c else owner[f]

            if cell_map[other] >= 0 and weights[f] > best_weight:
                best_weight = weights[f]
                attached = cell_map[other]

        if attached >= 0:
            cell_map[c] = attached
        else:
            cell_map[c] = n_coarse
            n_coarse += 1

    return cell_map, n_coarse


def galerkin(A, cell_map, n_coarse):
    """
    A_c = P^T A P for the 0/1 injection prolongator P. This is algebraically what brae's
    LDU face scatter (device_amg.cu:432-470) and OpenFOAM's GAMG coarsen() compute; writing
    it as a dense triple product here keeps the reference implementation short.
    """
    P = np.zeros((A.shape[0], n_coarse))
    P[np.arange(A.shape[0]), cell_map] = 1.0

    return P.T @ A @ P, P


def build_levels(A, owner, neighbour, weights, maps=None, target=4):
    """
    Returns [(A_g, map_g, P_g)] per level. If `maps` is given (a real dumped hierarchy) it is
    used verbatim; otherwise the pairwise selector builds one.
    """
    levels = []
    A_g = A
    owner_g = owner
    neighbour_g = neighbour
    weights_g = weights
    level = 0

    while True:
        n = A_g.shape[0]

        if maps is not None:
            if level >= len(maps):
                break

            cell_map = maps[level]
            n_coarse = int(cell_map.max()) + 1
        else:
            if n <= target:
                break

            cell_map, n_coarse = pairwise_aggregate(owner_g, neighbour_g, weights_g, n)

            if n_coarse >= n:
                break

        A_c, P = galerkin(A_g, cell_map, n_coarse)
        levels.append((A_g, cell_map, P))

        # coarse-level addressing for the next round of aggregation
        pairs = {}

        for f in range(len(owner_g)):
            co = cell_map[owner_g[f]]
            cn = cell_map[neighbour_g[f]]

            if co == cn:
                continue

            key = (min(co, cn), max(co, cn))
            pairs[key] = pairs.get(key, 0.0) + weights_g[f]

        if not pairs:
            break

        ordered = sorted(pairs)
        owner_g = np.array([k[0] for k in ordered], dtype=int)
        neighbour_g = np.array([k[1] for k in ordered], dtype=int)
        weights_g = np.array([pairs[k] for k in ordered])
        A_g = A_c
        level += 1

    levels.append((A_g, None, None))
    return levels


# ---------------------------------------------------------------------------------------- #
# The recorder


class Recorder:
    def __init__(self, solver, n_fine):
        self.solver = solver
        self.n_fine = n_fine
        self.steps = []
        self.state = []
        self.step = 0

    def emit(self, iteration, phase, grid, x, r, detail="", resnorm=None):
        self.steps.append({
            "solver": self.solver,
            "step": self.step,
            "iter": iteration,
            "phase": phase,
            "grid": grid,
            "ngrid": len(x),
            "detail": detail,
            "resnorm": "" if resnorm is None else f"{resnorm:.12e}",
        })

        for cell in range(len(x)):
            self.state.append({
                "solver": self.solver,
                "step": self.step,
                "grid": grid,
                "cell": cell,
                "x": f"{x[cell]:.12e}",
                "r": f"{r[cell]:.12e}",
            })

        self.step += 1


# ---------------------------------------------------------------------------------------- #
# Smoothers


def jacobi_sweep(A, b, x, omega):
    """x += omega * (b - A x) / diag  -- brae's smoothT (device_amg.cu:125-136)."""
    return x + omega * (b - A @ x) / np.diag(A)


def gauss_seidel_sweep(A, b, x, reverse=False):
    """Forward/backward Gauss-Seidel, OpenFOAM's GaussSeidelSmoother."""
    x = x.copy()
    order = range(len(x) - 1, -1, -1) if reverse else range(len(x))

    for i in order:
        x[i] = (b[i] - A[i] @ x + A[i, i] * x[i]) / A[i, i]

    return x


# ---------------------------------------------------------------------------------------- #
# V-cycles


def vcycle_brae(levels, grid, b_g, recorder, iteration, resnorm):
    """
    brae's V(1,1): damped Jacobi omega=0.8, injection restriction, Galerkin coarse operator,
    coarsest grid solved by many Jacobi sweeps (device_amg.cu:2237-2310).
    """
    A_g, cell_map, P = levels[grid]

    if cell_map is None:
        x = np.zeros(len(b_g))

        for _ in range(200):
            x = jacobi_sweep(A_g, b_g, x, OMEGA_BRAE)

        recorder.emit(iteration, "coarse_solve", grid, x, b_g - A_g @ x,
                      f"{len(b_g)} cells, direct-strength Jacobi solve", resnorm)
        return x

    x = np.zeros(len(b_g))

    for _ in range(NPRE_BRAE):
        x = jacobi_sweep(A_g, b_g, x, OMEGA_BRAE)
        recorder.emit(iteration, "presmooth", grid, x, b_g - A_g @ x,
                      f"damped Jacobi omega={OMEGA_BRAE}", resnorm)

    r = b_g - A_g @ x
    recorder.emit(iteration, "residual", grid, x, r, "r = b - A x", resnorm)

    b_c = P.T @ r
    recorder.emit(iteration, "restrict", grid + 1, np.zeros(len(b_c)), b_c,
                  f"inject {len(r)} -> {len(b_c)} cells", resnorm)

    x_c = vcycle_brae(levels, grid + 1, b_c, recorder, iteration, resnorm)

    x = x + P @ x_c
    recorder.emit(iteration, "prolong", grid, x, b_g - A_g @ x,
                  f"broadcast {len(x_c)} -> {len(x)} cells", resnorm)

    for _ in range(NPOST_BRAE):
        x = jacobi_sweep(A_g, b_g, x, OMEGA_BRAE)
        recorder.emit(iteration, "postsmooth", grid, x, b_g - A_g @ x,
                      f"damped Jacobi omega={OMEGA_BRAE}", resnorm)

    return x


def vcycle_amgx(levels, grid, b_g, recorder, iteration, resnorm):
    """AMGX AGGREGATION/SIZE_2 + BLOCK_JACOBI, 2 pre / 2 post, relaxation 0.75."""
    A_g, cell_map, P = levels[grid]

    if cell_map is None:
        x = np.linalg.solve(A_g, b_g)
        recorder.emit(iteration, "coarse_solve", grid, x, b_g - A_g @ x,
                      f"{len(b_g)} cells, dense LU", resnorm)
        return x

    x = np.zeros(len(b_g))

    for _ in range(NPRE_AMGX):
        x = jacobi_sweep(A_g, b_g, x, OMEGA_AMGX)
        recorder.emit(iteration, "presmooth", grid, x, b_g - A_g @ x,
                      f"block Jacobi omega={OMEGA_AMGX}", resnorm)

    r = b_g - A_g @ x
    recorder.emit(iteration, "residual", grid, x, r, "r = b - A x", resnorm)

    b_c = P.T @ r
    recorder.emit(iteration, "restrict", grid + 1, np.zeros(len(b_c)), b_c,
                  f"inject {len(r)} -> {len(b_c)} cells", resnorm)

    x_c = vcycle_amgx(levels, grid + 1, b_c, recorder, iteration, resnorm)

    x = x + P @ x_c
    recorder.emit(iteration, "prolong", grid, x, b_g - A_g @ x,
                  f"broadcast {len(x_c)} -> {len(x)} cells", resnorm)

    for _ in range(NPOST_AMGX):
        x = jacobi_sweep(A_g, b_g, x, OMEGA_AMGX)
        recorder.emit(iteration, "postsmooth", grid, x, b_g - A_g @ x,
                      f"block Jacobi omega={OMEGA_AMGX}", resnorm)

    return x


def scale_correction(A_g, field, source):
    """
    OpenFOAM's GAMGSolver::scaleCorrection (transcribed in src/OpenFOAM/matrices/gamg.cu:206-219).
    An energy line-search on the prolonged correction PLUS a Jacobi post-term; dropping the
    second half is what makes a naive GAMG reference drift from the real solver.

        sf    = (field . source) / (field . A field)
        field = sf * field + (source - sf * A field) / diag
    """
    Acf = A_g @ field
    numerator = float(field @ source)
    denominator = float(field @ Acf)

    if denominator >= 0:
        denominator = max(denominator, 1e-300)
    else:
        denominator = min(denominator, -1e-300)

    sf = numerator / denominator

    return sf * field + (source - sf * Acf) / np.diag(A_g), sf


# ---------------------------------------------------------------------------------------- #
# Outer solvers


def run_brae(levels, A, b, n_iter, recorder):
    """AMG-preconditioned CG: exactly one V(1,1) per Krylov iteration."""
    x = np.zeros(len(b))
    r = b - A @ x
    norm_factor = np.abs(b).sum()
    history = [np.abs(r).sum() / norm_factor]
    p = None
    rho_old = None

    for iteration in range(1, n_iter + 1):
        resnorm = np.abs(r).sum() / norm_factor
        recorder.emit(iteration, "krylov_start", 0, x, r, "CG iteration begins", resnorm)

        z = vcycle_brae(levels, 0, r, recorder, iteration, resnorm)
        rho = float(z @ r)

        if p is None:
            p = z.copy()
        else:
            p = z + (rho / rho_old) * p

        rho_old = rho
        Ap = A @ p
        alpha = rho / float(p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap

        resnorm = np.abs(r).sum() / norm_factor
        history.append(resnorm)
        recorder.emit(iteration, "krylov_update", 0, x, r,
                      f"x += {alpha:.4f} p", resnorm)

    return history


def run_amgx(levels, A, b, n_iter, recorder):
    """AMGX's outer PCG with the aggregation-AMG V-cycle as preconditioner."""
    x = np.zeros(len(b))
    r = b - A @ x
    norm_factor = np.abs(b).sum()
    history = [np.abs(r).sum() / norm_factor]
    p = None
    rho_old = None

    for iteration in range(1, n_iter + 1):
        resnorm = np.abs(r).sum() / norm_factor
        recorder.emit(iteration, "krylov_start", 0, x, r, "CG iteration begins", resnorm)

        z = vcycle_amgx(levels, 0, r, recorder, iteration, resnorm)
        rho = float(z @ r)

        if p is None:
            p = z.copy()
        else:
            p = z + (rho / rho_old) * p

        rho_old = rho
        Ap = A @ p
        alpha = rho / float(p @ Ap)
        x = x + alpha * p
        r = r - alpha * Ap

        resnorm = np.abs(r).sum() / norm_factor
        history.append(resnorm)
        recorder.emit(iteration, "krylov_update", 0, x, r, f"x += {alpha:.4f} p", resnorm)

    return history


def run_of(levels, A, b, n_iter, recorder, anchor=None):
    """
    OpenFOAM GAMG as a STANDALONE solver -- there is no Krylov outer loop at all, which is the
    single biggest structural difference from brae and AMGX. Structure follows OpenFOAM's
    GAMGSolver::solve (transcribed in src/OpenFOAM/matrices/gamg.cu:355-393):

        restrict the finest residual all the way down (nPreSweeps = 0, no smoothing going down)
        direct-solve the coarsest grid
        on the way up: prolong -> scaleCorrection -> nPostSweeps Gauss-Seidel
        at the finest: prolong -> scaleCorrection -> psi += corr -> nFinestSweeps Gauss-Seidel
    """
    coarsest = len(levels) - 1
    x = np.zeros(len(b))
    norm_factor = np.abs(b).sum()
    residual = b - A @ x
    history = [np.abs(residual).sum() / norm_factor]

    for iteration in range(1, n_iter + 1):
        # GAMG carries no state between cycles beyond psi itself, so each cycle can be started
        # from the REAL OpenFOAM psi of the previous cycle. The animated trajectory is then
        # genuine OpenFOAM output and only the intra-cycle breakdown is a reference decomposition.
        if anchor is not None and (iteration - 1) in anchor:
            x = anchor[iteration - 1].copy()
            residual = b - A @ x

        resnorm = np.abs(residual).sum() / norm_factor
        recorder.emit(iteration, "krylov_start", 0, x, residual,
                      "GAMG cycle begins (standalone, no Krylov outer)", resnorm)

        # restrict the residual down every level, smoothing nothing on the way
        source = {}
        current = residual

        for k in range(coarsest):
            _, _, P = levels[k]
            current = P.T @ current
            source[k + 1] = current
            recorder.emit(iteration, "restrict", k + 1, np.zeros(len(current)), current,
                          f"inject {levels[k][0].shape[0]} -> {len(current)} cells", resnorm)

        correction = {}
        A_c = levels[coarsest][0]
        correction[coarsest] = np.linalg.solve(A_c, source[coarsest])
        recorder.emit(iteration, "coarse_solve", coarsest, correction[coarsest],
                      source[coarsest] - A_c @ correction[coarsest],
                      f"{A_c.shape[0]} cells, dense direct solve", resnorm)

        # prolong, scale, post-smooth on the way back up
        for k in range(coarsest - 1, 0, -1):
            A_k, _, P = levels[k]
            field = P @ correction[k + 1]
            field, sf = scale_correction(A_k, field, source[k])
            correction[k] = field
            recorder.emit(iteration, "prolong", k, field, source[k] - A_k @ field,
                          f"broadcast {len(correction[k + 1])} -> {len(field)}, "
                          f"correction scale {sf:.3f}", resnorm)

            for _ in range(NPOST_OF):
                field = gauss_seidel_sweep(A_k, source[k], field)
                recorder.emit(iteration, "postsmooth", k, field, source[k] - A_k @ field,
                              "Gauss-Seidel", resnorm)

            correction[k] = field

        # finest level: prolong the level-1 correction onto psi, then smooth psi itself
        _, _, P0 = levels[0]
        finest = P0 @ correction[1]
        finest, sf = scale_correction(A, finest, residual)
        x = x + finest
        recorder.emit(iteration, "prolong", 0, x, b - A @ x,
                      f"broadcast {len(correction[1])} -> {len(x)}, "
                      f"correction scale {sf:.3f}", resnorm)

        for _ in range(NFINEST_OF):
            x = gauss_seidel_sweep(A, b, x)
            recorder.emit(iteration, "postsmooth", 0, x, b - A @ x,
                          "finest-grid Gauss-Seidel", resnorm)

        residual = b - A @ x
        resnorm = np.abs(residual).sum() / norm_factor
        history.append(resnorm)
        recorder.emit(iteration, "krylov_update", 0, x, residual, "cycle complete", resnorm)

    return history


# ---------------------------------------------------------------------------------------- #


def read_reference(path, n):
    if not os.path.exists(path):
        return None

    psi = collections.defaultdict(lambda: np.zeros(n))

    with open(path) as handle:
        for row in csv.DictReader(handle):
            psi[int(row["cycle"])][int(row["cell"])] = float(row["psi"])

    return psi


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", help="case directory containing trace/")
    parser.add_argument("--iters", type=int, default=8, help="iterations to expand (default 8)")
    args = parser.parse_args()

    trace = os.path.join(args.case, "trace")
    diag, upper, lower, owner, neighbour = read_ldu(os.path.join(trace, "matrix.csv"))
    A = dense(diag, upper, lower, owner, neighbour)
    n = A.shape[0]
    b = read_rhs(os.path.join(trace, "rhs.csv"), n)

    # Face weights for aggregation are |Sf|; on these uniform meshes every internal face has
    # the same area, so unit weights reproduce the dumped hierarchies exactly.
    weights = np.ones(len(owner))

    of_maps = read_hierarchy(os.path.join(trace, "of_hierarchy.csv"))
    brae_maps = read_hierarchy(os.path.join(trace, "brae_hierarchy.csv"))

    print(f"system: {n} cells, {len(owner)} internal faces, cond={np.linalg.cond(A):.1f}")
    print(f"hierarchies: OpenFOAM {[len(m) for m in of_maps]} (real dump), "
          f"brae {[len(m) for m in brae_maps]} (real dump), AMGX reconstructed")

    configs = [
        ("of_gamg", build_levels(A, owner, neighbour, weights, maps=of_maps), run_of),
        ("amgx", build_levels(A, owner, neighbour, weights, maps=None, target=2), run_amgx),
        ("brae", build_levels(A, owner, neighbour, weights, maps=brae_maps), run_brae),
    ]

    all_steps = []
    all_state = []
    validation = []

    for name, levels, runner in configs:
        recorder = Recorder(name, n)
        reference = read_reference(os.path.join(trace, f"{name}_cycles.csv"), n)

        if runner is run_of:
            history = runner(levels, A, b, args.iters, recorder, anchor=reference)
        else:
            history = runner(levels, A, b, args.iters, recorder)
        all_steps.extend(recorder.steps)
        all_state.extend(recorder.state)

        if reference is not None:
            deviations = []

            for iteration in range(1, min(args.iters, max(reference)) + 1):
                rows = [s for s in recorder.steps
                        if s["iter"] == iteration and s["phase"] == "krylov_update"]

                if not rows:
                    continue

                step = rows[-1]["step"]
                x_ref = reference[iteration]
                x_mine = np.array([float(s["x"]) for s in all_state
                                   if s["solver"] == name and s["step"] == step])
                scale = max(np.abs(x_ref).max(), 1e-30)
                deviations.append(np.abs(x_mine - x_ref).max() / scale)

            validation.append((name, max(deviations) if deviations else float("nan")))

        print(f"  {name}: {len(recorder.steps)} frames, "
              f"resnorm {history[0]:.2e} -> {history[-1]:.2e}")

    # Every grid's operator, so the replay can draw the matrix itself shrinking down the
    # hierarchy: A (20x20) -> A_1 (10x10) -> A_2 (5x5) -> ... Dense long form; these are
    # small by construction.
    grid_rows = []
    aggregate_rows = []

    for name, levels, _ in configs:
        chain = np.arange(n)

        for grid, (A_g, cell_map, _) in enumerate(levels):
            for i in range(A_g.shape[0]):
                for j in range(A_g.shape[1]):
                    if A_g[i, j] != 0.0:
                        grid_rows.append({
                            "solver": name,
                            "grid": grid,
                            "row": i,
                            "col": j,
                            "value": f"{A_g[i, j]:.12e}",
                        })

            # which finest-grid cell belongs to which cell of this grid
            for cell in range(n):
                aggregate_rows.append({
                    "solver": name,
                    "grid": grid,
                    "finest_cell": cell,
                    "grid_cell": int(chain[cell]),
                })

            if cell_map is not None:
                chain = cell_map[chain]

    with open(os.path.join(trace, "grids.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["solver", "grid", "row", "col", "value"])
        writer.writeheader()
        writer.writerows(grid_rows)

    with open(os.path.join(trace, "aggregates.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle,
                                fieldnames=["solver", "grid", "finest_cell", "grid_cell"])
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"wrote {trace}/grids.csv ({len(grid_rows)} nonzeros)")
    print(f"wrote {trace}/aggregates.csv ({len(aggregate_rows)} rows)")

    with open(os.path.join(trace, "steps.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["solver", "step", "iter", "phase", "grid", "ngrid", "detail", "resnorm"],
        )
        writer.writeheader()
        writer.writerows(all_steps)

    with open(os.path.join(trace, "state.csv"), "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["solver", "step", "grid", "cell", "x", "r"])
        writer.writeheader()
        writer.writerows(all_state)

    print(f"\nwrote {trace}/steps.csv ({len(all_steps)} frames)")
    print(f"wrote {trace}/state.csv ({len(all_state)} rows)")

    print("\nvalidation -- reference cycle body vs the real solver's own dumped psi")
    print("(relative max deviation over the traced iterations; small = the reference")
    print(" implementation reproduces the real solver step for step)")

    for name, deviation in validation:
        verdict = "MATCHES" if deviation < 1e-6 else "DIFFERS"
        print(f"  {name:<10} {deviation:.3e}  {verdict}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
