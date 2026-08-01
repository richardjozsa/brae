#!/usr/bin/env python3
"""
Fold every solver's per-cycle trace for one case into a single tidy CSV that the replay
animation and any downstream plot can read:

    trace/compare.csv    solver,cycle,resnorm,err_inf,fine_sweeps,fine_spmv

`resnorm` is recomputed here from the dumped matrix and each solver's dumped psi, using one
identical formula (OpenFOAM's L1 norm, |b - A x|_1 / |b|_1). The solvers themselves report
different norms -- OpenFOAM GAMG an L1 normFactor norm, AMGX an L2 relative norm -- so their
own printed residuals are not comparable. Recomputing from psi makes them comparable.

`fine_sweeps` / `fine_spmv` are the work-normalised axis. Comparing cycle counts alone is
misleading: the three solvers do very different amounts of work inside one cycle.

    OpenFOAM GAMG   nPreSweeps 0 + nPostSweeps 2 + nFinestSweeps 2   -> 4 fine sweeps/cycle
    AMGX            presweeps 2 + postsweeps 2                       -> 4 fine sweeps/cycle
    brae AMG-PCG    NPRE 1 + NPOST 1, plus the CG matvec             -> 2 fine sweeps/cycle

Usage:  python3 make_compare.py demo/amgpcg/case1024
"""

import argparse
import collections
import csv
import os
import sys

import numpy as np


# Fine-grid smoothing sweeps and matrix-vector products charged to one cycle of each solver.
# Sweeps come from the configured pre/post counts; each sweep costs one SpMV, and brae's
# outer CG additionally applies A once per iteration.
WORK_PER_CYCLE = {
    "of_gamg":   {"sweeps": 4, "spmv": 4, "label": "OpenFOAM GAMG"},
    "amgx":      {"sweeps": 4, "spmv": 4, "label": "AMGX (PCG+AGGREGATION)"},
    "brae":      {"sweeps": 2, "spmv": 3, "label": "brae AMG-PCG"},
    "brae_cheb": {"sweeps": 4, "spmv": 5, "label": "brae AMG-PCG (Chebyshev deg 2)"},
    "brae_gs":   {"sweeps": 2, "spmv": 3, "label": "brae AMG-PCG (multicolor GS)"},
    "brae_soc":  {"sweeps": 2, "spmv": 3, "label": "brae AMG-PCG (strength filter)"},
}


def read_system(trace_dir):
    """Rebuild the dense A and b from the LDU CSV. Fine for the demo sizes."""
    diag = {}
    faces = []

    with open(os.path.join(trace_dir, "matrix.csv")) as handle:
        for row in csv.DictReader(handle):
            value = float(row["value"])
            owner = int(row["owner"])
            neighbour = int(row["neighbour"])

            if row["kind"] == "diag":
                diag[owner] = value
            else:
                faces.append((row["kind"], owner, neighbour, value))

    n = len(diag)
    A = np.zeros((n, n))

    for cell, value in diag.items():
        A[cell, cell] = value

    for kind, owner, neighbour, value in faces:
        if kind == "upper":
            A[owner, neighbour] = value
        else:
            A[neighbour, owner] = value

    b = np.zeros(n)

    with open(os.path.join(trace_dir, "rhs.csv")) as handle:
        for row in csv.DictReader(handle):
            b[int(row["cell"])] = float(row["b"])

    return A, b


def read_cycles(path, n):
    """cycle -> psi vector, from a *_cycles.csv trace."""
    psi = collections.defaultdict(lambda: np.zeros(n))

    with open(path) as handle:
        for row in csv.DictReader(handle):
            psi[int(row["cycle"])][int(row["cell"])] = float(row["psi"])

    return psi


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("case", help="case directory containing trace/")
    args = parser.parse_args()

    trace_dir = os.path.join(args.case, "trace")
    A, b = read_system(trace_dir)
    n = len(b)

    x_exact = np.linalg.solve(A, b)
    norm_factor = np.abs(b).sum()

    eigenvalues = np.linalg.eigvalsh(A)
    print(f"system: {n} cells, symmetric={np.allclose(A, A.T)}, "
          f"SPD={eigenvalues.min() > 0}, cond={np.linalg.cond(A):.1f}")

    out_path = os.path.join(trace_dir, "compare.csv")
    rows = []

    for key, work in WORK_PER_CYCLE.items():
        path = os.path.join(trace_dir, f"{key}_cycles.csv")

        if not os.path.exists(path):
            continue

        psi = read_cycles(path, n)

        for cycle in sorted(psi):
            x = psi[cycle]
            resnorm = np.abs(b - A @ x).sum() / norm_factor
            err_inf = np.abs(x - x_exact).max()

            rows.append({
                "solver": key,
                "label": work["label"],
                "cycle": cycle,
                "resnorm": f"{resnorm:.12e}",
                "err_inf": f"{err_inf:.12e}",
                "fine_sweeps": cycle * work["sweeps"],
                "fine_spmv": cycle * work["spmv"],
            })

    with open(out_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["solver", "label", "cycle", "resnorm", "err_inf",
                        "fine_sweeps", "fine_spmv"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"wrote {out_path} ({len(rows)} rows)")

    # Summary: cycles and fine-grid SpMVs to reach 1e-8.
    print()
    print(f"{'solver':<32} {'cycles to 1e-8':>15} {'fine SpMV to 1e-8':>19}")

    for key, work in WORK_PER_CYCLE.items():
        matching = [r for r in rows if r["solver"] == key]

        if not matching:
            continue

        converged = [r for r in matching if float(r["resnorm"]) < 1e-8]

        if not converged:
            print(f"{work['label']:<32} {'not reached':>15} {'-':>19}")
            continue

        first = min(converged, key=lambda r: r["cycle"])
        print(f"{work['label']:<32} {first['cycle']:>15} {first['fine_spmv']:>19}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
