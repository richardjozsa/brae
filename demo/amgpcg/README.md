# AMG-PCG demo: same cells, same matrix, three solvers

A 20-cell OpenFOAM case solved by **OpenFOAM GAMG**, **AMGX** and **brae's AMG-PCG**, with every
internal step of every cycle dumped to CSV and replayed as video. Small enough that the whole
20x20 matrix, the agglomeration, and each smoothing sweep fit on one frame.

```
demo/amgpcg/
  case20/                 4 x 5 x 1 = 20 cells    the teaching mesh
  case1024/               32 x 32 x 1 = 1024      the comparison mesh (20 cells is too small
                                                  to compare convergence honestly, see below)
  of_apps/dumpAmgTrace/   OpenFOAM app: matrix, rhs, real GAMG hierarchy, per-cycle state
  trace_brae.cu           brae tool: real agglomeration, Galerkin operators, per-iteration state
  trace_amgx.c            AMGX tool: LDU -> CSR, per-iteration state
  make_steps.py           expands each iteration into its internal steps
  make_compare.py         work-normalised convergence comparison
  replay.py               CSV -> mp4 / gif
  run_all.sh              regenerate everything
  out/                    rendered videos
```

Regenerate everything:

```bash
./demo/amgpcg/run_all.sh
```

Render a video from CSVs that already exist:

```bash
python3 demo/amgpcg/replay.py demo/amgpcg/case20 --solver brae  -o out/brae.mp4
python3 demo/amgpcg/replay.py demo/amgpcg/case20 --compare      -o out/compare.mp4
python3 demo/amgpcg/replay.py demo/amgpcg/case20 --solver brae --field r --gif -o out/res.gif
```

`replay.py` computes nothing. It reads the CSVs and draws them, so editing a CSV changes the
video. Swap it for your own renderer using the schema below.

---

## What the video shows

Four panels, driven frame by frame from `steps.csv` + `state.csv`:

| panel | what it shows |
|---|---|
| **mesh** | the cells, filled by the vector the current step is acting on, with the **aggregate boundaries of the active grid** drawn on top. Restriction visibly merges cells; prolongation visibly splits them back. |
| **matrix** | the operator of the active grid as a heatmap with its coefficients printed. It shrinks 20x20 -> 10x10 -> 5x5 -> 2x2 going down the cycle and grows back coming up. |
| **V-cycle** | schematic of the level chain with the current position marked. |
| **convergence** | all three solvers, marker on the current cycle. |

Colour is scaled **per grid**, held fixed across the run. A single global scale would wash the
coarse levels out: grid 0 carries the solution (O(1)) while grids 1+ carry a correction to it,
orders of magnitude smaller.

---

## How real is each number

This matters more than the animation, so it is stated precisely.

| artefact | source | real? |
|---|---|---|
| mesh, cell centres | `blockMesh` | **real** |
| fine matrix `A`, rhs `b` | OpenFOAM `fvm::laplacian`, boundary coefficients folded into diag/source exactly as `lduMatrix::solve` does | **real** |
| OpenFOAM GAMG hierarchy | `GAMGAgglomeration::New`, `faceAreaPair` | **real dump** |
| brae hierarchy + Galerkin coarse operators | `buildAMG` + `amgGalerkin` via `trace_brae` | **real dump** |
| OpenFOAM GAMG per-cycle `psi` | real `simpleFoam`-style solves at `maxIter = 1, 2, 3, ...` | **real** |
| AMGX per-iteration `psi` | real AMGX 2.5.0, same config as `bench/amgxFoam/amgxSolver.C:205-215` | **real** |
| brae per-iteration `psi` | real `deviceAMGPCG` | **real** |
| **intra-cycle steps** (each sweep, restriction, prolongation) | **reference implementations** in `make_steps.py` | see below |
| **AMGX hierarchy** | **reconstructed** SIZE_2 greedy pairwise | see below |

**Why the intra-cycle steps are reference implementations.** Neither OpenFOAM nor AMGX exposes
its per-sweep vectors; there is no API for "give me `x` halfway through the V-cycle". To animate
the process at all, the cycle bodies are re-implemented from the documented algorithms and then
checked against each solver's own dumped `psi`. `make_steps.py` prints the deviation every run:

```
of_gamg    5.575e-03      structurally faithful; residual gap from OpenFOAM's coarse-level
                          interface handling, which agglomerates boundary coefficients
                          separately rather than folding them into the diagonal
amgx       1.488e-01      hierarchy is reconstructed, not AMGX's own
brae       9.043e-08      MATCHES -- the reference V-cycle reproduces the real solver
```

Two mitigations are in place. GAMG carries no state between cycles beyond `psi`, so each animated
cycle is **anchored to the real OpenFOAM `psi` of the previous cycle** — what you see at every
cycle boundary is genuine OpenFOAM output, and only the breakdown within one cycle is a
reference decomposition. And brae's leg needs no mitigation: it reproduces the real solver to
1e-8, so the brae video is faithful end to end.

If you are cutting a video, **prefer the brae and OpenFOAM legs for step-by-step narration**, and
use AMGX for the convergence curve only.

---

## Findings

### The 20-cell case is too small to compare convergence

AMGX solves it in **one iteration**: its hierarchy bottoms out at `min_coarse_rows=2` and the
coarse solve inverts the toy system exactly. Any convergence claim made on 20 cells is an
artefact. That is why `case1024` exists. Use 20 cells to explain *mechanism*, 1024 to compare
*rates*.

### Cycle counts, 1024 cells, uniform Laplacian, to `|r|_1 / |b|_1 < 1e-8`

Recomputed from each solver's dumped `psi` with one identical formula — the solvers report
different norms natively (OpenFOAM L1 with a normFactor, AMGX relative L2) and their printed
residuals are **not** comparable.

| solver | cycles | fine-grid SpMV | sweeps per cycle |
|---|---:|---:|---|
| OpenFOAM GAMG | 16 | 64 | 0 pre + 2 post GS + 2 finest |
| AMGX | 11 | 44 | 2 pre + 2 post block-Jacobi |
| **brae AMG-PCG** | **17** | **51** | 1 pre + 1 post damped Jacobi |

**brae does not win on cycle count**, and the demo should not claim it does. It runs the
cheapest and weakest cycle of the three: fewest sweeps, so more cycles. On *work* it passes
OpenFOAM GAMG (51 vs 64 fine-grid SpMV) and still trails AMGX on this clean isotropic problem.

The repo's own `validation/scale_v2_results.md:16-18` says the same thing at scale: 2-2.8x more
V-cycles than OpenFOAM GAMG. The real advantages are elsewhere — one graph launch per solve with
zero host syncs, an FP32 cycle, and a hierarchy built once per mesh and cached to disk.

### Mesh-independence: the finding that matters

Cycles to `|r|_1 / |b|_1 < 1e-8` on a uniform 2D Laplacian, both solvers on the same mesh
(`demo/amgpcg/case1M` and the `scale_*` sequence). A textbook multigrid method is
*mesh-independent*: the cycle count stays flat as the mesh refines.

| cells | OpenFOAM GAMG | levels | brae default | levels | brae `SA` | brae `SA` + `SOC` |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 16 | 8 | 24 | 6 | 17 | 17 |
| 16,384 | 16 | 10 | 34 | 8 | 19 | 18 |
| 65,536 | 15 | 12 | 48 | 10 | 21 | 21 |
| 262,144 | 15 | 14 | 68 | 12 | 24 | 24 |
| **1,000,000** | **15** | 16 | **92** | 14 | **36** | **25** |

**OpenFOAM GAMG is mesh-independent (16 -> 15). brae's default is not (24 -> 92).** The default
cycle count tracks the level count almost linearly — roughly `4-6.6 x nLevels` — which is the
signature of *unsmoothed* aggregation: with a 0/1 injection prolongator the coarse-grid
correction loses accuracy at every level, and nothing in a V(1,1) damped-Jacobi cycle recovers
it. Confirmed directly by capping the depth at 1M: 14 levels -> 92 cycles, 11 -> 70, 8 -> 49
(`BRAE_AMG_TARGET`). Coarsest-solve accuracy is *not* the cause — `BRAE_NCOARSE_CG` 16, 64 and
200 all give 92.

**`BRAE_AMG_SA=1` is the fix and it is off by default.** Smoothed aggregation
(`P = (I - omega D^-1 A) P_tent`) takes 1M from 92 to 36 cycles, and adding
`BRAE_AMG_SOC=0.05` reaches 25 — near mesh-independent (17 -> 25 across a 244x mesh range)
and within 1.7x of GAMG at 1M.

Caveats before acting on this:

- **This is a clean isotropic Poisson problem, where GAMG is at its strongest.** On the real
  stiff CFD pressure matrix in `validation/matrixDumpGAMG` the ranking inverts and brae's
  default beats GAMG (see the table below). Relative standing is problem-dependent; the
  *loss of mesh-independence* is structural and will show up on any large problem.
### Per-solve wall-clock: SA pays above ~200k cells, and loses below it

Measured with `trace_brae <case> -10` (benchmark mode): hierarchy built **once outside** the
timing loop, since the mesh is static and `-partition` caches it to disk; each repeat re-runs
`amgGalerkin` + the full solve to `1e-8` from `x = 0`; best of 10 after 3 warm-ups. GB10.

| cells | default: iters, solve, ms/cycle | SA+SOC: iters, solve, ms/cycle | SA speedup |
|---:|---|---|---:|
| 4,096 | 24, 12.7 ms, 0.51 | 17, 26.4 ms, 1.54 | **0.48x** |
| 16,384 | 34, 18.3 ms, 0.52 | 18, 28.3 ms, 1.56 | 0.65x |
| 65,536 | 48, 24.5 ms, 0.51 | 21, 35.0 ms, 1.64 | 0.70x |
| 262,144 | 68, 79.0 ms, 1.15 | 24, 57.8 ms, 2.31 | **1.37x** |
| 1,000,000 | 92, 334.5 ms, 3.61 | 25, 142.5 ms, 5.41 | **2.35x** |

**The crossover sits between 65k and 262k cells.** Below it, SA's heavier cycle (3x the cost at
small sizes, where the extra kernel launches dominate and the hierarchy is too shallow for
smoothing to matter) outweighs its better convergence. Above it, SA wins and the margin grows:
at 1M the cycle costs 1.5x more but you need 3.7x fewer of them.

This also explains the 12,225-cell result further down, where SA measured *worse* than the
default (49.4 vs 44.6 cycles) — that mesh is far below the crossover.

So the recommendation is **not** "make SA the default". It is **make SA conditional on problem
size** (or on hierarchy depth, which is the actual driver). `amgGalerkin` overhead is negligible
either way: +1.8 ms default, +7.4 ms SA per SIMPLE step at 1M.

Not yet validated on a real CFD pressure matrix at scale — everything above is uniform isotropic
Poisson. Do that before touching a default.

### GPU vs GPU on the same GPU: brae vs AMGX device solve

The fair comparison is brae against another GPU solver on the *same* GPU, not against a CPU. Both
brae and AMGX solve the identical matrix on the GB10, to `|r|_1/|b|_1 < 1e-8`, solve time only
(hierarchy already built, best of 10 after warm-up):

| cells | AMGX (iters, solve) | brae default (iters, solve) | brae SA+SOC (iters, solve) | brae-best vs AMGX |
|---:|---|---|---|---:|
| 65,536 | 46 it, 95.3 ms | 48 it, 30.2 ms | 21 it, 35.7 ms | **3.2x** |
| 262,144 | 79 it, 340.2 ms | 68 it, 87.6 ms | 24 it, 56.8 ms | **6.0x** |
| 1,000,000 | 90 it, 654.8 ms | 92 it, 317.5 ms | 25 it, 140.4 ms | **4.7x** |

**brae's solve is 3-6x faster than AMGX's on the same GPU, at nearly the same iteration count.**
This is the clean apples-to-apples number, and it isolates the real advantage: AMGX and brae run
structurally the same algorithm (PCG + pairwise aggregation + V-cycle), so the gap is not the
math — it is that brae's LDU-gather matvec hits ~85% of memory bandwidth where AMGX's `csrmv`
sits far lower, and brae launches far fewer kernels per cycle.

Two honesty notes on this table:
- AMGX's reported `setup+solve` came out equal to `solve`, which is implausible (a matrix
  re-upload + `AMGX_solver_setup` is not free). The instrumentation is not correctly isolating
  AMGX's per-step re-setup cost, so **only the solve-only column is trustworthy here** — and it
  already favors brae. The setup gap (brae's cached hierarchy vs AMGX re-aggregating every step)
  is real but is quantified properly in `bench/H100`, not here.
- Even the solve-only comparison is on GB10, the *worst* hardware for brae (no HBM). On H100 the
  brae column shrinks further relative to AMGX; iteration counts are identical on any GPU.

### The CPU comparison, and why the hardware matters

Same solves, against OpenFOAM GAMG on **one CPU core** (the dump app is serial):

| cells | OF GAMG, 1 core | brae default | brae SA+SOC | brae-best vs 1-core OF |
|---:|---|---|---|---:|
| 65,536 | 15 it, 40 ms | 48 it, 27 ms | 21 it, 33 ms | 1.48x |
| 262,144 | 15 it, 190 ms | 68 it, 86 ms | 24 it, 57 ms | 3.33x |
| 1,000,000 | 15 it, 740 ms | 92 it, 325 ms | 25 it, 143 ms | 5.16x |

**Do not quote these as a win.** The baseline is one core; the repo's public claims are against
20-24 cores. GAMG parallelises well, so a 20-core run of the same solve would land near 55-90 ms
at 1M — which would put brae *behind* on this problem, on this hardware. That is consistent with
`validation/scale_v2_results.md`, which measured brae 1.7-2.3x slower than 20-core OpenFOAM on
GB10.

The hardware is the confound. GB10 has no HBM — its GPU shares the CPU's LPDDR at ~273 GB/s —
and brae's cycles are bandwidth-bound. Every ms/cycle number above is therefore a worst case.
On an H100 (~2 TB/s HBM3, ~7x the bandwidth) the per-cycle times shrink while the CPU baseline
does not, which is exactly the mechanism `bench/H100/H100_GPU_COMPARISON_REPORT.md` reports.
**Iteration counts would not change at all** — those are pure arithmetic, identical on any GPU.
Re-run the *timing* tables on H100; leave the *cycle-count* tables alone.

Two defects surfaced at 1M:

- **`BRAE_CHEBYSHEV=1` failed to converge** — 5000 iterations, residual stalled at 1.6e-3, while
  it works at 1024 cells. **Root cause found and fixed** (`src/cuda/device_amg.cu`): the
  power-iteration eigenvalue estimate (`estimateLambdaMax`) seeded from the *constant* vector,
  which is the Laplacian's near-null-space — in exact arithmetic a constant start never leaves
  the smallest mode, so in 12 iterations it badly under-estimated `lambda_max`. An
  under-estimated Chebyshev interval leaves the true top modes above it, where the polynomial
  amplifies instead of damps: divergence, and it worsens with mesh depth exactly as observed.
  Fix: seed with a period-7 sawtooth (broad spectral content, like the SA path already uses),
  25 iterations instead of 12, and a 1.2x interval-top safety factor. **Compiles clean; not yet
  re-run on GPU** (machine was down). Verify with:
  `BRAE_CHEBYSHEV=1 build/trace_brae demo/amgpcg/case1M 0` — expect ~25-40 iterations, not 5000.
- **`BRAE_CORR_SCALING=1` originally had no effect in this harness** — not a brae defect, just a
  harness gap: correction scaling is a `deviceAMGPCG` argument (by design, since it forces the
  host-driven flexible-CG path), wired by `gpuSimpleFoam.cu:416`, and `trace_brae` was not
  passing it. Now fixed: `trace_brae` reads `BRAE_CORR_SCALING` and passes it through, so its
  effect can be measured once the GPU is back.

`BRAE_AMG_FP32` is innocent: 92 cycles with it on and off.

### Where the smoother variants pay

On this uniform mesh, `BRAE_CHEBYSHEV`, `BRAE_AMG_GS` and `BRAE_AMG_SOC` do **not** help
(17 cycles each, Chebyshev costs more work for the same count). They are anisotropy cures. On a
stretched, stiff mesh they are worth ~2x: measured on `validation/matrixDumpGAMG` (12,225 cells,
`tolerance 1e-8, relTol 0`), mean cycles per solve over 10 SIMPLE steps —

| config | mean cycles | vs OpenFOAM GAMG |
|---|---:|---:|
| OpenFOAM GAMG | 70.9 | — |
| brae default | 44.6 | 1.59x fewer |
| brae `BRAE_AMG_GS=1` | 39.1 | 1.81x fewer |
| brae `BRAE_CHEBYSHEV=1` | 35.7 | **1.99x fewer** |
| brae `BRAE_AMG_SOC=0.05` | 35.8 | **1.98x fewer** |

At the loose tolerance production SIMPLE actually uses (`relTol 0.1`) the same case gives
OpenFOAM 2.3 and brae 2.8 cycles — brae slightly behind. **The "fewer iterations" claim is
real but tolerance-dependent**; it holds when you demand many digits, not at engineering
tolerance.

---

## CSV schema

All files live in `<case>/trace/`. Long/tidy format throughout — one observation per row.

### `mesh.csv`
```
cell,x,y,z,volume
```
Cell centres and volumes. `replay.py` recovers the structured `(i, j)` index by sorting unique
coordinates, which works for any block mesh.

### `matrix.csv` — the fine operator, LDU form
```
kind,index,owner,neighbour,value
```
`kind` is `diag`, `upper` or `lower`. For `diag`, `index = owner = neighbour = cell`. For a
face, `index` is the face id and `owner`/`neighbour` are the two cells it connects, so
`A[owner, neighbour] = upper[f]` and `A[neighbour, owner] = lower[f]`. Boundary contributions
are already folded into `diag`.

This is the native storage of both OpenFOAM and brae — there is no CSR anywhere in brae, which
is the point of `docs/memory-model.md`. `trace_amgx.c` converts to CSR because AMGX requires it;
that conversion is the per-iteration tax every offload solver pays.

### `rhs.csv`
```
cell,b
```

### `of_hierarchy.csv`, `brae_hierarchy.csv`
```
level,fine_cell,coarse_cell,finest_cell
```
Two row kinds. When `fine_cell >= 0` and `finest_cell = -1`: the level's own restriction map,
`fine_cell` of this level goes into `coarse_cell` of the next. When `fine_cell = -1`: the
chained map, telling you which cell of `level` the **original finest** cell `finest_cell` ends
up in — this is what colours the mesh panel.

### `brae_levels.csv` — the real Galerkin operators
```
grid,kind,index,owner,neighbour,value
```
Same LDU encoding as `matrix.csv`, one block per grid (`grid 0` is the fine matrix).

### `*_cycles.csv` — real per-iteration state
`of_cycles.csv` / `of_gamg_cycles.csv`, `amgx_cycles.csv`, `brae_cycles.csv`, plus the variants
`brae_cheb_cycles.csv`, `brae_gs_cycles.csv`, `brae_soc_cycles.csv`.
```
cycle,cell,psi,residual,resnorm
```
`cycle = 0` is the initial state (`psi = 0`). `residual` is `b - A psi` per cell; `resnorm` is
whatever the solver itself reported, which is **not** comparable across solvers — use
`compare.csv` for that.

### `steps.csv` — one row per animation frame
```
solver,step,iter,phase,grid,ngrid,detail,resnorm
```
`phase` is one of `krylov_start`, `presmooth`, `residual`, `restrict`, `coarse_solve`,
`prolong`, `postsmooth`, `krylov_update`. `grid` is the active level (0 = finest), `ngrid` its
cell count, `detail` a human-readable note (e.g. `damped Jacobi omega=0.8`,
`correction scale 1.032`).

### `state.csv` — the vector each frame acts on
```
solver,step,grid,cell,x,r
```
`cell` indexes the **active grid**, not the finest, so it runs `0..ngrid-1`. Join to
`aggregates.csv` to paint it back onto the original mesh.

### `grids.csv` — every level's operator, dense long form
```
solver,grid,row,col,value
```
Nonzeros only.

### `aggregates.csv`
```
solver,grid,finest_cell,grid_cell
```
Which cell of `grid` each original finest cell belongs to. This is the join that makes
`state.csv` paintable on the mesh.

### `compare.csv` — the comparable convergence table
```
solver,label,cycle,resnorm,err_inf,fine_sweeps,fine_spmv
```
`resnorm` recomputed uniformly as `|b - A x|_1 / |b|_1`; `err_inf` is `max|x - x_exact|` against
a dense solve. `fine_sweeps` / `fine_spmv` are the work-normalised axes — plot against these,
not against `cycle`, when comparing solvers with different cycle bodies.

---

## Reproducing on your own case

Any OpenFOAM case with a `T` field and a GAMG entry for `T` in `fvSolution` works:

```bash
cd yourCase && blockMesh && dumpAmgTrace -maxCycles 30
BRAE_AMG_TARGET=4 build/trace_brae yourCase 30       # target only needed below ~64 cells
LD_LIBRARY_PATH=$AMGX_ROOT/lib build/trace_amgx yourCase 30
python3 demo/amgpcg/make_steps.py yourCase --iters 6
python3 demo/amgpcg/replay.py yourCase --compare -o compare.mp4
```

Above a few thousand cells drop the in-cell labels (`replay.py` does this automatically past 40
cells) and expect `state.csv` to grow linearly with `cells x frames`.

---

## Notes for whoever edits the solver next

Two things surfaced while building this.

**`BRAE_AMG_TARGET`** was added to `device_amg.cu` so the coarsening floor (default 64 cells) can
be overridden. Without it any mesh below 64 cells builds a zero-level hierarchy and the AMG
degenerates to the coarsest solve alone.

**`maxIter`, `tol` and `relTol` in the PCG graph cache key — FIXED (D2).** `deviceAMGPCGGraph`
used to cache its conditional graph on `psi.data()` alone, while all three convergence controls
are literal kernel arguments baked into the captured `pcgSetCondK` node
(`pcgSetCondK<<<...>>>(c.handle, c.sRes.data(), tol, c.sInit.data(), relTol, c.sIter.data(),
maxIter-1)`). Reusing the cache with any of them changed silently replayed the stale
bound/tolerance. Now `PCGGraphCache` carries `keyTol/keyRelTol/keyMaxIter` and both the local and
distributed capture guards compare all four, so a changed control forces one re-capture; re-capture
cost is zero in the SIMPLE loop where the controls are constant per field. `trace_brae.cu` still
sets `BRAE_PCG_DEVICE=0` for its growing-`maxIter` sweep — belt-and-suspenders, and it keeps the
trace on the host loop that is documented bit-identical to the graph path. See `BUGS.md` (D2).
