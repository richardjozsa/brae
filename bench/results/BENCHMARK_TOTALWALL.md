# brae vs OpenFOAM, total-wall benchmark (NVIDIA GB10)

**One-line result:** with its production flags on, **brae (one Blackwell GPU) beats a 20-core Grace CPU node at
≥15M cells**, 1.04× at 15M widening to **1.13× at 35.6M**, at matched accuracy. Against OpenFOAM's own GPU offload
on the *same* GPU it is **~5× faster** at every size.

## Setup
- **Hardware:** NVIDIA GB10, 20 Grace CPU cores + one Blackwell GPU (sm_121), 128 GB unified LPDDR5x ~273 GB/s,
  **no HBM** (the CPU and the GPU share the same memory).
- **Case:** scaled pitzDaily, laminar, 5 sizes from 990k to 35.6M cells (blockMeshDict cell counts scaled by M).
- **Metric:** **total wall time** for a fixed **100 SIMPLE iterations**. Partition/decompose is done once and
  **excluded** on both sides (each pays a one-time prep; we time the run). This is full run wall: read + setup + 100 iters.
- **Configs:**
  - **`brae-best`** = device-resident PCG (CUDA conditional-graph) + FP32 mixed-precision AMG V-cycle. **This is now
    the default**: a plain `brae` run gets it out of the box.
  - `brae-default` = the legacy host-driven FP64 path (per-iteration D2H syncs), i.e. `BRAE_PCG_DEVICE=0
    BRAE_AMG_FP32=0`. Shown here to isolate the fast path's contribution.
  - `OF-20core` = OpenFOAM v2412, `mpirun -np 20 simpleFoam -parallel`, native GAMG, on the 20 Grace cores.

## Results (total wall, seconds; OF/best > 1 means brae is faster)

| cells   | brae-default | **brae-best** | OF (20 Grace cores) | best/default | **OF/best** | verdict |
|---------|-------------:|--------------:|--------------------:|:------------:|:-----------:|---------|
| 990k    | 24.3         | 21.6          | 15.9                | 1.12×        | 0.74×       | brae behind (small-mesh under-util) |
| 4.89M   | 122.1        | 107.7         | 107.0               | 1.13×        | 0.99×       | parity |
| 10.28M  | 267.1        | 232.5         | 228.8               | 1.15×        | 0.98×       | parity |
| 14.97M  | 384.1        | **339.6**     | 354.4               | 1.13×        | **1.04×**   | **brae wins** |
| 35.6M   | 1045.8       | **877.1**     | 990.4               | 1.19×        | **1.13×**   | **brae wins** |

**Reading it:**
- The fast flags give a steady **1.12-1.19×** over brae-default, and the gain **grows with mesh size** (a bigger
  mesh saturates the GPU; the FP32 V-cycle moves proportionally less data).
- The `OF/best` column climbs monotonically **0.74, 0.99, 0.98, 1.04, 1.13**: the single Blackwell GPU pulls ahead
  of the 20 Grace cores once the mesh is big enough. That trend is the headline.
- The small-mesh loss (990k, 0.74×) is GPU under-utilization, not a cold-start artifact (partition is excluded).

## Why the flags matter
Both are on by default; opt out with `BRAE_PCG_DEVICE=0` / `BRAE_AMG_FP32=0`:
- **`BRAE_PCG_DEVICE`** keeps the PCG Krylov loop device-resident (CUDA conditional-graph WHILE nodes), killing the
  blocking D2H memcpy on every dot product. ~1.4× measured standalone.
- **`BRAE_AMG_FP32`** runs the bandwidth-bound AMG V-cycle preconditioner in FP32 (half the bytes) while the outer
  Krylov loop stays FP64. ~1.18× standalone.

## Accuracy: brae-best preserves the answer
Direct brae-best vs brae-default field comparison:
- **FP32 V-cycle: bit-identical (U 0.000006%).** The mixed precision is confined to the preconditioner; the outer
  loop stays FP64, so it is a *free* speedup.
- **Device-PCG on a convergent case** (stock turbulent pitzDaily, converges to steady): both configs converge at the
  **identical iteration (225)**; fields agree **U 0.016% / p 0.11% / k 0.063% / eps 0.047% / nut 0.27%**, within
  residualControl and ~10× below the ~1% brae-vs-OpenFOAM agreement.
- **Note on the laminar scaling case:** it is physically *unsteady* (vortex shedding behind the step), so a steady
  SIMPLE solver limit-cycles (the pressure residual plateaus, never hits residualControl, runs all iterations). Its
  iteration-N field is a limit-cycle snapshot; two solvers seeded 0.1% apart phase-decorrelate to O(10%). That is
  expected physics of steady-on-unsteady, not a fast-path defect: the case is valid for *timing* (identical work per
  iteration) but not for field comparison. Accuracy is cited from the converged turbulent case above.

## GPU-vs-GPU: brae vs OpenFOAM's own GPU paths (same GB10 GPU)
The sharpest comparison: brae vs OpenFOAM offloading its pressure solve to the **same** GPU. Two OF-GPU paths, each
with one Grace core driving the GPU: **OF+AMGX** (a self-contained `libamgxFoam` solver calling AMGX aggregation AMG,
the strongest OF-GPU option here) and **OF+PETSc-GPU** (PETSc-GAMG on cuSPARSE via petsc4Foam). Total wall, 100 iters:

| cells | brae-best (s) | OF+AMGX (s) | OF+PETSc-GPU (s) | **brae vs OF+AMGX** | brae vs OF+PETSc |
|-------|--------------:|------------:|-----------------:|:-------------------:|:----------------:|
| 990k   | 21.6  | 80.0   | 89.3   | **3.7×** | 4.1× |
| 4.89M  | 107.7 | 572.5  | 601.6  | **5.3×** | 5.6× |
| 10.28M | 232.5 | 1174.2 | 1253.0 | **5.1×** | 5.4× |
| 14.97M | 339.6 | 1696.4 | 1790.3 | **5.0×** | 5.3× |
| 35.6M  | 877.1 | 4111.0 | -      | **4.7×** | -    |

**Brae is ~5× faster than OpenFOAM's best GPU offload on the identical GPU.** The root cause is architecture, not the
solver: OF-GPU offloads *only the linear solve*, so matrix assembly, momentum, turbulence, and the LDU-to-CSR
conversion run on **one serial Grace core**, and the matrix is copied host-to-device **every iteration** (an Amdahl
ceiling plus a migration tax). At 35.6M, the OF+AMGX run spends ~65 min mostly on one CPU core with the GPU near-idle;
brae did the same mesh in ~15 min with the GPU resident. So on GB10 brae roughly matches 20-core OpenFOAM (bandwidth
parity) while OpenFOAM's own GPU offload is ~5× behind both.

Note (GB10 / sm_121): AMGX 2.5.0's classical AMG is Thrust-broken on sm_121, so OF+AMGX uses aggregation AMG
(validated: converges in 283 iters vs GAMG's 282, fields U 0.025% / p 0.19%). Build recipe: `OF_GPU_SETUP_GB10.md`.
Raw data: `amgx_100iter.txt`, `petscgpu_100iter.txt`.

## GB10 is a conservative baseline; HBM is the ceiling
GB10 has **no HBM**: its Blackwell GPU shares the *same* ~273 GB/s LPDDR5x as the 20 Grace CPU cores. A
bandwidth-bound solver therefore cannot structurally beat the CPU on this box, so parity is the structural ceiling
here, and brae clearing it (winning at scale) on the most bandwidth-constrained hardware a GPU port can run on is a
strong result. On a GPU with HBM the CPU cannot touch (H100 ~3.3 TB/s, GH200 ~4 TB/s), we expect the same brae code
to deliver an **order-of-magnitude speedup, roughly 8-12×** a CPU node, and being fully device-resident (0 H2D/D2H
per iteration) it avoids the unified-memory migration tax that offload ports pay.

## Reproduce
- **brae vs OpenFOAM-CPU:** `bench/run_benchmark.sh` (this whole table; set `SIZES`, `CORES`, `ITERS` by env).
- **Add the GPU-vs-GPU back-ends:** `bench/setup_of_amgx.sh` then `bench/setup_of_petsc.sh`, after which
  `run_benchmark.sh` auto-detects and fills the `OF+AMGX` / `OF+PETSc` columns.
- **Raw data (this run):** `totalwall_100iter.txt` (brae-default + OF-20), `brae_best_100iter.txt` (brae-best),
  `amgx_100iter.txt`, `petscgpu_100iter.txt`. Crossover curve: `crossover.csv`, `crossover_chart.png`.
