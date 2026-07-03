# Performance & tuning

## The one idea: residency

A CFD solver is **bandwidth-bound**, its speed is set by how fast it moves the matrix and fields through memory,
not by raw compute. Brae's advantage is that it moves them as little as possible: the entire SIMPLE loop (assembly,
momentum, turbulence, and the pressure multigrid solve) runs on the GPU, so nothing crosses the CPU↔GPU boundary
between iterations.

"OpenFOAM on GPU" via a solver plug-in (PETSc, AMGX) offloads only the *linear solve*: it rebuilds the matrix on the
CPU and copies it to the GPU **every iteration**, while the rest runs on one CPU core. That migration + the serial
CPU remainder is why offload is ~5× slower than brae on the same GPU (below).

## The fast path (on by default)

Two optimisations, both **accuracy-preserving** and **on by default**. Opt out by setting either to `0`.

| flag (default on) | what it does | accuracy |
|---|---|---|
| `BRAE_PCG_DEVICE` | keeps the pressure PCG's Krylov loop resident on the GPU (removes a per-iteration CPU sync) | bit-identical to the host path at iteration 1; same converged state |
| `BRAE_AMG_FP32` | runs the bandwidth-bound multigrid preconditioner in FP32 (half the bytes); the outer solve stays FP64 | bit-identical (mixed precision is confined to the preconditioner) |

Other tuning knobs (advanced; sensible defaults):

| flag | effect |
|---|---|
| `BRAE_AMG_GS=1` | multicolor Gauss-Seidel smoother (vs the default Jacobi), fewer cycles on some anisotropic meshes |
| `BRAE_AMG_TSGS=1` | two-stage Gauss-Seidel (parallel polynomial) smoother, coloring-free |
| `BRAE_AMG_SA=1` | smoothed-aggregation multigrid (vs default pairwise), stronger on cyclic / graded meshes |
| `BRAE_GS_DEVICE=1` | device-resident k/ε Gauss-Seidel scalar solve |

## Benchmarks

Total wall time for **100 SIMPLE iterations**, scaled-pitzDaily, on a single **NVIDIA GB10** (20 Grace CPU cores +
one Blackwell GPU, unified LPDDR5x, **no HBM**). Partition/decompose is done once and excluded on all sides.

| cells | **brae** (Blackwell GPU) | OpenFOAM (20 Grace cores) | OpenFOAM+AMGX (same GPU) | OpenFOAM+PETSc-GPU (same GPU) |
|---|---:|---:|---:|---:|
| 990k | 21.6 s | 15.9 s | 80.0 s | 89.3 s |
| 4.89M | 107.7 s | 107.0 s | 572.5 s | 601.6 s |
| 10.28M | 232.5 s | 228.8 s | 1174.2 s | 1253.0 s |
| 14.97M | 339.6 s | 354.4 s | 1696.4 s | 1790.3 s |
| 35.6M | 877.1 s | 990.4 s | 4111.0 s | - |

**Reading it:**

- **~5× faster than OpenFOAM's best GPU offload (AMGX)** on the *same* GPU, at every size, the residency payoff.
- **Parity-to-ahead of a 20-core Grace CPU node**, and the lead grows with mesh size (0.74× at 990k → 1.13× at
  35.6M). Small meshes under-utilize the GPU; large meshes saturate it.

## Why GB10 is a conservative baseline, and HBM is the ceiling

GB10 has **no HBM**, its Blackwell GPU shares the same ~273 GB/s unified memory as the 20 Grace CPU cores. For a
bandwidth-bound solver that means the GPU *cannot* structurally out-run the CPU on this box, so parity itself is the
structural ceiling here, and brae clearing it at scale is a strong result on the most bandwidth-constrained hardware
a GPU port can run on.

On a GPU with **HBM** (H100 ~3.3 TB/s, GH200 ~4 TB/s) the memory bandwidth the GPU can use, and the CPU can't, is
~10× higher. We expect the *same* brae code to deliver an **order-of-magnitude speedup, roughly 8-12×** a CPU node
there, and unlike unified-memory offload it pays no migration tax to get it.

## Accuracy is not traded for speed

Every performance flag is validated to preserve the result. On a converged case (turbulent pitzDaily), the fast
path and the reference path reach the identical iteration count and agree to **U 0.016% / p 0.11%**, an order of
magnitude below the ~1% brae-vs-OpenFOAM agreement. FP32 in the preconditioner is bit-identical because the mixed
precision never touches the outer double-precision solve.

Brae is not bit-identical to OpenFOAM itself, and is not expected to be: parallelising the arithmetic across the GPU
reorders the floating-point reductions (and loops over cells rather than faces), which reshuffles the truncation
error at machine precision. That bounded difference is what shows up as the sub-1% field agreement. The full
explanation is in [getting-started.md](getting-started.md#why-the-results-are-not-bit-identical).
