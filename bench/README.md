# brae benchmarks, replay them on your GPU

Everything here is self-contained so you can reproduce the numbers on your own hardware. The headline comparison is
**total wall time for a fixed number of SIMPLE iterations**, with partition/decompose done once and excluded, the
honest apples-to-apples: both sides pay a one-time prep, we time the run.

## Quick start (brae vs OpenFOAM-CPU)

```bash
cd bench
./run_benchmark.sh                 # brae (1 Blackwell GPU) vs OpenFOAM (20 Grace cores), a few mesh sizes
```

Needs: a built `brae` (`../build/brae`) and OpenFOAM sourced. Tune with env vars:

```bash
BRAE=../build/brae CORES=16 SIZES="9 20 29" ITERS=100 ./run_benchmark.sh
```

Output is a table + `CSV` in the work dir. `SIZES` are `blockMesh` scale factors on pitzDaily
(3 ≈ 110k, 9 ≈ 990k, 20 ≈ 4.9M, 29 ≈ 10M, 35 ≈ 15M, 54 ≈ 35.6M cells).

## Adding the GPU-vs-GPU back-ends (optional)

To also benchmark OpenFOAM offloading the pressure solve to the *same* GPU:

```bash
# OpenFOAM + AMGX  (our self-contained AMGX solver, no PETSc)
AMGX_DIR=/path/to/amgx ./setup_of_amgx.sh          # or AMGX_SRC=... CUDA_ARCH=121 to build AMGX first

# OpenFOAM + PETSc-GPU  (PETSc-GAMG on cuSPARSE, via petsc4Foam), slow build (~30 min)
./setup_of_petsc.sh
```

`run_benchmark.sh` **auto-detects** these once built (`libamgxFoam.so` / `libpetscFoam.so`) and fills in the
`OF+AMGX` / `OF+PETSc` columns.

## What's in here

| file | what |
|---|---|
| `run_benchmark.sh` | the main harness, brae vs OF-CPU (+ OF-AMGX / OF-PETSc if built) |
| `setup_of_amgx.sh` | build the OF+AMGX back-end (`amgxFoam/` solver + AMGX for your arch) |
| `setup_of_petsc.sh` | build the OF+PETSc-GPU back-end (PETSc + petsc4Foam) |
| `amgxFoam/` | self-contained OpenFOAM→AMGX solver (LDU→CSR + raw AMGX C API, no PETSc) |
| `spmv_bench.cu` | a standalone SpMV micro-benchmark (LDU vs CSR vs cuSPARSE) on a real matrix |
| `results/` | reference results measured on a single NVIDIA GB10 (20 Grace cores + one Blackwell GPU) |

## Reference results (single NVIDIA GB10: 20 Grace CPU cores + one Blackwell GPU, no HBM)

Total wall, 100 iters (`results/totalwall_100iter.txt`, `amgx_100iter.txt`, `petscgpu_100iter.txt`):

| cells | brae | OpenFOAM (20 Grace c) | OF+AMGX | OF+PETSc-GPU |
|---|---:|---:|---:|---:|
| 4.89M | 108 s | 107 s | 573 s | 602 s |
| 14.97M | 340 s | 354 s | 1696 s | 1790 s |
| 35.6M | 877 s | 990 s | 4111 s | - |

- **~5× faster than OpenFOAM's best GPU offload (AMGX)** on the same GPU (full residency vs per-iteration migration).
- **Parity-to-ahead of a 20-core Grace CPU node**, widening with mesh size. GB10 has no HBM, so its Blackwell GPU
  shares the Grace CPU's bandwidth, a conservative baseline; on an HBM GPU (H100/GH200) we expect an
  order-of-magnitude speedup, roughly 8-12× a CPU node.

Full write-up + the crossover chart: `results/BENCHMARK_TOTALWALL.md`, `results/crossover.csv`,
`results/crossover_chart.png`.

## Notes & caveats

- **Wall-clock has clock variance**, the harness runs each back-end back-to-back per size (a controlled A/B); run
  it a couple of times and compare.
- **GPU arch matters for the OF-GPU back-ends.** AMGX and PETSc must be built for *your* GPU (see the setup scripts).
  On brand-new archs (e.g. GB10 sm_121) AMGX's classical AMG is broken, the `amgxFoam` solver defaults to
  aggregation. The full arch/MPI trap list is in `results/OF_GPU_SETUP_GB10.md`.
- The scaled-laminar pitzDaily is a **timing** workload (identical work per iteration on all back-ends); it's an
  unsteady flow, so don't read its iteration-N field as a converged solution. Brae's field accuracy is validated
  separately in `../validation/`.
