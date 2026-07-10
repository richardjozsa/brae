# brae on H100, GPU-vs-GPU CFD benchmark

A same-hardware comparison of the [brae](https://github.com/simd-ai/brae) device-resident CFD engine against the
other common ways to run OpenFOAM-class CFD on a GPU, all on **one NVIDIA H100 PCIe (80 GB)**, on an identical case
with identical schemes, solver tolerances, and convergence criteria. Every number below was measured on this machine.
Two figures are labelled as *derived* (effective HBM bandwidth in §5, cost in §6); nothing else is estimated. Blank
cells mark a runner that failed or was not measured, with the reason stated.

This is the H100 counterpart to the earlier GB10 study (`bench/results/BENCHMARK_TOTALWALL.md`). GB10 has no HBM (its
GPU shares the CPU's LPDDR5X), so there brae only reached parity with a CPU node. This report tests the follow-up
question, whether the same code delivers an order-of-magnitude speedup on a GPU with real HBM.

## Headline result

Per-SIMPLE-iteration wall time at 4.89 M cells (laminar scaled pitzDaily), lower is better:

| runner | ms / iteration | brae is |
|---|---:|---:|
| **brae**, device-resident | **263.5** | - |
| OpenFOAM, 24 CPU cores (reference) | 647.7 | 2.5× faster |
| SPUMA, device-resident OpenFOAM-GPU port | 1022.2 | 3.9× faster |
| OpenFOAM + AMGX, GPU offload | 6795.7 | 25.8× faster |
| OpenFOAM + PETSc-GPU, GPU offload | 7897.1 | 30.0× faster |

brae is fastest at every mesh size tested (440 k → 28 M cells), at matched accuracy (< 1 % on U and p for all
runners). The reason: brae keeps the entire SIMPLE loop on the GPU and its sparse matrix-vector kernel reaches **85 %
of the card's HBM bandwidth**, while the linear-solve offloads leave the GPU idle most of the time (the matrix is
rebuilt on one CPU core and copied to the GPU every iteration).

---

## 1. Machine and software

| | |
|---|---|
| GPU | NVIDIA H100 PCIe, 80 GB (81559 MiB HBM2e), compute capability sm_90, 350 W, PCIe Gen5 |
| GPU memory bandwidth | ~2.0 TB/s (H100 **PCIe** HBM2e, the PCIe part, not the 3.35 TB/s SXM) |
| NVIDIA driver | 580.126.20 (CUDA driver 13.0) |
| CUDA toolkit (nvcc) | 12.6 (all GPU code built for sm_90) |
| CPU | AMD EPYC 9334, 24 vCPU (1 thread/core), KVM guest |
| System memory | 235 GiB |
| OS / kernel | Ubuntu 24.04.4 LTS / 6.8.0-106 |
| Date | 2026-07-08 |

| component | version | build notes |
|---|---|---|
| brae | commit `be6b538` | prebuilt, sm_90; default device-resident fast path (`BRAE_PCG_DEVICE=1 BRAE_AMG_FP32=1`) |
| OpenFOAM | ESI v2412 (2412.260127) | apt from dl.openfoam.com; system Open MPI 4.1.6 |
| PETSc | 3.25.3 (git 5d1858d) | `--with-cuda`, sm_90, built with OpenFOAM's mpicc; links cuSPARSE/cuBLAS |
| petsc4Foam | external-solver (git 090b5a7) | links the system libmpi.so.40 (matches OpenFOAM) |
| AMGX | 2.5.0 (git 3188dce) | sm_90; used via the in-repo `amgxFoam` LDU→CSR + AMGX C-API solver |
| SPUMA | git `14c47bd` | CINECA/EU-exaFOAM OpenFOAM-GPU port; built with nvc++ 24.11 (HPC SDK 24.11), `NVARCH=90` |

---

## 2. Method

Two cases are used, because one scaled case cannot serve both goals:

**Accuracy case**, the stock turbulent pitzDaily tutorial (kEpsilon, 12 225 cells), unchanged. It converges to a
steady field. Every runner executes a fixed 2 000 SIMPLE iterations (residualControl removed, so all stop at the same
iteration with the field settled), and each runner's U / p / k field is compared cell-by-cell to brae's (relative L2
norm), with a < 1 % pass bar.

**Performance case**, laminar scaled pitzDaily at fixed iteration count. Laminar is required for scaling: refining a
kEpsilon *wall-function* mesh pushes y+ out of the log-law range, so the wall functions break and the case never
converges (at 440 k cells OpenFOAM-CPU limit-cycles indefinitely). Laminar has no wall functions, so the mesh refines
cleanly and every runner does identical work per iteration, a clean basis for timing, memory, and bandwidth.
residualControl is removed so all runners execute exactly the same iteration count. This matches the harness in
`bench/run_benchmark.sh`.

Mesh sizes follow the blockMesh scale factor M on pitzDaily: cells ≈ 12 225 · M². Partition/decompose is done once and
excluded from every timed figure. The pressure tolerance is 1e-6 / relTol 0.1 for every runner.

**Runner configurations:**
- **brae**, 1 GPU, fully device-resident (mesh, fields, SIMPLE loop, and all linear solves on the GPU).
- **OpenFOAM-CPU**, `mpirun -np 24 simpleFoam -parallel`, native GAMG on 24 cores. Host-CPU reference only, not a
  cost-matched comparison.
- **OpenFOAM + PETSc-GPU**, 1 CPU core + GPU; PETSc CG + GAMG on cuSPARSE (`mat_type aijcusparse`, `pc_type gamg`,
  `-use_gpu_aware_mpi 0`) via petsc4Foam. Offloads only the pressure solve.
- **OpenFOAM + AMGX**, 1 CPU core + GPU; the in-repo `amgxFoam` solver (LDU→CSR + AMGX C API), PCG + aggregation AMG.
  Offloads only the pressure solve.
- **SPUMA**, the whole SIMPLE loop on the GPU via generic wrapper kernels on unified (managed) memory, run as
  `simpleFoam -pool fixedSizeMemoryPool -poolSize <GB>` with SPUMA's `twoStageGaussSeidel` GPU smoothers (required on
  a discrete GPU, see §5; the default `GaussSeidel` runs on the CPU and thrashes managed memory over PCIe).

---

## 3. Accuracy, every runner matches brae to < 1 %

Fixed 2 000 iterations, stock turbulent pitzDaily (12 225 cells), internal-field relative L2 versus brae:

| runner | p | U | k | result |
|---|---|---|---|---|
| brae | reference | reference | reference | converges at iteration 224 |
| OpenFOAM-CPU (GAMG) | 0.567 % | 0.365 % | 0.240 % | pass |
| OpenFOAM + PETSc-GPU | 0.567 % | 0.365 % | 0.239 % | pass |
| OpenFOAM + AMGX | 0.567 % | 0.365 % | 0.240 % | pass |
| SPUMA | 0.567 % | 0.365 % | 0.240 % | pass |

The four OpenFOAM-based runners are identical to one another because they share OpenFOAM's outer SIMPLE loop and
differ only in the pressure linear solver. brae is an independent GPU reimplementation and still lands within 0.57 %
on p and 0.37 % on U. The pointwise maximum difference on p (~17 %) is a single cell at the backward-step pressure
singularity, where brae loops over cells and OpenFOAM loops over faces, so the truncation error lands slightly
differently; the field-wide L2 norm is the meaningful measure and passes comfortably.

---

## 4. Performance

Laminar scaled pitzDaily, fixed iterations, partition excluded. Reproduce with `./run_benchmark.sh`.

![Per-iteration wall time versus mesh size on one H100, log-log. brae (green) is lowest at every size; OpenFOAM-CPU (grey) and SPUMA (blue) form a middle tier; the PETSc and AMGX GPU offloads (red, orange) are 7-8x above SPUMA. brae's 35.6M-cell out-of-memory point is marked.](perf_chart.png)

**Wall time per SIMPLE iteration (ms):**

| cells | brae | OpenFOAM-CPU (24c) | SPUMA | OpenFOAM+AMGX | OpenFOAM+PETSc-GPU |
|---|---:|---:|---:|---:|---:|
| 440 k | **26.2** | 47.0 | 164.7 | 366.3 | 346.9 |
| 2.07 M | **97.4** | 222.4 | 453.4 | 1969.6 | 2311.3 |
| 4.89 M | **263.5** | 647.7 | 1022.2 | 6795.7 | 7897.1 |
| 9.58 M | **733.9** | 1131.9 | - | - | - |
| 14.98 M | **1079.1** | 2346.5 | 2945.4 | 20738.0 | 24520.9 |

(PETSc/AMGX/SPUMA were not run at 9.58 M; that point is bracketed by 4.9 M and 15 M.)

**How much faster brae is, on the same GPU / same host:**

| cells | vs PETSc-GPU | vs AMGX | vs SPUMA | vs CPU (24c) |
|---|---:|---:|---:|---:|
| 440 k | 13.2× | 14.0× | 6.3× | 1.8× |
| 2.07 M | 23.7× | 20.2× | 4.7× | 2.3× |
| 4.89 M | 30.0× | 25.8× | 3.9× | 2.5× |
| 14.98 M | 22.7× | 19.2× | 2.7× | 2.2× |

There are two clear tiers. The device-resident engines (brae, SPUMA) and the CPU sit at the bottom of the chart; the
linear-solve offloads (PETSc, AMGX) sit 7-8× above SPUMA and are slower than the CPU alone, because they run only the
pressure solve on the GPU and copy a freshly assembled matrix from one CPU core every iteration. SPUMA runs the whole
loop on the GPU and so clears the offloads by 2-8×, trailing only brae. Against the offloads brae is 13-30× faster
(the ratio peaks near 4.9 M, then eases to ~20× at 15 M as brae's multigrid hierarchy deepens while the offloads scale
linearly). Against SPUMA brae is 2.7-6.3× faster, and that gap narrows as the mesh grows. On GB10 (no HBM) the same
GPU-vs-GPU comparison was about 5×; on the H100's HBM it is 20-30×.

**Peak GPU memory and the device-resident ceiling.** brae's footprint is roughly linear at ~2.8 KB/cell (1833 / 6171
/ 14011 / 41479 MiB at 440 k / 2.07 M / 4.89 M / 14.98 M). The offloads keep the mesh in host RAM and put only the
linear system on the GPU (PETSc 963 / 2495 / 5071 MiB, AMGX 1353 / 2251 / 2627 MiB at the first three sizes). SPUMA's
figure is dominated by its pre-allocated pool and its managed pages migrate, so it is not directly comparable. brae's
80 GB budget caps the mesh at about 28 M cells:

| cells | brae peak GPU memory | result |
|---|---:|---|
| 14.98 M | 40.5 GB | ok |
| 21.56 M | 57.9 GB | ok |
| 28.17 M | 75.2 GB | ok (largest that fits) |
| 35.65 M | 79.2 GB, then exit 1 | out of memory |

---

## 5. Why brae is faster, HBM bandwidth

The question is which approaches actually use the H100's HBM. Measured with `nsys` (kernel mix, memory traffic) and
`ncu` (per-kernel HBM throughput) on the 2.07 M-cell case. Two factors multiply: how much of the wall-clock the GPU is
busy, and how much of HBM's bandwidth each kernel reaches when it runs.

**GPU-active fraction (nsys: kernel + on-GPU memcpy time ÷ wall):**

| runner | GPU-active |
|---|---:|
| brae @ 4.89 M | 84.0 % |
| brae @ 14.98 M | 78.6 % |
| PETSc / AMGX offloads | GPU-starved, CPU-bound run, GPU idle the large majority of the time |

brae keeps the whole loop on the device (its pressure solve is captured into a CUDA graph), so it stays ~80 %
GPU-active across mesh sizes. The offloads are the genuine idle case: the run is dominated by one CPU core assembling
the matrix and by the per-iteration host→device copy, which is corroborated by their being slower than the 24-core
CPU. (Note: `nvidia-smi utilization.gpu` is a coarse sampled flag and mis-reads brae's few-large-kernel pattern; the
nsys kernel-time-over-wall figure above is the reliable measure.)

**Per-kernel HBM throughput of the pressure solve (ncu, % of the ~2.0 TB/s peak):**

| kernel | runner | % of HBM peak | GB/s |
|---|---|---:|---:|
| `amulKernel`, LDU SpMV (FP64) | brae | 84.9 % | 1731 |
| `gsColorT`, colored Gauss-Seidel smoother | brae | 65.1 % | 1326 |
| `cusparse::csrmv_v3`, fine-grid SpMV | AMGX | 60.3 % | 1227 |
| `cuda::lambdaKernel`, generic wrapper (mean) | SPUMA | 6.7 % (max 75.5 %) | ~140 |
| `cusparse::csrmv_v3`, SpMV (coarse-grid sample) | PETSc | 4-16 % | 88 |

brae's SpMV reaches 1731 GB/s, 85 % of this card's HBM peak, which also confirms the peak (1731 / 0.849 ≈ 2040 GB/s,
the ~2 TB/s PCIe part). At ~80 % GPU-active and ~0.65 time-weighted average kernel bandwidth, brae realizes roughly
half of the H100's total HBM bandwidth over a full run (derived estimate); the offloads realize about 1-2 %. That
order-of-magnitude gap in realized bandwidth is the mechanism behind the wall-clock gap.

**Where the offloads' GPU time goes.** nsys shows the offloads spend much of their small GPU time rebuilding the AMG
hierarchy every iteration rather than solving: AMGX's aggregation setup (`fill_A`, `compute_sparsity`,
`findStrongestNeighbour`) is ~21 % of its GPU kernel time, and PETSc's GAMG shows device radix-sort / scan / `csrgeam`
CSR-assembly kernels, because a newly assembled matrix arrives from the host each iteration. brae assembles and keeps
its matrix and hierarchy resident, so it pays that cost once.

**Why brae leads SPUMA, though both are device-resident.** SPUMA runs the loop through one generic lambda wrapper
launched about 1600 times per iteration; most launches are small and latency-bound (mean 7 % of HBM), a few reach
76 %. brae instead fuses the work into a few large kernels that each saturate HBM (SpMV at 85 %). SPUMA's continuous
stream of tiny kernels is also why coarse `nvidia-smi` sampling reported it as "more utilized", it is busy, but with
low-efficiency work, which is why it is 3-6× slower than brae.

**SPUMA configuration note (discrete-GPU requirement).** SPUMA's default GAMG smoother, `GaussSeidel`, is sequential
and runs on the CPU. On a discrete GPU that drags the managed pressure/velocity fields back over PCIe every iteration:
measured with nsys on 440 k over 8 iterations, `GaussSeidel` moved 24.4 GB (12.2 HtoD + 12.2 DtoH) with 526 k GPU page
faults, the top CPU page-fault site being `Foam::GaussSeidelSmoother::scalarSmooth`. Switching to SPUMA's own
`twoStageGaussSeidel` / `twoStageSymGaussSeidel` GPU smoothers cut migration to ~1.7 GB and made SPUMA 3.4× faster
(440 k: 779 → 165 ms/iteration). All SPUMA numbers here use the GPU smoothers. This is harmless on GB10's unified
memory (CPU access to managed memory is free there) but essential on a discrete GPU.

---

## 6. Cost, $ per 100 SIMPLE iterations at $3/h

Cost per 100 iterations = per-iteration wall × 100 × ($3 / 3600 s). A steady simpleFoam run typically needs a few
hundred iterations, so a converged run costs 2-5× these figures, identically for every runner, so the ratios are the
point.

| cells | brae | OpenFOAM-CPU (24c) | SPUMA | OpenFOAM+AMGX | OpenFOAM+PETSc-GPU |
|---|---:|---:|---:|---:|---:|
| 440 k | **$0.0022** | $0.0039 | $0.0137 | $0.0305 | $0.0289 |
| 2.07 M | **$0.0081** | $0.0185 | $0.0378 | $0.1641 | $0.1926 |
| 4.89 M | **$0.0220** | $0.0540 | $0.0852 | $0.5663 | $0.6581 |
| 14.98 M | **$0.0899** | $0.1955 | $0.2454 | $1.728 | $2.043 |

At 4.9 M a run costs about 30× less on brae than on OpenFOAM+PETSc-GPU, 26× less than on OpenFOAM+AMGX, 3.9× less than
on SPUMA, and 2.5× less than on all 24 CPU cores. The linear-solve offloads cost roughly 12× more per run than the CPU
alone ($0.66 vs $0.054 at 4.9 M): at 1-5 % duty cycle the GPU is a net cost increase over the CPU for this workload.
Only the device-resident engines make the H100 pay off.

---

## 7. Caveats

- H100 **PCIe** (~2 TB/s HBM2e), not SXM (~3.35 TB/s). The ~2 TB/s peak is corroborated by ncu (brae's SpMV at 1731
  GB/s = 84.9 %).
- OpenFOAM-CPU is a host-CPU reference, not a cost-matched comparison. It runs on 24 vCPUs of a KVM guest; a bare-metal
  EPYC would be faster, so the ~2.2-2.5× brae-vs-CPU margin is indicative, not exact.
- The performance case is laminar and physically unsteady, so it is a timing workload (identical work per iteration),
  not a field-comparison case. Field accuracy is validated separately on the converged turbulent case (§3).
- The GPU-active figures come from a clean idle-machine sweep. The ncu/nsys per-kernel runs were taken while another
  build occupied the CPU, but per-kernel HBM throughput and kernel/memcpy counts are GPU-side and independent of CPU
  load; no wall-clock figure is taken from those runs. The PETSc SpMV percentage is a coarse-grid-dominated sample and
  is treated as a low bound.
- The offloads use single-GPU, single-rank (one CPU core driving the GPU), the standard petsc4Foam/AMGX setup.
  Multi-rank decomposition would add CPU parallelism to assembly but also inter-rank halo copies; it does not change
  the finding that only the pressure solve runs on the GPU.
- nvcc is 12.6 while the driver exposes CUDA 13.0; all GPU code is built for sm_90 with CUDA 12.6.

---

## 8. How to reproduce

One script prints the per-iteration times and the brae speedup ratios. It builds each mesh in a scratch dir, times
the run, and deletes it, no output data is kept.

```bash
# 1. build brae (see the repo README) and source OpenFOAM v2412:
source /usr/lib/openfoam/openfoam2412/etc/bashrc

# 2. (optional) build the GPU offloads / SPUMA so their columns appear, all for sm_90:
../setup_of_petsc.sh      # OpenFOAM + PETSc-GPU  -> libpetscFoam.so
../setup_of_amgx.sh       # OpenFOAM + AMGX       -> libamgxFoam.so
../setup_spuma.sh         # SPUMA (needs nvc++ / NVIDIA HPC SDK)

# 3. run the benchmark (brae + OpenFOAM-CPU always; PETSc/AMGX auto-detected; SPUMA if SPUMA_BIN is set):
cd bench/H100
./run_benchmark.sh
#   knobs:  SIZES="6 13 20"  ITERS=100  CORES=24
#           SPUMA_BIN=/path/to/spuma/platforms/linux64NvidiaDPInt32Opt/bin/simpleFoam  SPUMA_POOL=24
```

`SIZES` are blockMesh scale factors on pitzDaily (cells ≈ 12 225 · M²): 6 ≈ 440 k, 13 ≈ 2 M, 20 ≈ 4.9 M, 35 ≈ 15 M.
The script prints the two tables in §4 (ms per iteration, and how many times faster brae is).
