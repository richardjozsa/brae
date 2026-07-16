# Multi-GPU support: phased implementation plan

How brae extends from single-GPU to multi-GPU (single-node) and multi-node clusters, so a mesh larger than
one GPU's VRAM can be solved, and so new solvers (compressible, transient PIMPLE) reuse the same parallel
machinery instead of reinventing it.

**Status:** plan only. Nothing here is implemented. Staged in the `cudafoam` parent folder for review; it
moves to `brae/docs/multi-gpu.md` when the work starts, on the `feat/multi-gpu-nvshmem` branch.

---

# Part A -- Design principles (the "why")

Four decisions frame every phase. Detail is condensed here; the phases in Part B are the actionable core.

**A1. One seam, N solvers.** brae mirrors OpenFOAM's separation: the solver assembles an LDU system; the
parallel foundation solves it, blind to the physics. So the compressible/PIMPLE solvers later add only
*assembly + outer loop* and reuse the whole foundation. The seam is `Pstream` (transport) + `lduInterface`
(`ProcessorInterface`). Everything below it operates on an assembled LDU matrix + a partitioned mesh.

```
   PER-SOLVER   incompressible SIMPLE | compressible | pimple ...   (add one module per physics)
   ------------------------- THE SEAM: emits LDU, needs halo/reduce -------------------------
   FOUNDATION   Krylov (parallelPCG) | SpMV (parallelAmul) | ProcessorInterface | decomp
                Transport: Pstream  ->  NVSHMEM device backend (tier chosen at runtime)
```

**A2. One code path, NVSHMEM from day one.** The transport is NVSHMEM device-initiated `put`/`signal`. Chosen
because it is the only option that (a) stays *inside* brae's CUDA-graph capture (MPI host calls break
capture), and (b) auto-tiers the transport itself, so we write one code path:

| peer location | tier (NVSHMEM picks via `nvshmem_ptr`) | path |
|---|---|---|
| same GPU / NVLink / NVSwitch | P2P fast path | direct GPU load/store |
| same node, PCIe-P2P | P2P fast path | direct over PCIe (the 2xL40S case) |
| different node | IBGDA (GPU-initiated) or host proxy | over InfiniBand / RoCE |

The same binary runs single-GPU -> single-node multi-GPU -> multi-node. NVSHMEM is a multi-node PGAS library
(one symmetric heap spans all PEs across all nodes), not a single-node one.

**A3. Two structural costs we accept.** (1) `libnvshmem` becomes a build dependency, bootstrapped off the
existing MPI communicator; gated behind `BRAE_WITH_NVSHMEM` so a single-GPU checkout still compiles.
(2) Halo buffers move to NVSHMEM's symmetric heap (`nvshmem_malloc`), a second device allocator alongside
`DevicePool` -- small (O boundary faces), noted in `docs/memory-model.md` when it lands.

**A4. Reductions are a separate, later lever.** The global all-reduce in PCG is a *numerics* choice (pipelined
/ s-step CG), orthogonal to the transport. Deferred to Phase 5, done only if profiling shows it hot.

---

# Part B -- The phases

Each phase is an independently testable unit with its own exit gate. The ordering front-loads all correctness
risk onto GB10 (cheap, 2 ranks on 1 GPU) before spending on the 2xL40S instance.

Machines used (neither GB10 nor L40S has NVLink -- see Part C):

| box | role |
|---|---|
| **GB10** (single GPU, unified mem) | Phases 0-3 correctness, via 2+ ranks on the one GPU; graph capture |
| **2xL40S** (PCIe, no NVLink) | Phase 4-5 true multi-GPU; P2P-vs-proxy fallback; scaling trend |
| rented NVLink (2xA100 / 2x3090) | Phase 5 absolute perf (the 1.5-2x wins) |
| IB cluster (rented) | Phase 6 the IBGDA/network tier |

### Starting line: the host-distributed path already exists

This is not a from-scratch build. The solver-agnostic distributed foundation is already implemented on the
**host (MPI)** and tested at np = 1 / 2 / 4 / 8:

- `processor_interface.cu` -- the halo exchange (gather `psi[faceCells]` -> `Pstream::isend/irecv` -> scatter
  `result -= coeff*psiNbr`), on `std::vector` buffers.
- `scotch_decomposition.cu`, `local_mesh`, `domain_decomposition.cu` -- decompose + `buildLocalMesh` +
  reconstruction, tested by `test_decompose_exchange`, `test_local_mesh`, `test_proc_delta`.
- `ldu_spmv` / `pcg` / `pbicgstab` distributed drivers -- tested by `test_parallel_spmv`, `test_parallel_pcg`,
  `test_parallel_pbicgstab`, `test_parallel_poisson`.
- Whole distributed SIMPLE + turbulence -- `test_parallel_simplestep`, `test_parallel_loop`,
  `test_parallel_turbloop`, and the `brae_simpleFoam` MPI app.

So the algorithm and the seam are proven. **Multi-GPU work = replace the transport behind the seam
(host-MPI std::vector -> device-NVSHMEM symmetric heap) and connect it to the device-resident solver
(`device_spmv` / `device_pcg` / `device_simple`), which today is single-GPU.** The existing host tests become
the bit-for-bit oracle the device path must match -- exactly what the `test_interface_exchange.cu` header
already anticipates.

### Phase 0a -- Rank <-> GPU binding + launch

- **Status: DONE** (`test_gpu_bind`, np=1/2/4/8 green on GB10).
- **Goal:** N ranks launch under `mpirun`, each bound to a GPU, with the single-GPU path unchanged at N=1.
- **Build:**
  - `Pstream::init` binds rank -> GPU with `cudaSetDevice(rank % nGpuVisible)`. On GB10 all ranks -> GPU 0; on
    L40S rank i -> GPU i. Log the binding so topology is visible.
  - The `brae` device app accepts `mpirun -np N` (the entry already exists for `brae_simpleFoam`).
- **Files:** `cf_pstream.cu` (device binding in `init`), `gpuSimpleFoam.cu` (accept the N-rank launch).
- **Machine:** GB10, N ranks / 1 GPU.
- **Exit gate:** N=1 device run byte-identical to today's single-GPU run; each rank reports its bound device;
  no double-binding / context errors at N=2/4/8.

### Phase 0b -- Per-rank DeviceMesh upload + device reconstruction

- **Status: DONE** (`test_gpu_decompose`, np=1/2/4/8 green; maxLocalCells 12225->6126->3078->1539 as np 1->2->4->8, the OOM signal).
- **Goal:** each rank uploads only *its* partition to its GPU, and reconstruction works on the device path.
  Decomposition + host reconstruction already exist (`test_decompose_exchange` / `test_local_mesh`); the new
  bit is the per-rank `DeviceMesh` upload. This is the piece that fixes OOM -- no GPU holds the whole mesh.
- **Build:**
  - Feed each rank's existing `LocalMesh` into the `DeviceMesh` upload (per partition, not the global mesh).
  - Reconstruction on the device path: D2H the local field, then reuse the existing `cellProcAddr` gather;
    master writes one global field.
- **Files:** `device_mesh.cuh` (upload a `LocalMesh`), a device-path reconstruct helper. Decomposition
  (`scotch_decomposition.cu`, `local_mesh`) is reused as-is.
- **Machine:** GB10, 2 ranks / 1 GPU.
- **Exit gate:** N=2 device round-trips a field (partition -> upload -> D2H -> gather -> compare) exactly
  against the host `test_decompose_exchange` oracle; peak VRAM per rank scales with the partition, not the
  global mesh (the measurable OOM-relief signal).

### Phase 1a -- NVSHMEM build wiring + bootstrap + symmetric heap

- **Status: DONE** (`test_nvshmem_smoke`, np=1/2/4 green; builds ON and OFF). Implementation notes learned:
  NVSHMEM 3.6.5 ships in the HPC SDK (aarch64) with the MPI bootstrap plugin + IBGDA/IBRC/UCX. Phase 1a is
  host-only: include `<nvshmem_host.h>` (NOT `<nvshmem.h>`, which pulls device defines needing -rdc), and call
  the exported `nvshmemx_hostlib_init_attr` / `nvshmemx_hostlib_finalize` (the inline `nvshmemx_init_attr`
  resolves a device-lib-only symbol). Symmetric buffers MUST be freed before `nvshmem_finalize` (SymBuffer
  lifetime < Pstream::finalize) -- a real ordering constraint the interface teardown must honour.
- **Goal:** brae builds and links NVSHMEM behind `BRAE_WITH_NVSHMEM`, bootstraps it off the MPI communicator,
  and can allocate symmetric-heap buffers. No halo exchange yet -- this is the substrate Phase 1b sits on.

- **Build (file by file):**

  1. **Build wiring (`CMakeLists.txt`).**
     - Add `option(BRAE_WITH_NVSHMEM "device NVSHMEM transport" ON)`; `find_package(NVSHMEM)` (or find
       `libnvshmem_host` + `libnvshmem_device` + headers) guarded by it.
     - NVSHMEM device calls need **relocatable device code**: set
       `CMAKE_CUDA_SEPARABLE_COMPILATION ON` and `-rdc=true` on `brae_core` (and any target that launches a
       kernel calling `nvshmem_*`), then device-link against `nvshmem_device`; link `nvshmem_host` on the host
       side. Define `-DBRAE_WITH_NVSHMEM` for the code paths.
     - Note in the file: with NVSHMEM off, the interface falls back to the existing MPI backend, so single-GPU
       CI still builds.

  2. **Symmetric-heap allocator (new `src/parallel/pstream/sym_buffer.cuh`).**
     - A small RAII wrapper mirroring `DeviceBuffer<T>` but backed by `nvshmem_malloc` / `nvshmem_free`
       instead of `DevicePool` -- because only symmetric-heap memory is a legal NVSHMEM `put` target. Same
       `data()/size()/copyFrom/copyTo` surface so call sites look identical.
     - Keep it separate from `DeviceBuffer` (the pool) on purpose: internal work stays in the pool; only the
       comm buffers live in the symmetric heap. (This is the second-allocator note for `docs/memory-model.md`.)

  3. **NVSHMEM bootstrap (`cf_pstream.cu` / `cf_pstream.cuh`).**
     - In `Pstream::init`, after `MPI_Init`, call `nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, ...)` with
       `MPI_COMM_WORLD` so PE ids line up with MPI ranks. In `finalize`, `nvshmem_finalize()` before
       `MPI_Finalize`. All under `#ifdef BRAE_WITH_NVSHMEM`.
     - Add device reduction helpers later (Phase 3); Phase 1a only needs init/finalize + the heap.

- **Files:** `CMakeLists.txt` (NVSHMEM find/link, `-rdc=true`, flag), new `src/parallel/pstream/sym_buffer.cuh`,
  `cf_pstream.cu` / `cf_pstream.cuh` (bootstrap).
- **Machine:** GB10.
- **Exit gate:** builds and links with `BRAE_WITH_NVSHMEM` ON *and* OFF (OFF falls back to the MPI backend so
  single-GPU CI still compiles); `nvshmem_init`/`finalize` run clean; a trivial 2-PE symmetric-heap `put`
  microbench delivers the value on GB10.
- **Risks / notes:** `-rdc=true` + device-link is the step most likely to fight the existing build; get a
  one-kernel NVSHMEM smoke target linking before touching `brae_core`.

### Phase 1b -- Device halo backend + halo test

- **Status: DONE for correctness** (`test_gpu_interface_exchange`, np=1/2/4/8 green; matches the host
  `test_interface_exchange` oracle bit-for-bit -- neighbour value delivery AND the +/-1 stencil). Implemented as
  `DeviceHalo` (`device_halo.cuh/.cu`), a per-rank manager of all interfaces (one symmetric send/recv pair, one
  barrier) -- the device counterpart of the host init-all/wait/update-all loop. **Key implementation decision:**
  the transport is the NVSHMEM **on-stream host API** (`nvshmemx_putmem_on_stream` + `nvshmemx_barrier_all_on_stream`)
  with plain CUDA pack/scatter kernels, so NOTHING needs `-rdc`/device-link. Symmetric buffers are sized to the
  global max total faces (collective malloc must be symmetric); a one-time MPI offset exchange tells each
  neighbour where in our recv buffer to write.
- **GRAPH-CAPTURE FINDING (moves to Phase 5):** the on-stream barrier is **not** CUDA-graph-capturable -- it
  fails fatally with "operation not permitted when stream is capturing". So the on-stream exchange is
  stream-ordered but cannot sit inside a captured graph. A fully in-graph exchange needs the **device-initiated
  put+signal** path (`nvshmemx_*_put_signal_nbi` + device signal wait inside a kernel), which DOES need
  `-rdc`/device-link -- deferred to Phase 5 as the performance + graph-replay upgrade behind this same
  `DeviceHalo` interface.
- **Goal (original):** a device-resident halo backend that exchanges halo values GPU-to-GPU via NVSHMEM and
  reproduces `test_interface_exchange` (host MPI) bit-for-bit.

- **Build (file by file):**

  1. **Device interface backend (`processor_interface.cuh` / `.cu`, or a sibling
     `device_processor_interface`).** Give `ProcessorInterface` a device path holding:
     - `DeviceBuffer<label> faceCells_d_` (the gather index, uploaded once),
     - `SymBuffer<scalar> sendBuf_d_`, `recvBuf_d_` (symmetric heap, size = #interface faces),
     - a signal word in the symmetric heap for `put_signal` completion.
     - `initInterfaceMatrixUpdate(const scalar* psi_d)` -> launch a **pack kernel**
       (`sendBuf_d_[f] = psi_d[faceCells_d_[f]]`) then `nvshmemx_double_put_signal_nbi` to the neighbour PE's
       `recvBuf_d_` (device-initiated, so it lives inside the kernel / on the capture stream).
     - `updateInterfaceMatrix(scalar* result_d, const scalar* coeffs_d)` -> after the signal wait, a **scatter
       kernel** doing `result_d[faceCells_d_[f]] -= coeffs_d[f] * recvBuf_d_[f]` -- the exact arithmetic the
       host `updateInterfaceMatrix` does today (`processor_interface.cu:41`), just on device.
     - Preserve the init-all / wait / update-all ordering the host path and `parallelAmul` already rely on.

  2. **Device halo test (new `tests/test_gpu_interface_exchange.cu`).** Port `test_interface_exchange.cu`: same
     1-D chain, `psi[cell] = global index`, but `psi` is a `DeviceBuffer`; assert the device `neighbourField`
     equals the neighbour's global boundary index and the stencil matches -- i.e. **equals the host oracle
     bit-for-bit**. Register at np = 1/2/4/8 with `--oversubscribe`, mirroring the existing entries.

  3. **Graph-capture check (extend the test).** Wrap the pack+put+wait+scatter in a `cudaGraph` capture +
     replay and assert the replayed result matches the eager result -- proving the transport survives brae's
     replay model (the reason NVSHMEM was chosen over MPI).

- **Machine:** GB10, 2 ranks / 1 GPU (NVSHMEM allows multiple PEs per GPU; the P2P self/loopback path
  exercises the full pack -> put -> scatter plumbing without a second card).
- **Exit gate (met):** `test_gpu_interface_exchange` passes at np = 1/2/4/8 and matches the host
  `test_interface_exchange` exactly. (Graph-capture of the exchange re-scoped to Phase 5 -- see the finding
  above.)
- **Risks / notes:** the put target must be in the symmetric heap (Phase 1a); on GB10 the "peer" is the same
  GPU, so this validates plumbing but not true P2P bandwidth (that arrives on 2xL40S in Phase 4a).
  Symmetric buffers must be freed before `nvshmem_finalize` (tear the `DeviceHalo` down before
  `Pstream::finalize()`).

### Phase 2 -- Distributed device SpMV

- **Status: DONE** (`test_gpu_parallel_spmv`, np=1/2/4/8 green; device distributed A*psi == serial oracle at
  rel 3.7e-15). Implemented `deviceParallelAmul` (`device_spmv.cu`): `halo.postExchange` -> local `deviceAmul`
  (overlaps) -> `halo.waitExchange` -> per-interface `updateInterfaceMatrix` scatter, mirroring host
  `parallelAmul`. **Two bugs found and fixed (both matter for Phases 3-4):** (1) NVSHMEM on-stream ops must be
  given `cudaStreamPerThread` explicitly -- a bare `0` is the LEGACY default stream to the precompiled NVSHMEM
  lib, a different stream from brae's per-thread-default kernels, so the put raced the pack; (2) the interface
  scatter must use `atomicAdd` -- a cell can own several cut faces on one interface, so distinct threads target
  the same result cell (same reason `cyclicAmulKernel` is atomic). Both surfaced only at np>=4.
- **Goal:** bring the halo-overlapped LDU product to the **device** SpMV. The host version (`ldu_spmv` +
  `parallelAmul`, tested by `test_parallel_spmv`) is the oracle; the new part is running it on `device_spmv`
  with the Phase 1 device interface. The lever: `deviceAmul` *already* does exactly this pattern for cyclic
  (`cyclicAmulKernel`) and AMI (`amiAmulKernel`) interfaces -- a processor interface is the same off-diagonal
  pass, with the neighbour `psi` coming from the halo `recvBuf` instead of a local index.

- **Build (file by file):**

  1. **`device_ldu.cuh` (`DeviceLduView`).** Add the processor-interface handle(s) alongside the existing
     `cyc*` / `ami*` fields -- a list of device interfaces (Phase 1) with their `faceCells_d`, `coeffs_d`, and
     symmetric-heap buffers. Same shape as the cyclic/AMI extension already there.

  2. **`device_spmv.cu` -- split `deviceAmul` for overlap.** Today `deviceAmul` launches `amulKernel` then the
     cyclic/AMI kernels synchronously. Add a parallel-aware path that preserves the host ordering:
     - `for each iface: iface.initInterfaceMatrixUpdate(psi.data())` -- pack + `nvshmem put_signal` (Phase 1),
       posted first so it overlaps the local product;
     - `amulKernel<<<>>>` (local diag + upper/lower) -- runs while the halo is in flight;
     - cyclic/AMI kernels as today (intra-GPU, no comms);
     - wait on the interface signals;
     - `for each iface: iface.updateInterfaceMatrix(Apsi.data(), coeffs_d)` -- the scatter kernel
       `Apsi[faceCell] -= ifCoeff * recvBuf[f]` from Phase 1.
     Keep the existing single-GPU `deviceAmul` signature working (empty interface list -> identical to today).

  3. **Test (new `tests/test_gpu_parallel_spmv.cu`).** Mirror `test_parallel_spmv`: same `kEpsCorrect` matrix,
     device fields, np = 1/2/4/8; assert device distributed `A*psi` == the host oracle and == single-GPU
     device `A*psi` after reconstruction.

- **Files:** `device_ldu.cuh` (interface fields), `device_spmv.cu` (overlapped parallel `deviceAmul`), reuse
  the Phase 1 interface; new `tests/test_gpu_parallel_spmv.cu` + CMake entries at np = 1/2/4/8.
- **Machine:** GB10, 2 ranks / 1 GPU.
- **Exit gate:** device distributed `A*psi` == host `test_parallel_spmv` result == single-GPU device `A*psi`
  on a real mesh (bit-for-bit; mind the reproducible accumulation order brae already relies on).
- **Risks / notes:** the local product overlaps the put only if the put is on the same per-thread stream and
  launched first; the interface scatter must run *after* the signal wait, not fused into `amulKernel`.

### Phase 3 -- Distributed device Krylov (the global reduction)

- **Goal:** the device solvers converge across partitions. Host `parallel_pcg` (tested by `test_parallel_pcg`,
  `test_parallel_poisson`) is the oracle. Two things change vs single-GPU: the Amul is the Phase-2 parallel
  one, and every **dot / sumMag / normFactor becomes a GLOBAL reduction**. The preconditioner stays local
  (DIC/Jacobi with interfaces dropped) exactly as the host template already does -- so only reductions and the
  SpMV cross the partition boundary.

- **The reduction has two tiers** (this is the crux of the phase):

  1. **Host-scalar solvers (`deviceJacobiPCG`).** These already surface each dot to the host via the pinned
     mailbox (`deviceDot`, `deviceSumMag`, `deviceReadScalar` -> host `scalar`). First cut: wrap each returned
     scalar in `Pstream::allReduce(v, Sum)` before it is used. Simple, correct, and enough for the L40S
     validation. It does put a blocking MPI call per reduction (breaks graph capture) -- acceptable here
     because this solver is not the graph-captured hot path.

  2. **Device-scalar / graph-captured solvers (`deviceJacobiBiCGStab`, the AMG-PCG in `device_amg`).** These
     keep alpha/omega/beta and the dots *on the device* (the `*Dev` kernels, `deviceDotInto`) precisely so the
     V-cycle can be captured as a CUDA graph. Here a host `allReduce` would defeat the purpose. Instead add a
     **device global all-reduce**: reduce locally into a symmetric-heap scalar, then `nvshmem` reduce across
     PEs, leaving the global value in device memory for the next `*Dev` kernel. Stays inside the capture.

- **Build (file by file):**

  1. **`cf_pstream` + NVSHMEM reduce wrapper.** Keep the existing host `Pstream::allReduce` (tier 1); add a
     device-resident `nvshmemReduceScalar(sym scalar in/out)` (tier 2).
  2. **`device_pcg.cu`.** Route the tier-1 dots through `Pstream::allReduce`; route `deviceJacobiBiCGStab`'s
     device scalars through the tier-2 reduce.
  3. **`device_pcg.cu :: deviceNormFactor`.** Its three reductions (`avgPsi = psi.ones / nC`, `n1`, `n2`) must
     be global: `psi.ones` and `nC` sum across ranks (global average), `n1`/`n2` sum across ranks. Feed the
     global `nC`/sums in so the normFactor matches the undecomposed value.
  4. **`device_amg`** (when it is the pressure solver): the coarse-level global sums use the tier-2 device
     reduce; the local V-cycle smoother is unchanged (like the host, coarse interfaces can be dropped).
  5. **Tests:** new `tests/test_gpu_parallel_pcg.cu` and `tests/test_gpu_parallel_poisson.cu` mirroring the
     host ones, np = 1/2/4/8; `device_pbicgstab` follows the same shape after PCG passes.

- **Files:** `device_pcg.cu`, `cf_pstream.*` (device reduce), `device_amg.cu` (coarse global sums), new
  parallel PCG/Poisson device tests.
- **Machine:** GB10, 2 ranks / 1 GPU.
- **Exit gate:** device distributed PCG (and Poisson) converge to the same solution and comparable iteration
  count as the host `test_parallel_pcg` oracle and single-GPU device PCG; the tier-2 reduce keeps the captured
  V-cycle replayable (checked in Phase 5).
- **Risks / notes:** reduction *order* differs once summed across ranks, so expect last-bit differences, not
  bit-exactness, vs single-GPU -- assert against the host distributed oracle (same partitioning) for equality,
  and against single-GPU within solver tolerance. `normFactor` is the easy thing to get subtly wrong; test it
  in isolation first.

### Phase 4a -- Laminar distributed device SIMPLE (TRUE multi-GPU, the OOM milestone)

- **Status: foundation DONE, loop IN PROGRESS.** The coupled-processor-boundary mechanism that EVERY explicit
  operator needs is built and tested: `DeviceHalo::scatterBoundaryValues` exchanges a field and writes the
  interpolated processor-face value `w*local + (1-w)*nbr` into the flattened `bval` array, so the UNCHANGED
  device operators (grad/div/interpolate) treat a processor face as a coupled boundary. Proven by
  `test_gpu_parallel_grad` (device `gaussGrad` on a decomposed mesh vs serial, np=1/2/4/8, rel 1.4e-15).
  Interpolation weights via host `computeProcWeights` (uploaded once); interface->bval mapping via `procStart`.
  **Remaining:** apply the same pattern to the other explicit operators (`deviceDiv`, interpolate, the `H()`
  assembly), wire the Phase-2/3 parallel SpMV/PCG into the momentum + pressure solves, and compose the
  distributed device SIMPLE loop with reconstruction.
- **Goal:** the **laminar** steady SIMPLE step (momentum + pressure, no turbulence) running across two
  *physical* GPUs. This is the phase that answers the original OOM question, and it is the natural cut point:
  the laminar loop closes the multi-GPU story end to end; turbulence (4b) then adds fields, not new machinery.
  The scoping insight: it is not just the linear solve that needs the halo -- every **explicit operator**
  (gradient, interpolation, the `H()` operator) reads the neighbour cell value at a processor face, so each
  needs its input field exchanged first. The host path already proves each one, operator by operator.

- **Build (file by file):**

  1. **Vector/tensor halo (extend the Phase 1 interface).** `U` and `gradU` are vector/tensor fields; add a
     vector `put` (3 / 9 components, or an interleaved buffer). A processor face is then treated as a coupled
     boundary whose neighbour value is the halo -- mirroring how `DeviceCyclic`/`DeviceAMI` already feed a
     neighbour value into the device operators.

  2. **`device_fvc.cu` -- explicit operators processor-aware.** `grad`, `interpolate`, `snGrad`, `flux`:
     exchange the input field, then at a processor face use the halo neighbour value in place of a boundary
     value. Oracle: `test_parallel_grad`, `test_proc_interp`, `test_parallel_transport`.

  3. **`device_fvm.cu` -- assembly / `H()` operator.** The momentum matrix `H` and convection assembly read
     neighbour `psi` at processor faces; exchange before assembling. Oracle: `test_parallel_matrixH`,
     `test_parallel_momentum`.

  4. **`device_simple.cu` -- the SIMPLE glue.** `H`, `rAU`, `phiHbyA`, the matrix flux, and the pressure
     corrector all cross processor faces (the corrector's face flux must be consistent on both sides). Wire the
     Phase 2/3 parallel SpMV/PCG into the momentum and pressure solves. Oracle: `test_parallel_predictor`,
     `test_parallel_simplestep`, `test_parallel_loop`.

  5. **Driver + reconstruction (`device_simple_foam.cuh`, `gpuSimpleFoam.cu`, new
     `parallel_device_simple.cuh`).** Orchestrate the distributed device loop (mirror the host
     `parallelSimpleStepLaminar` in `parallel_simple.cuh`); at the end, D2H each rank's fields and reconstruct
     via the Phase-0 `cellProcAddr` gather; master writes one global field. The `brae` app gains an
     `mpirun -np N` device path alongside the existing `brae_simpleFoam` host one.

- **Files:** the Phase-1 interface (vector variant), `device_fvc.cu`, `device_fvm.cu`, `device_simple.cu`,
  `device_simple_foam.cuh`, `gpuSimpleFoam.cu`, new `parallel_device_simple.cuh`; new end-to-end laminar
  device test.
- **Machine:** 2xL40S (GB10 still the fast per-operator correctness loop for steps 2-4). Also run 8 ranks
  oversubscribed for scale-out bugs (partitions with several processor neighbours, corner cells shared by 3+
  partitions, reduction associativity).
- **Exit gate:** a **laminar** case on 2 physical GPUs, reconstructed, matches single-GPU brae AND OpenFOAM
  within the roadmap's <1% field bar. **Plus the OOM proof:** a mesh too big for one GPU's memory runs split
  across the two.
- **Risks / notes:** the corrector face flux is the classic conservation trap -- the same processor face must
  see the same flux from both ranks; test global mass conservation explicitly. Bring operators up one at a
  time on GB10 (each has a host oracle) before the full loop on L40S.

### Phase 4b -- Turbulence over the distributed loop

- **Goal:** add the RANS models on top of the 4a laminar loop, so real turbulent cases run multi-GPU. No new
  communication machinery -- just more fields exchanged through the same interface.
- **Build (file by file):**
  1. **`device_divdevreff.cu` -- explicit stress source.** Reads neighbour gradients; exchange `U` (via the
     4a vector halo) before the stress divergence. Oracle: `test_parallel_divdevreff`.
  2. **`device_kepsilon.cu` / `device_komega_sst.cu` -- production + eddy viscosity.** The production term and
     `nut` update read neighbour gradients / field values at processor faces; exchange `k`, `eps`/`omega`,
     `nut` before each. Wire the k/eps(/omega) matrix solves through the Phase 2/3 parallel SpMV/PCG. Oracle:
     `test_parallel_kepsilon`, `test_parallel_turbloop`.
  3. **Driver.** Extend the 4a `parallel_device_simple.cuh` to the turbulent step (mirror the host
     turbulent-loop path that `test_parallel_turbloop` exercises).
- **Files:** `device_divdevreff.cu`, `device_kepsilon.cu`, `device_komega_sst.cu`, the turbulence exchange in
  `parallel_device_simple.cuh`; end-to-end turbulent device test mirroring `test_parallel_converged`.
- **Machine:** 2xL40S (GB10 for per-model correctness; 8 ranks oversubscribed for scale-out).
- **Exit gate:** a turbulent case (e.g. pitzDaily kOmegaSST) on 2 physical GPUs, reconstructed, matches
  single-GPU brae AND OpenFOAM within the roadmap's <1% field bar (near-wall within the documented ~10-15%).
- **Risks / notes:** wall functions read cell-adjacent values -- confirm a processor face never sits where a
  wall function expects a wall patch (decomposition should not split a near-wall cell from its wall face); the
  host turbulent tests already encode this, so match them first on GB10.

### Phase 5 -- Overlap, scaling, graph-replay at scale

- **Goal:** confirm the performance model brae was built on -- comms hidden under compute, and the distributed
  loop still captured-and-replayed -- and measure scaling honestly.

- **Build (file by file):**

  1. **Device-initiated halo for in-graph capture (`device_halo.cu` + a new `-rdc` TU).** Phase 1b proved the
     on-stream transport is correct but NOT graph-capturable (NVSHMEM's `nvshmemx_barrier_all_on_stream` fails
     under capture). To put the halo inside brae's captured V-cycle/PCG graph, add a device-initiated variant
     behind the same `DeviceHalo` interface: a kernel that packs and calls `nvshmemx_double_put_signal_nbi`,
     plus a device `signal_wait_until` -- both live inside kernels, so they capture. This is the TU that needs
     `CUDA_SEPARABLE_COMPILATION`/`-rdc` + device-link against `libnvshmem_device.a` (isolate it; do not make
     all of `brae_core` rdc). Then confirm this device path and the Phase-3 tier-2 device reduce sit inside the
     capture; anything on the host `Pstream::allReduce` (tier 1) must be outside the capture or promoted.
  2. **Overlap verification (nsys).** Confirm `amulKernel` (local product) actually overlaps the in-flight
     halo, and the reduction is not serializing the loop. Add a small profiling harness (mirror
     `test_gpu_benchmark` / `test_gpu_crossover`).
  3. **Scaling harness.** Strong + weak scaling over 2 (and 8 oversubscribed) ranks; record PCIe numbers on
     L40S and NVLink numbers on the rented box.
  4. **(Optional numerics) `device_pcg.cu`.** If reductions dominate at higher rank counts, evaluate a
     pipelined / s-step CG variant (fewer, overlapped global reductions). Pure Krylov-layer change; transport
     untouched.

- **Files:** `device_amg.cu` / `device_simple.cu` (capture spans comms), a profiling/scaling harness under
  `validation/perf/`, optional pipelined variant in `device_pcg.cu`.
- **Machine:** 2xL40S for the trend; a rented NVLink box (2xA100 / 2x3090) for the absolute 1.5-2x wins.
- **Exit gate:** the distributed inner solve replays from a captured graph with no host round-trip in the hot
  path; nsys shows the local product hiding the halo; scaling curve documented; NVLink wins confirmed on the
  rented box and the PCIe-vs-NVLink gap written down.
- **Risks / notes:** if any per-iteration host `allReduce` remains in the captured region, capture will either
  fail or silently exclude it -- this is the phase where the tier-1/tier-2 split from Phase 3 gets stress
  tested.

### Phase 6 -- Multi-node (optional, when needed)

- **Goal:** light up the inter-node IBGDA / network tier -- the one rung neither GB10 nor L40S exercises. No
  new solver code is expected: the same binary, one more transport tier under the seam.

- **Build (mostly ops / build config):**

  1. **NVSHMEM build with the network transport.** Rebuild NVSHMEM (and re-link `brae_core`) with the
     InfiniBand transports enabled -- IBGDA (GPU-initiated, GDA-KI) plus the IBRC/UCX fallback. Ensure
     GPUDirect RDMA is available on the nodes (driver + `nvidia-peermem`).
  2. **Cross-node bootstrap.** The `nvshmemx_init_attr(...MPI_COMM...)` path already spans nodes once launched
     under a multi-node `mpirun`/`srun`; alternatively the UID/IP bootstrap. No code change from Phase 1.
  3. **Confirm the tier.** Select the network transport (e.g. `NVSHMEM_REMOTE_TRANSPORT=ibgda`) and verify at
     runtime that `nvshmem_ptr()` returns null for the remote peer (network path, not P2P) and that IBGDA --
     not the host-proxy fallback -- is active (NVSHMEM info logging / env).
  4. **Symmetric-heap atomics.** If any reduction/signal uses device atomics over the network, ensure the
     UCX/IB atomics path is built (the same UCX note from the L40S PCIe case).

- **Files:** none new expected; build/link config for NVSHMEM IB transports; a 2-node scaling entry in the
  Phase-5 harness.
- **Machine:** an IB-connected cluster (rented).
- **Exit gate:** a 2-node run reproduces the single-node distributed result within solver tolerance; runtime
  confirms the IBGDA path is active (not host-proxy); inter-node scaling recorded.
- **Risks / notes:** this tier's failures are environmental (NIC/HCA config, GPUDirect RDMA, ACS, firmware),
  not algorithmic -- budget setup/debug time for the cluster, not for brae. Validate a trivial NVSHMEM
  put-across-nodes microbench before running the full solver.

---

# Part C -- Hardware ladder (what each rung proves)

One codebase throughout; each rung is a *test run*, not new code.

| rung | machine | proves | does NOT prove |
|---|---|---|---|
| logic | GB10, 2 ranks / 1 GPU | decomposition, halo, SpMV, Krylov correctness; graph capture | true 2-GPU; any perf |
| intra-node correctness | 2xL40S, 2 ranks | real multi-process/multi-GPU over PCIe; fallback logic | scale-out bugs; NVLink perf |
| scale-out | 2xL40S, 8 ranks oversubscribed | multi-neighbour partitions, corner cells, reduction associativity | NVLink perf; multi-node |
| perf | rented 2xA100 / 2x3090 | the 1.5-2x NVLink strong-scaling wins | multi-node IBGDA |
| network | IB cluster | the IBGDA/network tier | -- |

NVLink hardware note: NVIDIA dropped NVLink in Ada, so L40S / RTX 6000 Ada / 4090 will never have it. NVLink
lives on RTX 3090 (last consumer), RTX A6000/A5000 (Ampere), V100/A100/H100/H200/GH200/GB200. Cheapest real
NVLink for Phase 5: rent 2xA100 (or 2x RTX 3090 + bridge) by the hour rather than buy.

---

# Part D -- Open decisions

1. **Reduction transport** -- NVSHMEM all-reduce (one dependency) vs NCCL (robust drop-in). Start NVSHMEM;
   A/B against NCCL in Phase 5 if reductions are hot.
2. **Symmetric-heap sizing** -- auto-size from the partition's boundary-face count at init.
3. **Build-flag hardness** -- `BRAE_WITH_NVSHMEM` default-on with a single-GPU fallback build, vs a truly hard
   dependency with no flag. Recommendation: keep the flag (cheap CI/laptop insurance).
4. **Topology-aware placement (Tier 2)** -- optional; place heavily-communicating partitions on the
   fastest-connected GPUs. Defer; an optimization, not correctness.

---

# References

- Redesigning GROMACS Halo Exchange: GPU-initiated NVSHMEM (SC'25) -- https://arxiv.org/abs/2509.21527
- A Practical Guide to GPU-Initiated Communication (NVIDIA) --
  https://developer.nvidia.com/blog/a-practical-guide-to-gpu-initiated-communication-for-molecular-dynamics-at-scale/
- Demystifying NVSHMEM: symmetric memory + device-initiated ops -- https://arxiv.org/html/2606.05951v1
- CPU- and GPU-initiated Communication Strategies for CG on Large GPU Clusters (SC) --
  https://dl.acm.org/doi/10.1145/3712285.3759774
- Communication-reduced Conjugate Gradient Variants for GPU Clusters -- https://arxiv.org/abs/2501.03743
- Magnum IO NVSHMEM + GPUDirect Async (IBGDA) --
  https://developer.nvidia.com/blog/improving-network-performance-of-hpc-systems-using-nvidia-magnum-io-nvshmem-and-gpudirect-async/
- NVSHMEM FAQ (transport tiers, `nvshmem_ptr`, PCIe atomics) -- https://docs.nvidia.com/nvshmem/api/faq.html
- DGX Spark / GB10 hardware overview -- https://docs.nvidia.com/dgx/dgx-spark/hardware.html
