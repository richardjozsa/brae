# legacy/ — multi-GPU (distributed) code. OUT OF SCOPE.

Nothing in this directory is built, tested, or maintained. **Do not read it when working on brae, and do
not change it.** It is kept only so the multi-GPU work is recoverable if it is ever resumed.

"Multi-GPU" here means **across GPUs / across MPI ranks**: the `-parallel` entry point, NVSHMEM halo
exchange, distributed matrices, and the MPI host solver. It does **not** mean the intra-GPU parallelism of
ordinary CUDA kernels — that is what brae *is*, and all of it lives in `src/` and is fully in scope.

## What moved here

| path | what it was |
|---|---|
| `src/applications/solvers/simpleFoam/parallel_device_*.cuh` | the distributed device solver (momentum, pressure, turbulence, AMI, interface) |
| `src/applications/solvers/simpleFoam/simpleFoam.cu` | `brae_simpleFoam`, the MPI-parallel **host** solver (`mpirun -np N`) |
| `src/TurbulenceModels/RAS/parallel_kepsilon.cuh` | host distributed kEpsilon |
| `src/cuda/device_pcg_distributed.cu` | distributed PCG |
| `tests/` (34 files) | every `parallel_*`, `gpu_parallel_*`, `distributed_ami`, `nvshmem_smoke` test |
| `tests/ami_oracle.sh` | compares a single-GPU AMI run against a **distributed np=1** run, so it needs `-parallel`. Missed by the first sweep because its name matches none of the patterns above -- it stayed registered and failed once `-parallel` began refusing. The single-GPU AMI tests (`ami_weights`, `ami_device`, `ami_geometry`) are unaffected and remain in `tests/`. |

## What deliberately did NOT move, and why

Two pieces are referenced by **core single-GPU** code, so moving them would break the single-GPU build:

- **`src/parallel/pstream/`** (`device_halo`, `device_reduce`, `cf_pstream`, `sym_buffer`) — `device_pcg.cu`
  and `device_amg.cu` include `device_reduce.cuh`/`device_halo.cuh` for the on-stream reduction.
- **`src/cuda/distributed_ami.cuh`** — `device_spmv.cu` includes it for the cyclicAMI matvec coupling
  (`distributedAmiAmul`, called only when a non-null AMI is supplied).
- **`parallel_simple.cuh`, `local_assembly.cuh`, `parallel_{amul,matrix_ops,pbicgstab,pcg}.cuh`** — these
  carry the mesh-DECOMPOSITION helpers (`Partition`, `LocalMesh`, proc deltas/weights), and single-GPU
  gates depend on them transitively: `test_energy_frozen` (rhoSimpleFoam Gate 1) and `test_rho_flux` both
  construct a `Partition`, and `test_{local_assembly,local_convection,proc_delta,proc_interp}` exercise the
  decomposition directly. Quarantining them would have deleted real single-GPU coverage, so the split is
  drawn at the distributed SOLVERS rather than at everything named "parallel".

These stay in `src/` and remain in scope. They are shared infrastructure, not the distributed solver.

## The `-parallel` flag

`brae ... -parallel` now stops at start-up with a message pointing here, rather than silently running a
single-GPU solve under an MPI launcher — which would produce N identical redundant runs, each writing over
the others' output.
