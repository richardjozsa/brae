# OF-on-GPU on GB10 via petsc4Foam, setup, the AMGX blocker, and the working path

Goal: run OpenFOAM's pressure solve on the GB10 GPU (for a GPU-vs-GPU comparison against brae).
Reference recipe: NextFOAM blog (blog.nextfoam.co.kr/2024/01/10/gpu-accelerated-openfoam-with-petsc4foam/).

## TL;DR
- **OF-on-GPU WORKS on GB10 via PETSc-GAMG + cuSPARSE** (`pc_type gamg`, `mat_type aijcusparse`, `-use_gpu_aware_mpi 0`).
- **AMGX 2.5.0 CORE WORKS on GB10 sm_121**, PROVEN by a standalone C test (`/tmp/amgx_min.cu`: `AMGX_solver_setup` +
  `AMGX_solver_solve` on a 1D Laplacian → correct answer). AMGX is NOT the blocker.
- **PETSc's PCAMGX interface (`pc_type amgx`) IS broken on CUDA-13/sm_121**: its own Thrust matrix-conversion
  `parallel_for` throws "cudaErrorInvalidDeviceFunction" even with everything rebuilt pure-sm_121. A PETSc-3.24 ×
  CUDA-13 interface bug, not AMGX. → Use the **FOAM2CSR path** (calls the AMGX C API directly, bypassing PETSc's Thrust).
- **FOAM2CSR built for sm_121** (`libfoam2csr.so`). Remaining piece = the OF `solver amgx` RTS glue, which lived in the
  external-solver **`amgxwrapper` branch, now REMOVED upstream**. Write it by hand (FOAM2CSR `AmgXCSRMatrix::setValuesLDU`
  + raw AMGX) or find it on a mirror.

## The stack (all must agree on MPI + GPU arch)
GB10 = compute capability **12.1 (sm_121)**, Grace-Blackwell, unified LPDDR5x. CUDA 13.0.88, nvcc compiles sm_121
(even though its advertised `--list` stops at 120). System OpenMPI **4.1.6** at /usr (what OF v2412 + AMGX link).

| layer | location | build requirement | status |
|---|---|---|---|
| OpenFOAM | /usr/lib/openfoam/openfoam2412 | system OpenMPI 4.1.6 | prebuilt ✓ |
| PETSc | ~/space/amgx/petsc (PETSC_ARCH=arch-sysmpi) | `--with-cc=/usr/bin/mpicc` (sys-MPI) + `--with-cuda-arch=121` + `CUDAARCHS=121` | rebuilt ✓ |
| AMGX | ~/opt/amgx (src ~/space/amgx/AMGX) | `-DCMAKE_CUDA_ARCHITECTURES=121`, build BOTH `amgxsh` (.so) AND `amgx` (.a) | rebuilt ✓ but runtime-incompatible |
| petsc4Foam | ~/space/petsc4Foam | `./Allwmake` against arch-sysmpi PETSc | built ✓ |

## The four traps that cost time (in order hit)
1. **MPI mismatch.** PETSc was originally built vs CUDA-aware OpenMPI 5.0.7 (~/opt/ompi-cuda); OF + AMGX use system
   4.1.6. Two `libmpi` in one process → rebuild PETSc vs system MPI (`--with-cc=/usr/bin/mpicc`). Verify with
   `ldd libpetscFoam.so | grep mpi` → must be `/lib/aarch64-linux-gnu/libmpi.so.40`.
2. **PETSc CUDA arch.** nvcc 13's advertised arch list stops at 120, so PETSc auto-picked a `90;100;120` gencode fatbin
   for most objects → most kernels can't run on sm_121. Fix: pin `CUDAARCHS=121` + `--with-cuda-arch=121`, then a
   CLEAN rebuild (`find arch-sysmpi/obj -name '*.o' -delete`, stale objects are silently reused otherwise).
3. **AMGX arch (shared).** `~/opt/amgx/lib/libamgxsh.so` was `90;100;120`. Rebuild: `cmake -DCMAKE_CUDA_ARCHITECTURES=121 .
   && make -j amgxsh`. ~3 min (single arch).
4. **AMGX arch (STATIC, the hidden one).** PETSc links `-lamgx` = the STATIC `libamgx.a`, which it embeds into
   libpetsc.so. Rebuilding only `amgxsh` leaves `libamgx.a` at 90;100;120 → libpetsc.so stays mixed. Fix: also
   `make -j amgx` (static target, reuses the sm_121 objects → seconds), copy to ~/opt/amgx/lib, then relink PETSc
   (`rm arch-sysmpi/lib/libpetsc.so* && make all`). Verify: `cuobjdump --list-elf libpetsc.so | grep -oE 'sm_[0-9]+' | sort -u`
   must be `sm_121` ONLY.

After all four: libpetsc.so is pure sm_121, AMGX pure sm_121, yet `pc_type amgx` STILL fails (Thrust invalid device
function). => AMGX 2.5.0 is the wall. `pc_type gamg` (PETSc-native, cuSPARSE) works.

## Working fvSolution (PETSc-GAMG-GPU)
```
solvers
{
    p
    {
        solver petsc;
        petsc { options { ksp_type cg; mat_type aijcusparse; pc_type gamg; } }
        tolerance 1e-06; relTol 0.1;
    }
    "(U|k|epsilon|omega|f|v2)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-05; relTol 0.1; }
}
```
controlDict: `libs ("libpetscFoam.so");`
Runtime env (single-GPU, system MPI):
```
export PETSC_DIR=/home/ghost/space/amgx/petsc PETSC_ARCH=arch-sysmpi
export LD_LIBRARY_PATH=$PETSC_DIR/$PETSC_ARCH/lib:/home/ghost/opt/amgx/lib:$LD_LIBRARY_PATH
export PETSC_OPTIONS="-use_gpu_aware_mpi 0"   # system MPI isn't CUDA-aware; single-GPU doesn't need it
```
Verified: `PETSc-cg: Solving for p … No Iterations 4`, GPU active (~20-37% util at 990k).

## Reproduce
Build chain scripts (persistent): ~/space/{rebuild_petsc_sm121,make_petsc_sm121,fix_static_amgx}.sh.
Sweep: scratchpad/petscgpu_sweep.sh → bench_results/petscgpu_100iter.txt.
