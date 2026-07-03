#!/usr/bin/env bash
# ============================================================================
#  Build the OpenFOAM + PETSc-GPU benchmark back-end (the `of_petsc` column).
#
#  This is OpenFOAM offloading the pressure solve to the GPU via PETSc's own GAMG on cuSPARSE, through the
#  petsc4Foam (external-solver) module. Selecting in fvSolution:
#      p { solver petsc; petsc { options { ksp_type cg; mat_type aijcusparse; pc_type gamg; } } }
#  + libs ("libpetscFoam.so"); + env PETSC_OPTIONS="-use_gpu_aware_mpi 0".
#
#  Building PETSc+CUDA is environment-specific and slow (~20-40 min). This script captures the recipe that works,
#  with the GPU-arch and MPI traps handled. Everything is overridable by env.
#
#  Usage:  ./setup_of_petsc.sh
#  Env:  PETSC_SRC (clone target, default $HOME/petsc)  PETSC_ARCH (default arch-cuda)
#        CUDA_ARCH (default: autodetect)  MPICC (default /usr/bin/mpicc, MUST match OpenFOAM's MPI)
#
#  THE FOUR TRAPS (all handled below; see results/OF_GPU_SETUP_GB10.md for the full story):
#   1. MPI: PETSc, AMGX, and OpenFOAM must all link the SAME libmpi. Build PETSc with OpenFOAM's mpicc.
#   2. GPU arch: pin it, a newer GPU than nvcc's advertised list (e.g. GB10 sm_121) is auto-skipped otherwise.
#   3. Clean rebuild after changing arch (stale objects are silently reused).
#   4. At runtime, system MPI isn't CUDA-aware -> pass -use_gpu_aware_mpi 0 (single-GPU doesn't need it).
# ============================================================================
set -u
OFBASHRC="${OFBASHRC:-$(ls /usr/lib/openfoam/openfoam*/etc/bashrc /opt/openfoam*/etc/bashrc 2>/dev/null | head -1)}"
PETSC_SRC="${PETSC_SRC:-$HOME/petsc}"; PETSC_ARCH="${PETSC_ARCH:-arch-cuda}"
CUDA_ARCH="${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}"
MPICC="${MPICC:-/usr/bin/mpicc}"; MPICXX="${MPICXX:-/usr/bin/mpicxx}"; MPIFC="${MPIFC:-/usr/bin/mpif90}"

echo "== 1. PETSc with CUDA, pinned to sm_${CUDA_ARCH}, using OpenFOAM's MPI ($MPICC) =="
[ -d "$PETSC_SRC" ] || git clone -b release https://gitlab.com/petsc/petsc.git "$PETSC_SRC"
cd "$PETSC_SRC" || exit 2
export PATH=/usr/local/cuda/bin:/usr/bin:/bin CUDAARCHS="$CUDA_ARCH"      # trap 2: pin arch
unset PETSC_DIR
python3 ./configure \
  --download-fblaslapack=1 --with-cuda=1 --with-cudac=nvcc --with-cuda-arch="$CUDA_ARCH" \
  --with-debugging=0 --with-precision=double \
  --with-cc="$MPICC" --with-cxx="$MPICXX" --with-fc="$MPIFC" \
  PETSC_ARCH="$PETSC_ARCH" 2>&1 | tail -4
find "$PETSC_ARCH/obj" -name '*.o' -delete 2>/dev/null                    # trap 3: clean rebuild
make PETSC_DIR="$PETSC_SRC" PETSC_ARCH="$PETSC_ARCH" all 2>&1 | tail -4
[ -f "$PETSC_ARCH/lib/libpetsc.so" ] || { echo "PETSc build FAILED"; exit 1; }
echo "  libpetsc archs: $(cuobjdump --list-elf $PETSC_ARCH/lib/libpetsc.so 2>/dev/null | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' ')"

echo "== 2. petsc4Foam (external-solver) against this PETSc =="
set +u; source "$OFBASHRC" >/dev/null 2>&1; set -u
export PETSC_DIR="$PETSC_SRC" PETSC_ARCH="$PETSC_ARCH"
export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
P4F="${P4F:-$HOME/petsc4Foam}"
[ -d "$P4F" ] || git clone https://develop.openfoam.com/modules/external-solver.git "$P4F"
cd "$P4F" && ./Allwmake -j 2>&1 | tail -6
[ -f "$FOAM_USER_LIBBIN/libpetscFoam.so" ] && echo "OK -> $FOAM_USER_LIBBIN/libpetscFoam.so" || { echo "petsc4Foam build FAILED"; exit 1; }
echo "Now: run_benchmark.sh will auto-detect the of_petsc back-end (PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH)."
