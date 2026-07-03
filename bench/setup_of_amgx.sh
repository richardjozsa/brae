#!/usr/bin/env bash
# ============================================================================
#  Build the OpenFOAM + AMGX benchmark back-end (the `of_amgx` column).
#
#  `amgxFoam/` is a self-contained OpenFOAM lduMatrix::solver that converts OF's LDU matrix to CSR and calls the
#  AMGX C API directly, NO PETSc (it avoids PETSc's PCAMGX Thrust bug on new GPU archs). Selecting `solver amgx;`
#  in fvSolution + `libs ("libamgxFoam.so");` in controlDict makes OpenFOAM solve pressure on the GPU via AMGX.
#
#  Prereqs:
#    - OpenFOAM sourced (or set OFBASHRC)
#    - CUDA toolkit (nvcc)
#    - AMGX built as a SHARED lib for YOUR GPU arch (libamgxsh.so). If you don't have it, this script can build it
#      from AMGX source (set AMGX_SRC). NOTE: AMGX must be built for your exact arch, e.g. GB10 = sm_121; AMGX
#      2.5.0's CLASSICAL AMG is broken on sm_121 (Thrust), so the solver defaults to AGGREGATION (works everywhere).
#
#  Usage:   AMGX_DIR=/path/to/amgx  ./setup_of_amgx.sh
#     or:   AMGX_SRC=/path/to/AMGX/source  CUDA_ARCH=121  ./setup_of_amgx.sh   # builds AMGX first
# ============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
OFBASHRC="${OFBASHRC:-$(ls /usr/lib/openfoam/openfoam*/etc/bashrc /opt/openfoam*/etc/bashrc 2>/dev/null | head -1)}"
set +u; source "$OFBASHRC" >/dev/null 2>&1; set -u
AMGX_DIR="${AMGX_DIR:-$HOME/opt/amgx}"
CUDA_ARCH="${CUDA_ARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}"

# --- (optional) build AMGX for your arch ---
if [ -n "${AMGX_SRC:-}" ]; then
  echo "== building AMGX for sm_${CUDA_ARCH} from $AMGX_SRC =="
  mkdir -p "$AMGX_SRC/build"; ( cd "$AMGX_SRC/build"
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$AMGX_DIR" -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" .. >/dev/null 2>&1
    make -j"$(nproc)" amgxsh amgx 2>&1 | tail -3 )   # BOTH shared (.so) and static (.a)
  mkdir -p "$AMGX_DIR/lib" "$AMGX_DIR/include"
  cp -v "$AMGX_SRC/build"/libamgxsh.so "$AMGX_SRC/build"/libamgx.a "$AMGX_DIR/lib/" 2>/dev/null
  cp -v "$AMGX_SRC"/include/*.h "$AMGX_DIR/include/" 2>/dev/null
fi
[ -f "$AMGX_DIR/lib/libamgxsh.so" ] || { echo "ERROR: no $AMGX_DIR/lib/libamgxsh.so (set AMGX_DIR to a prebuilt AMGX, or AMGX_SRC to build it)"; exit 1; }
echo "AMGX archs: $(cuobjdump --list-elf "$AMGX_DIR/lib/libamgxsh.so" 2>/dev/null | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' ') (must include sm_${CUDA_ARCH})"

# --- build the amgxFoam OF solver ---
export AMGX_INC="$AMGX_DIR/include" AMGX_LIB="$AMGX_DIR/lib"
export LD_LIBRARY_PATH="$AMGX_LIB:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
cd "$HERE/amgxFoam" || exit 2
sed -i -E "s/-arch=sm_[0-9]+/-arch=sm_${CUDA_ARCH}/g" Make/options 2>/dev/null   # if the rule pins an arch
wclean >/dev/null 2>&1
echo "== wmake amgxFoam (AMGX_INC=$AMGX_INC) =="
wmake libso 2>&1 | tail -5
[ -f "$FOAM_USER_LIBBIN/libamgxFoam.so" ] && echo "OK -> $FOAM_USER_LIBBIN/libamgxFoam.so" || { echo "BUILD FAILED"; exit 1; }
echo "Now: run_benchmark.sh will auto-detect the of_amgx back-end."
