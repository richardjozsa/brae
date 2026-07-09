#!/usr/bin/env bash
# ============================================================================
#  H100 GPU-vs-GPU CFD benchmark.
#
#  Times a laminar scaled-pitzDaily SIMPLE solve on each available runner and
#  prints the wall time per iteration and how much faster brae is. This is the
#  table in H100_GPU_COMPARISON_REPORT.md. Nothing is kept on disk: each mesh is
#  built in a scratch dir, timed, and deleted.
#
#  Runners (auto-detected):
#    brae          device-resident GPU engine       (needs `brae` on PATH)
#    OpenFOAM-CPU  native GAMG on N cores            (always)
#    OF+PETSc-GPU  pressure offload, PETSc/cuSPARSE  (if libpetscFoam.so is built)
#    OF+AMGX       pressure offload, AMGX            (if libamgxFoam.so is built)
#    SPUMA         device-resident OpenFOAM-GPU port (if SPUMA_BIN is set)
#
#  Prereqs: OpenFOAM v2412 sourced. Competitor back-ends are built with the repo's
#           bench/setup_of_petsc.sh, setup_of_amgx.sh, setup_spuma.sh (all sm_90).
#
#  Usage:   ./run_benchmark.sh
#  Env:     SIZES="6 13 20"     blockMesh scale factor M (cells ~= 12225 * M^2)
#           ITERS=100  CORES=24  BRAE=brae  WORK=/tmp/h100_bench
#           SPUMA_BIN=/path/to/spuma/platforms/linux64NvidiaDPInt32Opt/bin/simpleFoam
#           SPUMA_POOL=24   NVARCH=90
# ============================================================================
set -u
BRAE="${BRAE:-brae}"; ITERS="${ITERS:-100}"; CORES="${CORES:-24}"
SIZES="${SIZES:-6 13 20}"; WORK="${WORK:-/tmp/h100_bench}"
SPUMA_BIN="${SPUMA_BIN:-}"; SPUMA_POOL="${SPUMA_POOL:-24}"; NVARCH="${NVARCH:-90}"
OFBASHRC="${OFBASHRC:-$(ls /usr/lib/openfoam/openfoam*/etc/bashrc /opt/openfoam*/etc/bashrc 2>/dev/null | head -1)}"
set +u; source "$OFBASHRC" >/dev/null 2>&1; set -u

command -v simpleFoam >/dev/null || { echo "ERROR: OpenFOAM not sourced (set OFBASHRC=...)"; exit 1; }
command -v "$BRAE"    >/dev/null || echo "WARNING: '$BRAE' not found; the brae column will be blank"

HAVE_PETSC=0; [ -f "$FOAM_USER_LIBBIN/libpetscFoam.so" ] && HAVE_PETSC=1
HAVE_AMGX=0;  [ -f "$FOAM_USER_LIBBIN/libamgxFoam.so"  ] && HAVE_AMGX=1
HAVE_SPUMA=0; [ -n "$SPUMA_BIN" ] && [ -x "$SPUMA_BIN" ] && HAVE_SPUMA=1
# default install locations match the repo's setup_of_petsc.sh / setup_of_amgx.sh
PETSC_DIR="${PETSC_DIR:-$HOME/petsc}"; PETSC_ARCH="${PETSC_ARCH:-arch-cuda}"; AMGX_DIR="${AMGX_DIR:-$HOME/opt/amgx}"
export PETSC_DIR PETSC_ARCH
export PETSC_OPTIONS="${PETSC_OPTIONS:--use_gpu_aware_mpi 0}"
export LD_LIBRARY_PATH="$PETSC_DIR/$PETSC_ARCH/lib:$AMGX_DIR/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

mkgrid(){ local M="$1" d="$2"; rm -rf "$d"; mkdir -p "$d"
  cp -r "$FOAM_TUTORIALS/incompressible/simpleFoam/pitzDaily/"{0,constant,system} "$d"/
  foamDictionary -entry simulationType         -set laminar   "$d/constant/turbulenceProperties" >/dev/null 2>&1
  foamDictionary -entry functions              -remove        "$d/system/controlDict"  >/dev/null 2>&1
  foamDictionary -entry SIMPLE/residualControl -remove        "$d/system/fvSolution"   >/dev/null 2>&1
  foamDictionary -entry endTime       -set "$ITERS" "$d/system/controlDict" >/dev/null 2>&1
  foamDictionary -entry writeInterval -set "$ITERS" "$d/system/controlDict" >/dev/null 2>&1
  python3 - "$d/system/blockMeshDict" "$M" <<'PY'
import re,sys; f,M=sys.argv[1],int(sys.argv[2]); o=[]
for ln in open(f):
    m=re.fullmatch(r'\((\d+)\s+(\d+)\s+(\d+)\)',ln.strip())
    o.append(re.sub(r'\(\d+\s+\d+\s+\d+\)',f'({int(m[1])*M} {int(m[2])*M} {int(m[3])})',ln) if m else ln)
open(f,'w').writelines(o)
PY
  ( cd "$d" && blockMesh >/dev/null 2>&1 ); }
wall(){ local a b; a=$(date +%s.%N); eval "$1" >/dev/null 2>&1; b=$(date +%s.%N); echo "$b - $a" | bc; }
perit(){ [ "${1:-}" = "-" ] && { echo "-"; return; }; printf "%.1f" "$(echo "scale=4; $1/$ITERS*1000" | bc)"; }
rat(){   [ "${2:-}" = "-" ] || [ -z "${2:-}" ] && { echo "-"; return; }; printf "%.1f" "$(echo "scale=3; $2/$1" | bc)"; }

mkdir -p "$WORK"
echo "brae=$(command -v "$BRAE" || echo none) | OF cores=$CORES | iters=$ITERS | PETSc=$HAVE_PETSC AMGX=$HAVE_AMGX SPUMA=$HAVE_SPUMA"
printf "\n%12s %9s %9s %9s %9s %9s\n" "cells" "brae" "OF-CPU" "SPUMA" "OF+AMGX" "OF+PETSc"
printf   "%12s %9s %9s %9s %9s %9s\n" "-----" "----" "------" "-----" "-------" "--------"
RATLINES=""
for M in $SIZES; do
  SRC="$WORK/grid_$M"; mkgrid "$M" "$SRC"
  NC=$(grep -aoE 'nCells:[0-9]+' "$SRC/constant/polyMesh/owner" | grep -oE '[0-9]+' | head -1)
  tB="-" tO="-" tS="-" tA="-" tP="-"
  # brae
  if command -v "$BRAE" >/dev/null; then
    BW="$WORK/brae_$M"; cp -r "$SRC" "$BW"; "$BRAE" -case "$BW" -partition >/dev/null 2>&1
    tB=$(wall "'$BRAE' -case '$BW'"); fi
  # OpenFOAM-CPU
  OW="$WORK/of_$M"; cp -r "$SRC" "$OW"
  printf 'FoamFile{version 2.0;format ascii;class dictionary;object decomposeParDict;}\nnumberOfSubdomains %d;method scotch;\n' "$CORES" > "$OW/system/decomposeParDict"
  ( cd "$OW" && decomposePar -force >/dev/null 2>&1 )
  tO=$(wall "( cd '$OW' && mpirun --allow-run-as-root -np $CORES simpleFoam -parallel )")
  # OF+PETSc-GPU
  if [ $HAVE_PETSC = 1 ]; then PW="$WORK/petsc_$M"; cp -r "$SRC" "$PW"
    foamDictionary -entry solvers/p -set '{solver petsc; petsc{options{ksp_type cg; mat_type aijcusparse; pc_type gamg;}} tolerance 1e-06; relTol 0.1;}' "$PW/system/fvSolution" >/dev/null 2>&1
    foamDictionary -entry libs -set '("libpetscFoam.so")' "$PW/system/controlDict" >/dev/null 2>&1
    tP=$(wall "( cd '$PW' && simpleFoam )"); fi
  # OF+AMGX
  if [ $HAVE_AMGX = 1 ]; then AW="$WORK/amgx_$M"; cp -r "$SRC" "$AW"
    foamDictionary -entry solvers/p/solver -set amgx "$AW/system/fvSolution" >/dev/null 2>&1
    foamDictionary -entry libs -set '("libamgxFoam.so")' "$AW/system/controlDict" >/dev/null 2>&1
    tA=$(wall "( cd '$AW' && simpleFoam )"); fi
  # SPUMA (own binary + GPU smoother + fixed memory pool; runs in its own env)
  if [ $HAVE_SPUMA = 1 ]; then SW="$WORK/spuma_$M"; cp -r "$SRC" "$SW"
    foamDictionary -entry solvers/p/smoother -set twoStageGaussSeidel "$SW/system/fvSolution" >/dev/null 2>&1
    foamDictionary -entry 'solvers/"(U|k|epsilon|omega|f|v2)"/smoother' -set twoStageSymGaussSeidel "$SW/system/fvSolution" >/dev/null 2>&1
    SP_SRC="$(cd "$(dirname "$SPUMA_BIN")/../../.." && pwd)"
    tS=$(wall "( export have_cuda=true NVARCH=$NVARCH FOAM_SIGFPE=false; source '$SP_SRC/etc/bashrc' >/dev/null 2>&1; cd '$SW' && '$SPUMA_BIN' -pool fixedSizeMemoryPool -poolSize $SPUMA_POOL )"); fi

  printf "%12s %9s %9s %9s %9s %9s\n" "$NC" "$(perit "$tB")" "$(perit "$tO")" "$(perit "$tS")" "$(perit "$tA")" "$(perit "$tP")"
  RATLINES="$RATLINES$(printf "\n%12s %9s %9s %9s %9s %9s" "$NC" "1.0" "$(rat "$tB" "$tO")" "$(rat "$tB" "$tS")" "$(rat "$tB" "$tA")" "$(rat "$tB" "$tP")")"
  rm -rf "$WORK/grid_$M" "$WORK"/{brae,of,petsc,amgx,spuma}_$M 2>/dev/null
done
echo
echo "ms per SIMPLE iteration (lower is better)."
echo
echo "how many times faster brae is (x):"
printf "%12s %9s %9s %9s %9s %9s\n" "cells" "brae" "OF-CPU" "SPUMA" "OF+AMGX" "OF+PETSc"
printf "%b\n" "$RATLINES"
