#!/usr/bin/env bash
# ============================================================================
#  Replay the brae benchmark on YOUR GPU.
#
#  Compares TOTAL WALL TIME for a fixed number of SIMPLE iterations of:
#     brae        , 1 GPU, fully device-resident
#     OpenFOAM    , N CPU cores (native GAMG, MPI)
#     OF + AMGX   , 1 GPU, OpenFOAM offloading the pressure solve (if built; see setup_of_amgx.sh)
#     OF + PETSc  , 1 GPU, OpenFOAM via PETSc-GAMG/cuSPARSE      (if built; see setup_of_petsc.sh)
#  on scaled-pitzDaily laminar meshes. Partition/decompose is done ONCE and EXCLUDED from the timed number
#  (both sides pay a one-time prep; we time the run, not the prep).
#
#  Usage:   ./run_benchmark.sh
#  Env:
#     BRAE       brae binary                (default: ../build/brae)
#     OFBASHRC   OpenFOAM etc/bashrc        (default: autodetect)
#     CORES      OpenFOAM CPU cores         (default: 20)
#     SIZES      blockMesh scale factors    (default: "3 9 20"  ~= 110k / 990k / 4.9M cells)
#     ITERS      SIMPLE iterations timed    (default: 100)
#     WORK       scratch dir                (default: /tmp/brae_bench)
#     AMGX_DIR   AMGX install (for OF+AMGX) (default: $HOME/opt/amgx)
#     PETSC_DIR / PETSC_ARCH  (for OF+PETSc)
# ============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
BRAE="${BRAE:-$HERE/../build/brae}"
CORES="${CORES:-20}"; ITERS="${ITERS:-100}"; SIZES="${SIZES:-3 9 20}"; WORK="${WORK:-/tmp/brae_bench}"
OFBASHRC="${OFBASHRC:-$(ls /usr/lib/openfoam/openfoam*/etc/bashrc /opt/openfoam*/etc/bashrc 2>/dev/null | head -1)}"
set +u; source "$OFBASHRC" >/dev/null 2>&1; set -u

[ -x "$BRAE" ] || { echo "ERROR: brae binary not found at '$BRAE' (set BRAE=...)"; exit 1; }
command -v simpleFoam >/dev/null || { echo "ERROR: OpenFOAM not sourced (set OFBASHRC=...)"; exit 1; }

# Optional GPU backends, auto-detected by the presence of their libraries
AMGX_DIR="${AMGX_DIR:-$HOME/opt/amgx}"; PETSC_DIR="${PETSC_DIR:-}"; PETSC_ARCH="${PETSC_ARCH:-arch-sysmpi}"
HAVE_AMGX=0;  [ -f "$FOAM_USER_LIBBIN/libamgxFoam.so"  ] && HAVE_AMGX=1
HAVE_PETSC=0; [ -f "$FOAM_USER_LIBBIN/libpetscFoam.so" ] && HAVE_PETSC=1
export LD_LIBRARY_PATH="$AMGX_DIR/lib:${PETSC_DIR:+$PETSC_DIR/$PETSC_ARCH/lib:}/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PETSC_OPTIONS="${PETSC_OPTIONS:--use_gpu_aware_mpi 0}"
echo "brae=$BRAE | OF cores=$CORES | iters=$ITERS | OF+AMGX=$([ $HAVE_AMGX = 1 ]&&echo yes||echo no) | OF+PETSc=$([ $HAVE_PETSC = 1 ]&&echo yes||echo no)"

mkgrid(){ local M="$1" d="$2"; rm -rf "$d"; mkdir -p "$d"
  cp -r "$FOAM_TUTORIALS/incompressible/simpleFoam/pitzDaily/"{0,constant,system} "$d"/
  foamDictionary -entry simulationType -set laminar "$d/constant/turbulenceProperties" >/dev/null 2>&1
  python3 - "$d/system/blockMeshDict" "$M" <<'PY'
import re,sys; f,M=sys.argv[1],int(sys.argv[2]); o=[]
for ln in open(f):
    m=re.fullmatch(r'\((\d+)\s+(\d+)\s+(\d+)\)',ln.strip())
    o.append(re.sub(r'\(\d+\s+\d+\s+\d+\)',f'({int(m[1])*M} {int(m[2])*M} {int(m[3])})',ln) if m else ln)
open(f,'w').writelines(o)
PY
  ( cd "$d"; blockMesh >/dev/null 2>&1 ); }
wall(){ local a b; a=$(date +%s.%N); eval "$1" >/dev/null 2>&1; b=$(date +%s.%N); echo "$b - $a"|bc; }
fmt(){ [ "${1:-}" = "-" ] || [ -z "${1:-}" ] && echo "-" || printf "%.1f" "$1"; }

mkdir -p "$WORK"; RES="$WORK/results.csv"
echo "nCells,brae_s,of${CORES}core_s,of_amgx_s,of_petsc_s" > "$RES"
printf "\n%10s %9s %11s %11s %11s\n" "nCells" "brae" "OF-${CORES}c" "OF+AMGX" "OF+PETSc"
printf "%10s %9s %11s %11s %11s\n" "------" "----" "------" "-------" "--------"
for M in $SIZES; do
  SRC="$WORK/mesh_$M"; mkgrid "$M" "$SRC"
  NC=$(grep -aoE 'nCells:?[[:space:]]*[0-9]+' "$SRC/constant/polyMesh/owner" 2>/dev/null | grep -oE '[0-9]+' | head -1)
  [ -n "$NC" ] || NC=$(grep -iE 'nCells' "$SRC/log.bm" 2>/dev/null | grep -oE '[0-9]+' | tail -1)
  # --- brae (fast path) ---
  BW="$WORK/brae_$M"; rm -rf "$BW"; cp -r "$SRC" "$BW"; foamDictionary -entry endTime -set $ITERS "$BW/system/controlDict" >/dev/null 2>&1
  BRAE_PCG_DEVICE=1 BRAE_AMG_FP32=1 "$BRAE" -case "$BW" -partition >/dev/null 2>&1     # warm cache (EXCLUDED)
  tB=$(wall "BRAE_PCG_DEVICE=1 BRAE_AMG_FP32=1 '$BRAE' -case '$BW'")
  # --- OpenFOAM, N cores ---
  OW="$WORK/of_$M"; rm -rf "$OW"; cp -r "$SRC" "$OW"; foamDictionary -entry endTime -set $ITERS "$OW/system/controlDict" >/dev/null 2>&1
  printf 'FoamFile{version 2.0;format ascii;class dictionary;object decomposeParDict;}\nnumberOfSubdomains %d;method scotch;\n' "$CORES" > "$OW/system/decomposeParDict"
  ( cd "$OW"; decomposePar -force >/dev/null 2>&1 )                                     # decompose (EXCLUDED)
  tO=$(wall "( cd '$OW'; mpirun -np $CORES simpleFoam -parallel )")
  # --- OF + AMGX (optional) ---
  tA="-"; if [ $HAVE_AMGX = 1 ]; then
    AW="$WORK/amgx_$M"; rm -rf "$AW"; cp -r "$SRC" "$AW"
    foamDictionary -entry solvers/p/solver -set amgx "$AW/system/fvSolution" >/dev/null 2>&1
    foamDictionary -entry libs -set '("libamgxFoam.so")' "$AW/system/controlDict" >/dev/null 2>&1
    foamDictionary -entry endTime -set $ITERS "$AW/system/controlDict" >/dev/null 2>&1
    tA=$(wall "( cd '$AW'; simpleFoam )"); fi
  # --- OF + PETSc-GPU (optional) ---
  tP="-"; if [ $HAVE_PETSC = 1 ]; then
    PW="$WORK/petsc_$M"; rm -rf "$PW"; cp -r "$SRC" "$PW"
    foamDictionary -entry solvers/p -set '{solver petsc; petsc{options{ksp_type cg; mat_type aijcusparse; pc_type gamg;}} tolerance 1e-6; relTol 0.1;}' "$PW/system/fvSolution" >/dev/null 2>&1
    foamDictionary -entry libs -set '("libpetscFoam.so")' "$PW/system/controlDict" >/dev/null 2>&1
    foamDictionary -entry endTime -set $ITERS "$PW/system/controlDict" >/dev/null 2>&1
    tP=$(wall "( cd '$PW'; simpleFoam )"); fi

  printf "%10s %9s %11s %11s %11s\n" "$NC" "$(fmt "$tB")" "$(fmt "$tO")" "$(fmt "$tA")" "$(fmt "$tP")"
  echo "$NC,$(fmt "$tB"),$(fmt "$tO"),$(fmt "$tA"),$(fmt "$tP")" >> "$RES"
  rm -rf "$OW"/processor* "$BW"/[1-9]* 2>/dev/null
done
echo; echo "CSV -> $RES"
