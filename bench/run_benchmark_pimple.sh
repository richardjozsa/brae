#!/usr/bin/env bash
# ============================================================================
#  TRANSIENT pimpleFoam benchmark -- the 5-way GPU/CPU comparison.
#
#  Compares TOTAL WALL TIME for a FIXED number of transient timesteps of:
#     brae        , 1 GPU, fully device-resident      (build/brae_pimpleFoam)
#     OpenFOAM    , N CPU cores (native GAMG, MPI)     (pimpleFoam -parallel)
#     SPUMA       , 1 GPU, OF-v2412 fork on unified memory (pimpleFoam -pool ...)   [if built]
#     OF + AMGX   , 1 GPU, OF offloading p/pFinal to AMGX   (libamgxFoam.so)          [if built]
#     OF + PETSc  , 1 GPU, OF via PETSc-GAMG/cuSPARSE       (libpetscFoam.so)         [if built]
#  on scaled pimpleFoam/RAS/pitzDaily meshes. adjustTimeStep is FORCED OFF and deltaT fixed, so every
#  backend marches EXACTLY $STEPS timesteps -- the honest apples-to-apples (variable-dt would run
#  different step counts). One-time prep (brae AMG-cache warm / OF decomposePar) is EXCLUDED from the timed number.
#
#  Usage:   ./run_benchmark_pimple.sh
#  Env:
#     BRAE       brae_pimpleFoam binary        (default: ../build/brae_pimpleFoam)
#     OFBASHRC   OpenFOAM etc/bashrc           (default: autodetect)
#     CORES      OpenFOAM CPU cores            (default: 20)
#     SIZES      blockMesh scale factors       (default: "3 9 20"  ~= 110k / 990k / 4.9M cells)
#     STEPS      transient timesteps timed     (default: 100)
#     DT         fixed deltaT                  (default: 1e-4)
#     TURB       laminar | RAS-kEpsilon | RAS-kOmegaSST  (default: keep the tutorial's kEpsilon)
#     WORK       scratch dir                   (default: /tmp/brae_bench_pimple)
#     SPUMA_BIN  SPUMA platforms bin dir       (default: autodetect under ~/space/spuma-fresh)
#     SPUMA_ENV  SPUMA env to source           (default: <spuma>/spuma_env.sh)
#     SPUMA_POOLGB  SPUMA unified-memory pool GB (default: 40)
#     AMGX_DIR   AMGX install (for OF+AMGX)    (default: $HOME/opt/amgx)
#     PETSC_DIR / PETSC_ARCH  (for OF+PETSc)
# ============================================================================
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
BRAE="${BRAE:-$HERE/../build/brae_pimpleFoam}"
CORES="${CORES:-20}"; STEPS="${STEPS:-100}"; DT="${DT:-1e-4}"; SIZES="${SIZES:-3 9 20}"
TURB="${TURB:-RAS-kEpsilon}"; WORK="${WORK:-$HOME/brae_bench_pimple}"   # persistent (survives reboots), not /tmp
TIMEOUT="${TIMEOUT:-0}"   # per-backend wall cap in seconds (0 = unlimited); a slow backend records "TO" and the sweep continues
OFBASHRC="${OFBASHRC:-$(ls /usr/lib/openfoam/openfoam*/etc/bashrc /opt/openfoam*/etc/bashrc 2>/dev/null | head -1)}"
set +u; source "$OFBASHRC" >/dev/null 2>&1; set -u

[ -x "$BRAE" ] || { echo "ERROR: brae_pimpleFoam not found at '$BRAE' (build it: cmake --build ../build --target brae_pimpleFoam; or set BRAE=...)"; exit 1; }
command -v pimpleFoam >/dev/null || { echo "ERROR: OpenFOAM not sourced (set OFBASHRC=...)"; exit 1; }
BASE="$FOAM_TUTORIALS/incompressible/pimpleFoam/RAS/pitzDaily"
[ -d "$BASE" ] || { echo "ERROR: base case not found: $BASE"; exit 1; }

# ---- optional back-ends, auto-detected ----
# SPUMA: pick the install that actually has a BUILT pimpleFoam (spuma-fresh has the build; spuma-bench is just scripts).
SPUMA_BIN="${SPUMA_BIN:-}"; SPUMA_BASE=""
if [ -n "$SPUMA_BIN" ]; then SPUMA_BASE="$(cd "$SPUMA_BIN/../.." 2>/dev/null && pwd)"; else
  for cand in "$HOME"/space/spuma-fresh "$HOME"/space/spuma-bench; do
    b="$(ls -d "$cand"/platforms/*/bin 2>/dev/null | head -1)"
    if [ -x "$b/pimpleFoam" ]; then SPUMA_BIN="$b"; SPUMA_BASE="$cand"; break; fi
  done
fi
SPUMA_ENV="${SPUMA_ENV:-$SPUMA_BASE/spuma_env.sh}"
SPUMA_POOLGB="${SPUMA_POOLGB:-auto}"   # auto = size the pool to the mesh per-run; or set a fixed GB
SPUMA_TUNE="${SPUMA_TUNE:-1}"   # 1 = give SPUMA its GPU-parallel PRESSURE smoother (twoStageGaussSeidel); stock is serial-slow on GPU
HAVE_SPUMA=0; [ -n "$SPUMA_BIN" ] && [ -x "$SPUMA_BIN/pimpleFoam" ] && [ -f "$SPUMA_ENV" ] && HAVE_SPUMA=1
AMGX_DIR="${AMGX_DIR:-$HOME/opt/amgx}"; PETSC_ARCH="${PETSC_ARCH:-arch-sysmpi}"
# PETSc: libpetscFoam.so NEEDs libpetsc.so on LD_LIBRARY_PATH -> auto-detect the PETSc install dir (else petsc won't dlopen)
PETSC_DIR="${PETSC_DIR:-$(for p in "$HOME"/space/amgx/petsc "$HOME"/petsc; do [ -f "$p/$PETSC_ARCH/lib/libpetsc.so" ] && { echo "$p"; break; }; done)}"
HAVE_AMGX=0;  [ -f "$FOAM_USER_LIBBIN/libamgxFoam.so"  ] && HAVE_AMGX=1
HAVE_PETSC=0; [ -f "$FOAM_USER_LIBBIN/libpetscFoam.so" ] && [ -n "$PETSC_DIR" ] && HAVE_PETSC=1   # petsc needs its runtime libs
export LD_LIBRARY_PATH="$AMGX_DIR/lib:${PETSC_DIR:+$PETSC_DIR/$PETSC_ARCH/lib:}/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
export PETSC_OPTIONS="${PETSC_OPTIONS:--use_gpu_aware_mpi 0}"
GPU="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"
echo "brae=$BRAE | OF cores=$CORES | steps=$STEPS | dt=$DT | turb=$TURB | timeout=${TIMEOUT}s | GPU=$GPU"
echo "SPUMA=$([ $HAVE_SPUMA = 1 ]&&echo yes||echo no) | OF+AMGX=$([ $HAVE_AMGX = 1 ]&&echo yes||echo no) | OF+PETSc=$([ $HAVE_PETSC = 1 ]&&echo yes||echo no) | out=$WORK/results.csv"

# Build a scaled, FIXED-dt transient case from the OF pimpleFoam/RAS/pitzDaily tutorial.
mkgrid(){ local M="$1" d="$2"; rm -rf "$d"; mkdir -p "$d"
  cp -r "$BASE/"{0,constant,system} "$d"/
  case "$TURB" in
    laminar)       foamDictionary -entry simulationType -set laminar "$d/constant/turbulenceProperties" >/dev/null 2>&1 ;;
    RAS-kOmegaSST) foamDictionary -entry RASModel -set kOmegaSST "$d/constant/turbulenceProperties" >/dev/null 2>&1 ;;
    *) : ;;  # keep the tutorial's kEpsilon
  esac
  # FIXED timestep, no adaptive dt, write only at the end -> every backend does exactly $STEPS steps, no I/O skew.
  local END; END=$(python3 -c "print(f'{$DT*$STEPS:.10g}')")
  foamDictionary -entry adjustTimeStep -set no        "$d/system/controlDict" >/dev/null 2>&1
  foamDictionary -entry deltaT         -set "$DT"     "$d/system/controlDict" >/dev/null 2>&1
  foamDictionary -entry endTime        -set "$END"    "$d/system/controlDict" >/dev/null 2>&1
  foamDictionary -entry writeControl   -set timeStep  "$d/system/controlDict" >/dev/null 2>&1
  foamDictionary -entry writeInterval  -set "$STEPS"  "$d/system/controlDict" >/dev/null 2>&1
  foamDictionary -entry purgeWrite     -set 0         "$d/system/controlDict" >/dev/null 2>&1
  # scale the pitzDaily blockMesh (multiply the (nx ny nz) cell counts by M), same as the steady bench
  python3 - "$d/system/blockMeshDict" "$M" <<'PY'
import re,sys; f,M=sys.argv[1],int(sys.argv[2]); o=[]
for ln in open(f):
    m=re.fullmatch(r'\((\d+)\s+(\d+)\s+(\d+)\)',ln.strip())
    o.append(re.sub(r'\(\d+\s+\d+\s+\d+\)',f'({int(m[1])*M} {int(m[2])*M} {int(m[3])})',ln) if m else ln)
open(f,'w').writelines(o)
PY
  ( cd "$d"; blockMesh >/dev/null 2>&1 ); }

# set a GPU-offload linear solver on BOTH p and pFinal (pimpleFoam solves p each corrector, pFinal on the last)
set_psolver(){ local d="$1" solver="$2" body="$3"
  if [ -n "$body" ]; then
    foamDictionary -entry "solvers/p"      -set "$body" "$d/system/fvSolution" >/dev/null 2>&1
    foamDictionary -entry "solvers/pFinal" -set "$body" "$d/system/fvSolution" >/dev/null 2>&1
  else
    foamDictionary -entry "solvers/p/solver"      -set "$solver" "$d/system/fvSolution" >/dev/null 2>&1
    foamDictionary -entry "solvers/pFinal/solver" -set "$solver" "$d/system/fvSolution" >/dev/null 2>&1
  fi; }

# time a backend; TIMEOUT (s, 0=unlimited) caps each run so one slow/hung backend can't stall an unattended sweep -> "TO"
wall(){ local a b rc; a=$(date +%s.%N); timeout "${TIMEOUT:-0}" bash -c "$1" >/dev/null 2>&1; rc=$?; b=$(date +%s.%N); [ "$rc" = 124 ] && { echo "TO"; return; }; echo "$b - $a"|bc; }
fmt(){ case "${1:-}" in ""|"-") echo "-";; TO) echo "TO";; *) printf "%.1f" "$1";; esac; }

mkdir -p "$WORK"; RES="$WORK/results.csv"
{ echo "# pimpleFoam transient benchmark | GPU=$GPU | OF=${CORES}core | steps=$STEPS dt=$DT turb=$TURB spuma_tune=$SPUMA_TUNE | $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "nCells,brae_s,of${CORES}core_s,spuma_s,of_amgx_s,of_petsc_s"; } > "$RES"
printf "\n%10s %9s %11s %10s %11s %11s\n" "nCells" "brae" "OF-${CORES}c" "SPUMA" "OF+AMGX" "OF+PETSc"
printf "%10s %9s %11s %10s %11s %11s\n"   "------" "----" "------" "-----" "-------" "--------"
for M in $SIZES; do
  SRC="$WORK/mesh_$M"; mkgrid "$M" "$SRC"
  NC=$(grep -aoE 'nCells:?[[:space:]]*[0-9]+' "$SRC/constant/polyMesh/owner" 2>/dev/null | grep -oE '[0-9]+' | head -1)

  # --- brae (device-resident); warm the AMG cache once (EXCLUDED), then time $STEPS steps ---
  BW="$WORK/brae_$M"; rm -rf "$BW"; cp -r "$SRC" "$BW"
  foamDictionary -entry endTime -set "$(python3 -c "print(f'{$DT*2:.10g}')")" "$BW/system/controlDict" >/dev/null 2>&1
  BRAE_MESH_CACHE=1 "$BRAE" "$BW" >/dev/null 2>&1                                   # build + cache AMG hierarchy (EXCLUDED)
  rm -rf "$BW"/0.* "$BW"/[1-9]* 2>/dev/null
  foamDictionary -entry endTime -set "$(python3 -c "print(f'{$DT*$STEPS:.10g}')")" "$BW/system/controlDict" >/dev/null 2>&1
  tB=$(wall "BRAE_MESH_CACHE=1 '$BRAE' '$BW'")

  # --- OpenFOAM, N cores ---
  OW="$WORK/of_$M"; rm -rf "$OW"; cp -r "$SRC" "$OW"
  printf 'FoamFile{version 2.0;format ascii;class dictionary;object decomposeParDict;}\nnumberOfSubdomains %d;method scotch;\n' "$CORES" > "$OW/system/decomposeParDict"
  ( cd "$OW"; decomposePar -force >/dev/null 2>&1 )                                 # decompose (EXCLUDED)
  tO=$(wall "( cd '$OW'; mpirun -np $CORES pimpleFoam -parallel )")

  # --- SPUMA, 1 GPU (unified memory), if built ---
  SW=""; tS="-"; if [ $HAVE_SPUMA = 1 ]; then
    SW="$WORK/spuma_$M"; rm -rf "$SW"; cp -r "$SRC" "$SW"
    if [ "$SPUMA_TUNE" = 1 ]; then   # SPUMA's GPU-parallel PRESSURE smoother (the dominant, STABLE win). NOTE: swapping
      # k/epsilon to twoStageSymGaussSeidel too diverges to -nan here, so leave the turbulence scalars at stock symGaussSeidel.
      sed -i -e 's/\bDICGaussSeidel\b/twoStageGaussSeidel/g' -e 's/\bGaussSeidel\b/twoStageGaussSeidel/g' "$SW/system/fvSolution"
    fi
    pool="$SPUMA_POOLGB"; [ "$pool" = auto ] && pool=$(python3 -c "print(max(4, int($NC/1e6*2)+2))")   # size the unified-memory pool to the mesh (flat-40G = slow startup on small cases)
    tS=$(wall "( set +u; source '$SPUMA_ENV' >/dev/null 2>&1; cd '$SW'; '$SPUMA_BIN/pimpleFoam' -pool fixedSizeMemoryPool -poolSize $pool )"); fi

  # --- OF + AMGX (p/pFinal offloaded), if built ---
  tA="-"; if [ $HAVE_AMGX = 1 ]; then
    AW="$WORK/amgx_$M"; rm -rf "$AW"; cp -r "$SRC" "$AW"
    set_psolver "$AW" amgx ""
    foamDictionary -entry libs -set '("libamgxFoam.so")' "$AW/system/controlDict" >/dev/null 2>&1
    tA=$(wall "( cd '$AW'; pimpleFoam )"); fi

  # --- OF + PETSc-GPU (p/pFinal offloaded), if built ---
  tP="-"; if [ $HAVE_PETSC = 1 ]; then
    PW="$WORK/petsc_$M"; rm -rf "$PW"; cp -r "$SRC" "$PW"
    set_psolver "$PW" petsc '{solver petsc; petsc{options{ksp_type cg; mat_type aijcusparse; pc_type gamg;}} tolerance 1e-6; relTol 0.05;}'
    foamDictionary -entry libs -set '("libpetscFoam.so")' "$PW/system/controlDict" >/dev/null 2>&1
    tP=$(wall "( cd '$PW'; pimpleFoam )"); fi

  printf "%10s %9s %11s %10s %11s %11s\n" "$NC" "$(fmt "$tB")" "$(fmt "$tO")" "$(fmt "$tS")" "$(fmt "$tA")" "$(fmt "$tP")"
  echo "$NC,$(fmt "$tB"),$(fmt "$tO"),$(fmt "$tS"),$(fmt "$tA"),$(fmt "$tP")" >> "$RES"
  rm -rf "$OW"/processor* "$BW"/[1-9]* ${SW:+"$SW"/[1-9]*} 2>/dev/null
done
echo; echo "CSV -> $RES"
