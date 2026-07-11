#!/usr/bin/env bash
# ============================================================================
#  SPUMA pressure-solver shootout — which SPUMA config is fastest on THIS GPU.
#
#  Runs the SAME laminar scaled-pitzDaily case through SPUMA with several
#  pressure-solver configurations and prints ms/iteration for each, to see
#  whether SPUMA's aDIC/aDILU async path beats its GAMG + twoStageGaussSeidel
#  path. It does on discrete GPUs (the CPU-bound default smoother thrashes
#  managed memory over PCIe); on unified-memory GB10 the trade-off differs, so
#  RUN THIS ON THE TARGET GPU and read the result as that GPU's answer.
#
#  Only the pressure (p) solver is varied; U/k/epsilon are held at the GPU
#  smoother so the comparison isolates the pressure solve (the dominant cost).
#  The pressure matrix is symmetric, so aDIC pairs with PCG (aDILU/PBiCGStab is
#  for the asymmetric momentum/turbulence equations).
#
#  Usage:   SPUMA_BIN=/path/to/spuma/.../bin/simpleFoam ./spuma_solver_compare.sh
#  Env:     SIZE=6     blockMesh scale factor M (cells ~= 12225 * M^2)
#           ITERS=50   SPUMA_POOL=24   NVARCH=<cc>   WORK=/tmp/spuma_cmp
# ============================================================================
set -u
SPUMA_BIN="${SPUMA_BIN:-/home/ghost/space/spuma-fresh/platforms/linuxARM64NvidiaDPInt32Opt/bin/simpleFoam}"
SIZE="${SIZE:-6}"; ITERS="${ITERS:-50}"; SPUMA_POOL="${SPUMA_POOL:-24}"; WORK="${WORK:-/tmp/spuma_cmp}"
NVARCH="${NVARCH:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d .)}"
OFBASHRC="${OFBASHRC:-$(ls /usr/lib/openfoam/openfoam*/etc/bashrc /opt/openfoam*/etc/bashrc 2>/dev/null | head -1)}"
[ -x "$SPUMA_BIN" ] || { echo "ERROR: SPUMA_BIN not found/executable: $SPUMA_BIN"; exit 1; }
set +u; source "$OFBASHRC" >/dev/null 2>&1; set -u
command -v blockMesh >/dev/null || { echo "ERROR: OpenFOAM not sourced (set OFBASHRC=...)"; exit 1; }
SP_SRC="$(cd "$(dirname "$SPUMA_BIN")/../../.." && pwd)"

# --- build the case once (laminar scaled pitzDaily, fixed iters) ---
SRC="$WORK/case"; rm -rf "$WORK"; mkdir -p "$SRC"
cp -r "$FOAM_TUTORIALS/incompressible/simpleFoam/pitzDaily/"{0,constant,system} "$SRC"/
foamDictionary -entry simulationType         -set laminar   "$SRC/constant/turbulenceProperties" >/dev/null 2>&1
foamDictionary -entry functions              -remove        "$SRC/system/controlDict"  >/dev/null 2>&1
foamDictionary -entry SIMPLE/residualControl -remove        "$SRC/system/fvSolution"   >/dev/null 2>&1
foamDictionary -entry endTime       -set "$ITERS" "$SRC/system/controlDict" >/dev/null 2>&1
foamDictionary -entry writeInterval -set "$ITERS" "$SRC/system/controlDict" >/dev/null 2>&1
python3 - "$SRC/system/blockMeshDict" "$SIZE" <<'PY'
import re,sys; f,M=sys.argv[1],int(sys.argv[2]); o=[]
for ln in open(f):
    m=re.fullmatch(r'\((\d+)\s+(\d+)\s+(\d+)\)',ln.strip())
    o.append(re.sub(r'\(\d+\s+\d+\s+\d+\)',f'({int(m[1])*M} {int(m[2])*M} {int(m[3])})',ln) if m else ln)
open(f,'w').writelines(o)
PY
( cd "$SRC" && blockMesh >/dev/null 2>&1 )
NC=$(grep -aoE 'nCells:[0-9]+' "$SRC/constant/polyMesh/owner" | grep -oE '[0-9]+' | head -1)

# --- configs: label | full p-solver fvSolution block ---
LABELS=(); PBLOCKS=()
add(){ LABELS+=("$1"); PBLOCKS+=("$2"); }
add "GAMG + twoStageGaussSeidel (benchmark)" '{solver GAMG; smoother twoStageGaussSeidel; nCellsInCoarsestLevel 10; tolerance 1e-6; relTol 0.1;}'
add "GAMG + GaussSeidel (stock default)"     '{solver GAMG; smoother GaussSeidel; nCellsInCoarsestLevel 10; tolerance 1e-6; relTol 0.1;}'
add "PCG + aDIC (SPUMA async precond)"       '{solver PCG; preconditioner aDIC; tolerance 1e-6; relTol 0.1;}'
add "PCG + DIC (stock precond)"              '{solver PCG; preconditioner DIC; tolerance 1e-6; relTol 0.1;}'
add "GAMG + aDIC (async smoother)"           '{solver GAMG; smoother aDIC; nCellsInCoarsestLevel 10; tolerance 1e-6; relTol 0.1;}'

echo "SPUMA solver shootout | bin=$SPUMA_BIN"
echo "GPU cc=$NVARCH | cells=$NC | iters=$ITERS | pool=${SPUMA_POOL}G | only p varies (U/k/eps = twoStageSymGaussSeidel)"
printf "\n%-44s %10s %9s\n" "p-solver config" "ms/iter" "vs base"
printf   "%-44s %10s %9s\n" "---------------" "-------" "-------"
BASE=""
for i in "${!LABELS[@]}"; do
  CW="$WORK/run_$i"; cp -r "$SRC" "$CW"
  foamDictionary -entry solvers/p -set "${PBLOCKS[$i]}" "$CW/system/fvSolution" >/dev/null 2>&1
  foamDictionary -entry 'solvers/"(U|k|epsilon|omega|f|v2)"/smoother' -set twoStageSymGaussSeidel "$CW/system/fvSolution" >/dev/null 2>&1
  a=$(date +%s.%N)
  ( export have_cuda=true NVARCH="$NVARCH" FOAM_SIGFPE=false; set +u; source "$SP_SRC/etc/bashrc" >/dev/null 2>&1; set -u
    cd "$CW" && "$SPUMA_BIN" -pool fixedSizeMemoryPool -poolSize "$SPUMA_POOL" > log 2>&1 )
  b=$(date +%s.%N)
  its=$(grep -c '^Time = ' "$CW/log" 2>/dev/null || echo 0)
  if grep -qiE 'FATAL|Unknown (preconditioner|smoother|solver)|not.*implemented|-nan|nan,' "$CW/log" 2>/dev/null; then
    why=$(grep -iE 'Unknown (preconditioner|smoother|solver)|FATAL|not.*implemented' "$CW/log" | head -1 | sed 's/^[^:]*://' | cut -c1-30)
    printf "%-44s %10s %9s\n" "${LABELS[$i]}" "-" "FAIL:${why}"
  elif [ "${its:-0}" -ge "$ITERS" ]; then
    ms=$(echo "scale=1; ($b - $a)/$ITERS*1000" | bc)
    [ -z "$BASE" ] && BASE="$ms"
    rel=$(echo "scale=2; $ms/$BASE" | bc)
    printf "%-44s %10s %9s\n" "${LABELS[$i]}" "$ms" "${rel}x"
  else
    printf "%-44s %10s %9s\n" "${LABELS[$i]}" "-" "${its:-0}/${ITERS} it"
  fi
  rm -rf "$CW"
done
echo
echo "Lower ms/iter is better. 'vs base' is relative to the first row (the benchmark's config)."
echo "NOTE: on unified-memory GB10 this is the GB10 answer; discrete GPUs (H100/L40S) may rank differently."
