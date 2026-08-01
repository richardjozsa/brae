#!/usr/bin/env bash
# Regenerate every CSV and every video for the AMG-PCG demo, from the OpenFOAM case up.
#
#   ./demo/amgpcg/run_all.sh                 # both cases, all traces, all videos
#   ./demo/amgpcg/run_all.sh case20          # just the 20-cell teaching case
#
# Prerequisites
#   OpenFOAM v2412 sourced (or edit FOAM_BASHRC below)
#   brae built:      cmake --build build --target trace_brae
#   AMGX at $AMGX_ROOT (default /home/ghost/opt/amgx)
#   python3 with numpy + matplotlib (+ imageio-ffmpeg for MP4)

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEMO="$ROOT/demo/amgpcg"
FOAM_BASHRC="${FOAM_BASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}"
AMGX_ROOT="${AMGX_ROOT:-/home/ghost/opt/amgx}"

CASES=("${@:-case20 case1024}")
read -r -a CASES <<< "${CASES[@]}"

# The 20-cell mesh is below brae's default coarsening target of 64 cells, so it would build a
# zero-level hierarchy. Force it to coarsen so the demo has a real V-cycle to show.
export BRAE_AMG_TARGET=4

echo "==> building the OpenFOAM dump app"
# shellcheck disable=SC1090
source "$FOAM_BASHRC"
( cd "$DEMO/of_apps/dumpAmgTrace" && wmake > /dev/null )

echo "==> building the brae trace tool"
cmake --build "$ROOT/build" --target trace_brae -j 8 > /dev/null

echo "==> building the AMGX trace tool"
gcc -O2 -o "$ROOT/build/trace_amgx" "$DEMO/trace_amgx.c" \
    -I"$AMGX_ROOT/include" -L"$AMGX_ROOT/lib" -lamgxsh \
    -Wl,-rpath,"$AMGX_ROOT/lib" 2> /dev/null

mkdir -p "$DEMO/out"

for CASE in "${CASES[@]}"; do
    DIR="$DEMO/$CASE"
    echo
    echo "================ $CASE ================"

    echo "--> blockMesh"
    ( cd "$DIR" && blockMesh > log.blockMesh 2>&1 )

    echo "--> OpenFOAM GAMG: matrix, rhs, hierarchy, per-cycle trace"
    ( cd "$DIR" && dumpAmgTrace -maxCycles 30 > log.dumpAmgTrace 2>&1 )
    cp "$DIR/trace/of_cycles.csv" "$DIR/trace/of_gamg_cycles.csv"

    echo "--> brae AMG-PCG: hierarchy, Galerkin operators, per-iteration trace"
    "$ROOT/build/trace_brae" "$DIR" 30

    echo "--> brae smoother variants"
    for VARIANT in "brae_cheb:BRAE_CHEBYSHEV=1" \
                   "brae_gs:BRAE_AMG_GS=1" \
                   "brae_soc:BRAE_AMG_SOC=0.05"; do
        NAME="${VARIANT%%:*}"
        ENVS="${VARIANT#*:}"
        env "$ENVS" "$ROOT/build/trace_brae" "$DIR" 30 > /dev/null
        mv "$DIR/trace/brae_cycles.csv" "$DIR/trace/${NAME}_cycles.csv"
    done
    "$ROOT/build/trace_brae" "$DIR" 30 > /dev/null   # restore the default trace

    echo "--> AMGX: per-iteration trace"
    LD_LIBRARY_PATH="$AMGX_ROOT/lib" "$ROOT/build/trace_amgx" "$DIR" 30 | tail -1

    echo "--> expanding iterations into internal steps"
    python3 "$DEMO/make_steps.py" "$DIR" --iters 6

    echo "--> work-normalised comparison"
    python3 "$DEMO/make_compare.py" "$DIR"
done

echo
echo "==> rendering videos from case20"
for SOLVER in brae of_gamg amgx; do
    python3 "$DEMO/replay.py" "$DEMO/case20" --solver "$SOLVER" \
        -o "$DEMO/out/$SOLVER.mp4" --fps 4
done
python3 "$DEMO/replay.py" "$DEMO/case20" --compare -o "$DEMO/out/compare.mp4" --fps 4

echo
echo "done. CSVs in <case>/trace/, videos in $DEMO/out/"
