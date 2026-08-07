#!/bin/bash
# E1: OF's controlDict write cadence on the COMPRESSIBLE driver -- writeControl / writeInterval /
# purgeWrite / deltaT, compared against OpenFOAM by the set of time directories each one produces.
#
# Found by dict_audit, not by reading OF's source. gpuRhoSimpleFoam looked at NONE of these keys: it wrote
# exactly one time directory, at the end. gpuSimpleFoam had the whole mechanism (12 references) and the
# compressible driver had zero, so a compressible case asking to write every 20 iterations silently got
# nothing until convergence -- no intermediate fields, no way to watch a run develop, purgeWrite ignored --
# while the run itself converged to the right answer. That is this project's signature failure: an input
# read off disk by nobody, with plausible output.
#
# The comparison is the DIRECTORY SET, not a field norm, because that is what these keys control. Three
# branches, each a different code path:
#   timeStep + purgeWrite 0 -> 0 20 40 60 80 100 104   (104 = residualControl, so this also pins C5)
#   timeStep + purgeWrite 2 -> 0 100 104               (Foam::Time TimeIO FIFO)
#   runTime  0.5, deltaT 0.1 -> fractional names        (the %g timeName path + the interval latch)
# All three matched OF exactly on first run.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/rhoBoxQ" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_wc_vs_of}

if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true

rm -rf "$WORK"; mkdir -p "$WORK"

# The mesh is shared by every variant, so build it once.
MESH="$WORK/mesh"
cp -r "$SRC" "$MESH"
mkdir -p "$MESH/0" && cp "$MESH"/0.orig/* "$MESH/0/"
( cd "$MESH" && blockMesh > log.blockMesh 2>&1 )

# $1 = tag, $2 = writeControl, $3 = writeInterval, $4 = purgeWrite, $5 = deltaT
setup()
{
    local dir="$1"
    rm -rf "$dir"; cp -r "$SRC" "$dir"
    cp -r "$MESH/constant/polyMesh" "$dir/constant/"
    mkdir -p "$dir/0" && cp "$dir"/0.orig/* "$dir/0/"
    sed -i -e "s/writeControl *[a-zA-Z]*;/writeControl $2;/" \
           -e "s/writeInterval *[0-9.]*;/writeInterval $3;/" \
           -e "s/purgeWrite *[0-9]*;/purgeWrite $4;/" \
           -e "s/deltaT *[0-9.]*;/deltaT $5;/" "$dir/system/controlDict"
}

# Time directory names, sorted numerically. `0.orig` is excluded by name, not by the glob -- [0-9]* matches it.
times()
{
    ls -d "$1"/[0-9]* 2>/dev/null | grep -v '\.orig$' | xargs -n1 basename 2>/dev/null | sort -g | tr '\n' ' '
}

bad=0
run_case()
{
    local tag="$1" wcl="$2" wi="$3" pw="$4" dt="$5"
    setup "$WORK/br_$tag" "$wcl" "$wi" "$pw" "$dt"
    "$BUILD/brae_rhoSimpleFoam" -case "$WORK/br_$tag" > "$WORK/br_$tag/log.brae" 2>&1
    setup "$WORK/of_$tag" "$wcl" "$wi" "$pw" "$dt"
    ( cd "$WORK/of_$tag" && rhoSimpleFoam > log.rhoSimpleFoam 2>&1 )

    local b o
    b=$(times "$WORK/br_$tag")
    o=$(times "$WORK/of_$tag")
    if [ -z "$o" ]; then
        echo "  $tag: FAIL OpenFOAM produced no time directories -- the case stopped testing anything"
        bad=$((bad + 1))
        return
    fi
    # More than the start directory must exist, else "they agree" is vacuous: both would trivially
    # produce just `0` if the write cadence were ignored on BOTH sides.
    local n
    n=$(echo "$o" | wc -w)
    if [ "$n" -lt 3 ]; then
        echo "  $tag: FAIL OpenFOAM wrote only $n director(y|ies) [$o] -- too few to test a cadence"
        bad=$((bad + 1))
        return
    fi
    if [ "$b" = "$o" ]; then
        echo "  $tag: OK  [$o]"
    else
        echo "  $tag: FAIL"
        echo "        brae: [$b]"
        echo "        OF  : [$o]"
        bad=$((bad + 1))
    fi
}

run_case "timestep"  timeStep 20  0 1
run_case "purge"     timeStep 20  2 1
run_case "runtime"   runTime  0.5 0 0.1

echo "wc_vs_openfoam: $bad failures over 3 cadences"
[ "$bad" -eq 0 ] || exit 1
