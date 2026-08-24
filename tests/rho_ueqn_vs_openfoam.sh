#!/usr/bin/env bash
# rhoSimpleFoam's UEqn.H against REAL OpenFOAM's own assembled momentum matrix.
#
# THE ORACLE is tools/dumpPEqn: OpenFOAM's rhoSimpleFoam carrying a stage harness that writes, at SIMPLE
# iteration BRAE_DUMP_STAGE_ITER, the momentum equation's observable content --
#
#   stage_rAU    1/UEqn.A()                        the diagonal, AFTER relax()
#   stage_UIC    UEqn.internalCoeffs() per patch
#   stage_UBC    UEqn.boundaryCoeffs()  per patch
#   stage_muEff  turbulence->muEff()                the DYNAMIC viscosity the assembly used
#
# Dumping at iteration 1 means the state assembled from is the START-TIME field set, which brae can
# reconstruct exactly through createFields -- so the comparison isolates UEqn.H and carries no accumulated
# trajectory difference.
#
# stage_muEff IS INJECTED INTO brae rather than recomputed. brae has no ported compressible turbulence
# model yet (a separate manifest component), and mixing an unported closure into this measurement would
# produce a number that cannot be attributed to either. With OpenFOAM's own muEff supplied, a failure here
# means the momentum ASSEMBLY is wrong and nothing else.
#
# THE CONTROL, which is the reason this gate exists at all: the binary also assembles with the KINEMATIC
# nu_eff -- what the incompressible divDevReff carries -- and requires that to DISAGREE with OpenFOAM.
# The factor of rho between the two is the entire difference between this solver's momentum equation and
# simpleFoam's, and it is a factor that looks plausible in every field plot, so a bound both forms passed
# would gate nothing.
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BIN="${BUILD:-$ROOT/build}/test_rho_ueqn_cpp"
SRC="$ROOT/validation/sbMatched"
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}

[ -x "$BIN" ]      || { echo "SKIP: $BIN not built"; exit 77; }
[ -d "$SRC" ]      || { echo "SKIP: fixture $SRC missing"; exit 77; }
[ -f "$OFBASHRC" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC" "$W/case" || exit 1
rm -rf "$W"/case/[1-9]* "$W"/case/0 "$W"/case/processor* "$W"/case/log.*

set +u
# shellcheck disable=SC1091
source "$OFBASHRC" > /dev/null 2>&1 || true
set -u

DUMP="$(command -v dumpPEqn || true)"
[ -z "$DUMP" ] && [ -n "${FOAM_USER_APPBIN:-}" ] && [ -x "$FOAM_USER_APPBIN/dumpPEqn" ] \
    && DUMP="$FOAM_USER_APPBIN/dumpPEqn"
[ -n "$DUMP" ] || { echo "SKIP: dumpPEqn not built -- (cd tools/dumpPEqn && wmake)"; exit 77; }
[ -d "$W/case/constant/polyMesh" ] || { echo "SKIP: fixture ships no mesh"; exit 77; }

cp -r "$W/case/0.orig" "$W/case/0"

# THE INLET IS REPLACED BY A PLAIN fixedValue, and that is deliberate isolation, not convenience.
# sbMatched's inlet is flowRateInletVelocity, whose value OpenFOAM derives from the mass flow rate and
# whichever rho it is handed. brae disagrees with OpenFOAM there by ~2.4e-01 on this case -- a real finding,
# but one belonging to that boundary condition, which is its own manifest component. Left in place it would
# put a BC error inside a number that is supposed to say whether UEqn.H is assembled correctly, and a
# number covering two components cannot be attributed to either.
#
# A uniform non-zero value in all three components is used so the inlet still exercises boundaryCoeffs in
# every component rather than being trivially zero. It need not be physical: what is under test is the
# assembly of a matrix from a given boundary state, and both codes are given the SAME state.
python3 - "$W/case/0/U" <<'PYEOF'
import re, sys
p = sys.argv[1]
s = open(p).read()
s = re.sub(r'(inlet\s*\{)[^}]*\}',
           r'\1\n        type            fixedValue;\n        value           uniform (1 2 3);\n    }',
           s, count=1)
open(p, 'w').write(s)
PYEOF
grep -q "fixedValue" "$W/case/0/U" || { echo "FAIL: could not neutralise the inlet BC"; exit 1; }

python3 - "$W/case" <<'PYEOF'
import os, re, sys
c = os.path.join(sys.argv[1], 'system/controlDict')
s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'^writeFormat .*',    'writeFormat     ascii;',    s, flags=re.M)
s = re.sub(r'^writePrecision .*', 'writePrecision  15;',       s, flags=re.M)
s = re.sub(r'^endTime .*',        'endTime         1;',        s, flags=re.M)
s = re.sub(r'^writeInterval .*',  'writeInterval   1;',        s, flags=re.M)
s = re.sub(r'^writeControl .*',   'writeControl    timeStep;', s, flags=re.M)
open(c, 'w').write(s)
PYEOF

# One SIMPLE iteration, dumping the momentum stages from it.
( cd "$W/case" && BRAE_DUMP_STAGE_ITER=1 "$DUMP" > dump.log 2>&1 ) \
    || { echo "FAIL: dumpPEqn did not run"; tail -20 "$W/case/dump.log"; exit 1; }
for f in stage_rAU stage_UIC stage_UBC stage_muEff stage_Uass; do
    [ -f "$W/case/1/$f" ] \
        || { echo "FAIL: dumpPEqn wrote no 1/$f"; tail -20 "$W/case/dump.log"; exit 1; }
done

"$BIN" "$W/case" 0 1
