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
# THE FIXTURE'S OWN INLET, as it ships. This gate used to neutralise sbMatched's flowRateInletVelocity
# because brae's inlet disagreed with OpenFOAM by 2.4e-01 and would have dominated a number meant to be
# about the momentum assembly. That boundary condition is now ported: OpenFOAM recomputes its value in
# updateCoeffs() from the registered rho's PATCH values, at construction (createFields.H builds rho before
# U) and again at every momentum assembly. With that in place the inlet carries EXACTLY the prescribed
# massFlowRate -- sum(phi) = -0.5 against OpenFOAM's -0.5 -- and rAU went 4.58e-05 -> 6.13e-15,
# boundaryCoeffs 4.15e-01 -> 4.89e-16. Neutralising it now would only hide the coupling it exercises.

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
