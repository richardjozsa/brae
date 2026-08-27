#!/usr/bin/env bash
# End-to-end simpleFoam yPlus output contract on a fresh retained kOmegaSST case.
set -eu

BIN="$1"
ARCHIVE="${2:-}"
if [ -z "$ARCHIVE" ] || [ ! -f "$ARCHIVE" ]; then
    echo "SKIP: retained Issue 37 archive variable is empty or archive is absent"
    exit 125
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "SKIP: nvidia-smi not available"
    exit 125
fi
nvidia-smi -L >/dev/null
TMPDIR=$(printenv TMPDIR 2>/dev/null || true)
if [ -z "$TMPDIR" ]; then TMPDIR=/tmp; fi
WORK=$(mktemp -d "$TMPDIR/brae-yplus-output.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
CASE="$WORK/case"
mkdir "$CASE"
tar -xf "$ARCHIVE" -C "$CASE"

cat > "$CASE/system/controlDict" <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application simpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 3;
deltaT 1;
writeControl timeStep;
writeInterval 3;
purgeWrite 2;
writeFormat ascii;
writePrecision 8;
writeCompression off;
timeFormat general;
timePrecision 6;
functions
{
    yPlus
    {
        type yPlus;
        libs ("libfieldFunctionObjects.so");
        executeControl timeStep;
        executeInterval 1;
        writeControl writeTime;
    }
    unknownFunction
    {
        type definitelyUnknownFunctionObject;
    }
}
EOF

OF_GZ="$CASE/2000/yPlus.gz"
OF_DAT="$CASE/postProcessing/yPlus/0/yPlus.dat"
mkdir -p "$CASE/postProcessing/yPlus/0"
printf '# retained OpenFOAM yPlus bytes\n' > "$OF_DAT"
gz_before=$(sha256sum "$OF_GZ" | awk '{print $1}')
dat_before=$(sha256sum "$OF_DAT" | awk '{print $1}')
"$BIN" -case "$CASE" > "$WORK/run.log" 2>&1

OUT="$CASE/postProcessing/braeYPlus/yPlus/0"
[ -f "$OUT/faceValues.dat" ]
[ -f "$OUT/patchSummary.dat" ]
[ -f "$OUT/metadata.dat" ]
python3 - "$OUT/faceValues.dat" "$OUT/patchSummary.dat" "$OUT/metadata.dat" "$WORK/run.log" <<'PY'
import math, sys
face, summary, meta, log = sys.argv[1:]
flines = [x.split() for x in open(face) if x.strip() and not x.startswith('#')]
slines = [x.split() for x in open(summary) if x.strip() and not x.startswith('#')]
assert flines and slines
assert all(len(x) == 8 for x in flines), flines[:2]
assert all(len(x) == 9 for x in slines), slines[:2]
assert {x[2] for x in flines} == {'body', 'lowerWall'}
assert {x[2] for x in slines} == {'body', 'lowerWall'}
assert {int(x[1]) for x in flines} == {3}
assert {int(x[1]) for x in slines} == {3}
assert all(float(x[5]) > 0 and float(x[6]) > 0 and math.isfinite(float(x[7])) and float(x[7]) >= 0 for x in flines)
assert all(0 <= float(x[8]) <= 100 for x in slines)
m = open(meta).read()
assert 'provenance=Brae-owned' in m and 'openfoam_yPlus_field_read=false' in m
assert 'stopping_reason=iteration_limit' in m and 'completed_iteration=3' in m and 'sample_count=1' in m
run = open(log).read()
assert 'brae NOTICE [solver-owned] functions/yPlus' in run
assert 'brae NOTICE [ignored] functions/unknownFunction' in run
assert 'brae NOTICE [ignored] functions/yPlus' not in run
assert 'yPlus sample iteration 3:' in run
print('Brae faceValues.dat and patchSummary.dat schemas/finite walls/final sample: OK')
print('solver-owned yPlus and unknown-function status: OK')
print('completion metadata: OK')
PY

[ "$(sha256sum "$OF_GZ" | awk '{print $1}')" = "$gz_before" ]
[ "$(sha256sum "$OF_DAT" | awk '{print $1}')" = "$dat_before" ]
echo "pre-existing OpenFOAM yPlus.gz and yPlus.dat remain byte-identical: OK"

COLLISION="$OUT/faceValues.dat"
printf 'collision sentinel\n' > "$COLLISION"
if "$BIN" -case "$CASE" > "$WORK/collision.log" 2>&1; then
    echo "FAIL: Brae yPlus collision was accepted"
    exit 1
fi
grep -q "Brae output already exists" "$WORK/collision.log"
grep -qx 'collision sentinel' "$COLLISION"
echo "Brae-owned yPlus collision fails closed and preserves bytes: OK"

NOCASE="$WORK/no-object"
mkdir "$NOCASE"
tar -xf "$ARCHIVE" -C "$NOCASE"
sed 's/endTime 2000;/endTime 1;/' "$NOCASE/system/controlDict" | sed '/functions/,$d' > "$NOCASE/system/controlDict.tmp"
cat >> "$NOCASE/system/controlDict.tmp" <<'EOF'
functions
{
}
EOF
mv "$NOCASE/system/controlDict.tmp" "$NOCASE/system/controlDict"
"$BIN" -case "$NOCASE" > "$WORK/no-object.log" 2>&1
if [ -e "$NOCASE/postProcessing/braeYPlus" ]; then
    echo "FAIL: no configured yPlus object created Brae output"
    exit 1
fi
echo "no configured yPlus object creates no Brae output: OK"
grep 'yPlus sample iteration' "$WORK/run.log"

# Unsupported yPlus must be an observable refusal, not a solver refusal. Use a fresh retained mesh with the
# model switched in the scratch extraction only; the omega field is a shape-compatible seed for epsilon and its
# wall-function names are changed in the scratch copy. The retained archive and all external evidence stay read-only.
KECASE="$WORK/kepsilon"
mkdir "$KECASE"
tar -xf "$ARCHIVE" -C "$KECASE"
sed -e 's/object omega;/object epsilon;/' -e 's/omegaWallFunction/epsilonWallFunction/g' \
    "$KECASE/0/omega" > "$KECASE/0/epsilon"
sed -i 's/model[[:space:]]*kOmegaSST;/model kEpsilon;/' \
    "$KECASE/constant/turbulenceProperties" "$KECASE/constant/momentumTransport"
cat > "$KECASE/system/controlDict" <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application simpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
writeFormat ascii;
writePrecision 8;
writeCompression off;
timeFormat general;
timePrecision 6;
functions
{
    yPlus
    {
        type yPlus;
        libs ("libfieldFunctionObjects.so");
        writeControl writeTime;
    }
    forceCoeffs
    {
        type forceCoeffs;
        patches (body);
        rho rhoInf;
        rhoInf 1.0;
        liftDir (0 0 1);
        dragDir (1 0 0);
        CofR (-0.502 0 0);
        pitchAxis (0 1 0);
        magUInf 1.0;
        lRef 1.04;
        Aref 0.112032;
    }
}
EOF
"$BIN" -case "$KECASE" > "$WORK/kepsilon.log" 2>&1
[ ! -e "$KECASE/postProcessing/braeYPlus" ]
grep -q "functions/yPlus: yPlus output refused: support is limited to incompressible RAS kOmegaSST" "$WORK/kepsilon.log"
grep -q "Time = 1" "$WORK/kepsilon.log"
grep -q "Cd=" "$WORK/kepsilon.log"
[ -f "$KECASE/postProcessing/braeForceHistory/forceCoeffs/0/coefficient.dat" ]
echo "unsupported kEpsilon yPlus refuses loudly while Cd/residual solve/force history continue: OK"

# A sample-time evaluation failure is an observable failure, not a SIMPLE failure: force history must retain the
# solver's terminal reason and must never be stamped numerical_failure merely because yPlus could not be written.
FAILCASE="$WORK/failure"
mkdir "$FAILCASE"
tar -xf "$ARCHIVE" -C "$FAILCASE"
cat > "$FAILCASE/system/controlDict" <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application simpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 1;
deltaT 1;
writeControl timeStep;
writeInterval 1;
writeFormat ascii;
writePrecision 8;
writeCompression off;
timeFormat general;
timePrecision 6;
functions
{
    yPlus { type yPlus; writeControl writeTime; }
    forceCoeffs
    {
        type forceCoeffs;
        patches (body);
        rho rhoInf;
        rhoInf 1.0;
        liftDir (0 0 1);
        dragDir (1 0 0);
        CofR (-0.502 0 0);
        pitchAxis (0 1 0);
        magUInf 1.0;
        lRef 1.04;
        Aref 0.112032;
    }
}
EOF
BRAE_TEST_FAIL_YPLUS_SAMPLE=1 "$BIN" -case "$FAILCASE" > "$WORK/failure.log" 2>&1
FAILY="$FAILCASE/postProcessing/braeYPlus/yPlus/0/metadata.dat"
FAILF="$FAILCASE/postProcessing/braeForceHistory/forceCoeffs/0/coefficient.dat.meta"
grep -q 'stopping_reason=yplus_output_failure' "$FAILY"
grep -q 'completed_iteration=1' "$FAILY"
grep -q 'sample_count=0' "$FAILY"
grep -q 'stopping_reason=iteration_limit' "$FAILF"
if grep -q 'stopping_reason=numerical_failure' "$FAILF"; then
    echo "FAIL: yPlus evaluation failure changed force-history stopping reason"
    exit 1
fi
grep -q 'SIMPLE reached endTime (1 iterations)' "$WORK/failure.log"
echo "yPlus evaluation failure leaves solver/force-history stopping reason untouched: OK"
