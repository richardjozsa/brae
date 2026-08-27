#!/usr/bin/env bash
# Real simpleFoam force-history contract. Both timing arms are generated from the same controlDict; the only
# difference is the forceCoeffs object itself. The pitzDaily fixture has actual wall faces and k-epsilon fields.
set -eu

BIN="${1:?brae binary}"
SRC="${2:?pitzDaily fixture}"
WORK="${3:?work directory}"
if [ ! -f "$SRC/constant/polyMesh/points" ] || [ ! -f "$SRC/0/U" ]; then
    echo "SKIP: fixture '$SRC' not present"
    exit 125
fi

rm -rf "$WORK"
mkdir -p "$WORK"
trap 'rm -rf "$WORK"' EXIT

make_case() {
    local d="$1"
    local with_force="$2"
    mkdir -p "$d"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0" "$d/"
    cat > "$d/system/controlDict" <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application simpleFoam;
startFrom startTime;
startTime 0;
stopAt endTime;
endTime 2000;
deltaT 1;
writeControl timeStep;
writeInterval 3;
writeFormat ascii;
writePrecision 16;
functions
{
EOF
    if [ "$with_force" = 1 ]; then
        cat >> "$d/system/controlDict" <<'EOF'
    coeffs
    {
        type forceCoeffs;
        patches (upperWall lowerWall);
        rho rhoInf;
        rhoInf 1;
        magUInf 10;
        Aref 1;
        lRef 1;
        liftDir (0 1 0);
        dragDir (1 0 0);
        pitchAxis (0 0 1);
        CofR (0 0 0);
    }
EOF
    fi
    cat >> "$d/system/controlDict" <<'EOF'
}
EOF
}

set_end_time() {
    local d="$1"
    # Keep the source text at 2000 so a whitespace drift cannot silently leave a long run under ctest.
    sed -i -E 's/^endTime[[:space:]]+2000;/endTime         3;/' "$d/system/controlDict"
    if ! grep -Eq '^endTime[[:space:]]+3;' "$d/system/controlDict"; then
        echo "FAIL: endTime substitution did not take effect in $d/system/controlDict"
        exit 1
    fi
}

run_timed() {
    local d="$1"
    local stamp="$2"
    /usr/bin/time -f '%e' -o "$WORK/$stamp.seconds" "$BIN" -case "$d" > "$WORK/$stamp.log" 2>&1
}

CASE="$WORK/force"
make_case "$CASE" 1
set_end_time "$CASE"
mkdir -p "$CASE/postProcessing/forceCoeffs/0"
cat > "$CASE/postProcessing/forceCoeffs/0/coefficient.dat" <<'EOF'
# stale OpenFOAM sample that must survive untouched
0 999 999 999
EOF
STALE="$(cat "$CASE/postProcessing/forceCoeffs/0/coefficient.dat")"
run_timed "$CASE" force

HISTORY="$CASE/postProcessing/braeForceHistory/coeffs/0/coefficient.dat"
META="$HISTORY.meta"
python3 - "$HISTORY" "$META" "$CASE/postProcessing/forceCoeffs/0/coefficient.dat" "$WORK/force.log" <<'PY'
import math, re, sys
history, meta, stale, log_path = sys.argv[1:]
lines = open(history).read().splitlines()
assert lines[0].startswith('# Brae simpleFoam forceCoeffs history (Brae-owned; object=coeffs)'), lines[:3]
assert any(x.startswith('# columns: Time Cd Cl Cm Fx Fy Fz Fpx Fpy Fpz Fvx Fvy Fvz wallTime Iteration') for x in lines)
assert any(x.startswith('# column contract: Time=') and 'Cd/Cl/Cm=normalized' in x for x in lines)
assert any(x.startswith('# repeatability: GPU wall-face reduction uses atomicAdd') for x in lines)
data = [x.split() for x in lines if x and not x.startswith('#')]
assert len(data) == 3, data
assert all(len(row) == 15 for row in data), data
assert all(math.isfinite(float(v)) for row in data for v in row), data
assert [float(row[0]) for row in data] == [1.0, 2.0, 3.0], data
assert [int(row[14]) for row in data] == [1, 2, 3], data
assert all(float(a[0]) < float(b[0]) for a, b in zip(data, data[1:])), data
assert all(float(a[14]) < float(b[14]) for a, b in zip(data, data[1:])), data
assert open(stale).read().strip() == '# stale OpenFOAM sample that must survive untouched\n0 999 999 999'
metadata = open(meta).read()
assert 'stopping_reason=iteration_limit' in metadata, metadata
assert 'sample_count=3' in metadata and 'completed_iterations=3' in metadata, metadata
log = open(log_path).read()
assert 'forceCoeffs: coeffs -> ' in log and 'braeForceHistory/coeffs/0/coefficient.dat' in log, log
assert 'brae NOTICE [ignored] functions/coeffs' not in log, log
assert log.count('brae NOTICE [solver-owned] functions/coeffs') == 1, log
assert 'sampled per completed SIMPLE iteration' in log and 'not OpenFOAM-identical' in log, log
m = re.findall(r'forceCoeffs .*Cd=([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)', log)
assert m, log[-1000:]
console_cd = float(m[-1]); history_cd = float(data[-1][1])
assert abs(console_cd-history_cd) <= max(5e-7, abs(history_cd)*5e-7), (console_cd, history_cd)
print('one sample per completed iteration: OK')
print('Brae-owned path preserves OpenFOAM coefficient.dat: OK')
print('self-identifying 15-column parse, finite and monotonic: OK')
print('final console Cd matches history Cd (tolerance 5e-7 relative/absolute): OK')
PY

# A stale Brae-owned output is a collision, not permission to truncate it.
COLLISION="$WORK/collision"
make_case "$COLLISION" 1
set_end_time "$COLLISION"
mkdir -p "$COLLISION/postProcessing/braeForceHistory/coeffs/0"
printf 'sentinel\n' > "$COLLISION/postProcessing/braeForceHistory/coeffs/0/coefficient.dat"
if "$BIN" -case "$COLLISION" > "$WORK/collision.log" 2>&1; then
    echo "FAIL: pre-existing Brae history output was accepted"
    exit 1
fi
grep -q "Brae history output already exists" "$WORK/collision.log"
grep -qx 'sentinel' "$COLLISION/postProcessing/braeForceHistory/coeffs/0/coefficient.dat"
echo "pre-existing Brae history refuses overwrite and preserves bytes: OK"

# No forceCoeffs object means no coefficient file is invented. This is the same generated global controlDict.
NOCASE="$WORK/no-force"
make_case "$NOCASE" 0
set_end_time "$NOCASE"
run_timed "$NOCASE" no-force
if [ -e "$NOCASE/postProcessing/forceCoeffs/0/coefficient.dat" ] || [ -e "$NOCASE/postProcessing/braeForceHistory" ]; then
    echo "FAIL: no forceCoeffs object created coefficient output"
    exit 1
fi
echo "no forceCoeffs object creates no coefficient output: OK"

# A missing required normalization value must stop before a misleading run.
BAD="$WORK/missing"
make_case "$BAD" 1
set_end_time "$BAD"
sed -i '/Aref 1;/d' "$BAD/system/controlDict"
if "$BIN" -case "$BAD" > "$WORK/missing.log" 2>&1; then
    echo "FAIL: missing Aref was accepted"
    exit 1
fi
grep -q "forceCoeffs 'coeffs': required entry 'Aref' is missing" "$WORK/missing.log"
echo "missing required Aref fails clearly: OK"

python3 - "$WORK/force.seconds" "$WORK/no-force.seconds" <<'PY'
import sys
force, no_force = (float(open(p).read()) for p in sys.argv[1:])
overhead = force - no_force
print(f"measured same-controlDict 3-iteration wall time: forceCoeffs={force:.2f}s no-force={no_force:.2f}s overhead={overhead:+.2f}s overhead_per_iteration={overhead/3:+.3f}s")
PY
echo "force-history generated file:"
sed -n '1,8p' "$HISTORY"
echo "force run wall seconds: $(cat "$WORK/force.seconds")"
