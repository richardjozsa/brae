#!/usr/bin/env bash
# Driver ctest for brae_pimpleFoam: end-to-end on a transient variant of the COMMITTED pitzDaily (kEpsilon URANS). Copies
# the case to a work dir, patches it to transient (pimpleFoam + Euler ddt + PIMPLE 2x2 correctors), runs 15 time steps,
# and checks the run is STABLE (bounded continuity, no NaN) and writes a valid OpenFOAM time directory with U + p. This
# exercises the whole driver: controlDict/fvSchemes(ddt)/fvSolution(PIMPLE) parse, turbulence model + field read/write,
# the transient time loop (pimpleStep with fvm::ddt on momentum + k/epsilon), and the OF-format writer.
# Skips (exit 125) if the fixture is absent (SKIP_RETURN_CODE), like ami_oracle.
set -eu
BIN="${1:?brae_pimpleFoam binary}"
SRC="${2:?committed pitzDaily case dir}"
WORK="${3:?work dir}"

if [ ! -f "$SRC/constant/polyMesh/points" ] || [ ! -f "$SRC/0/U" ] || [ ! -f "$SRC/system/fvSchemes" ]; then
    echo "SKIP: fixture '$SRC' not present"; exit 125
fi

rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$SRC/constant" "$SRC/system" "$SRC/0" "$WORK/"

# Transient controlDict with adaptive time stepping.  The global field-write cadence is deliberately sparse,
# while forceCoeffs requests every completed time step: these two independent clocks are a driver contract.
cat > "$WORK/system/controlDict" <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application     pimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         3e-4;
deltaT          2e-5;
adjustTimeStep  yes;
maxCo           0.2;
maxDeltaT       4e-5;
writeControl    timeStep;
writeInterval   15;
writePrecision  8;
functions
{
    coeffs
    {
        type            forceCoeffs;
        patches         (upperWall lowerWall);
        rho             rhoInf;
        rhoInf          1;
        magUInf         10;
        Aref            1;
        lRef            1;
        liftDir         (0 1 0);
        dragDir         (1 0 0);
        pitchAxis       (0 0 1);
        CofR            (0 0 0);
        writeControl    timeStep;
        writeInterval   1;
    }
}
EOF

# Euler ddt (replace the steadyState ddtSchemes block).
python3 - "$WORK/system/fvSchemes" <<'PY'
import re, sys
p = sys.argv[1]; t = open(p).read()
open(p, 'w').write(re.sub(r'ddtSchemes\s*\{[^}]*\}', 'ddtSchemes\n{\n    default         Euler;\n}', t, count=1))
PY

# PIMPLE dict (2 outer x 2 inner) so the transient coupling converges (1x1 is under-resolved -> diverges, expected).
python3 - "$WORK/system/fvSolution" <<'PY'
import sys
p = sys.argv[1]; t = open(p).read()
if 'PIMPLE' not in t:
    t += "\nPIMPLE\n{\n    nOuterCorrectors 2;\n    nCorrectors 2;\n    nNonOrthogonalCorrectors 0;\n}\n"
open(p, 'w').write(t)
PY

LOG="$WORK/run.log"
"$BIN" "$WORK" > "$LOG" 2>&1
rc=$?
if [ "$rc" -ne 0 ]; then echo "FAIL: brae_pimpleFoam exited $rc"; tail -25 "$LOG"; exit 1; fi

# stability: the last reported contLocal must be finite + small. A diverged run explodes it to >> 1 (or NaN -> the awk
# comparison is false), so `< 1.0` catches both.
cont=$(grep -oE "contLocal [0-9.eE+-]+" "$LOG" | tail -1 | awk '{print $2}')
if [ -z "$cont" ]; then echo "FAIL: no contLocal in output"; tail -25 "$LOG"; exit 1; fi
if ! awk -v c="$cont" 'BEGIN{ exit (c < 1.0) ? 0 : 1 }'; then
    echo "FAIL: continuity diverged (contLocal=$cont)"; tail -25 "$LOG"; exit 1
fi

# output: the final time dir (from the driver's "written <dir>" line) must hold U + p with no NaN/inf.
OUT=$(python3 - "$WORK" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
times = []
for path in root.iterdir():
    if not path.is_dir():
        continue
    try: times.append((float(path.name), path))
    except ValueError: pass
print(max(times)[1] if times else '')
PY
)
if [ -z "$OUT" ] || [ ! -f "$OUT/U" ] || [ ! -f "$OUT/p" ]; then
    echo "FAIL: no written U/p time directory"; tail -25 "$LOG"; exit 1
fi
if grep -qiE "\bnan\b|\binf\b" "$OUT/U" "$OUT/p"; then echo "FAIL: NaN/inf in written fields"; exit 1; fi

# Adaptive-time finalisation: the final directory must carry the time that was actually advanced, not a
# fixed-deltaT extrapolation.  Also prove this fixture really varied deltaT, since a fixed-step test cannot
# expose the defect.
python3 - "$LOG" "$OUT" <<'PY'
import math, pathlib, re, sys
log = pathlib.Path(sys.argv[1]).read_text()
times = [float(x) for x in re.findall(r'^Time = ([^ ]+)$', log, re.M)]
assert times, 'no computed Time lines'
steps = [times[0]] + [b - a for a, b in zip(times, times[1:])]
assert max(steps) - min(steps) > 1e-8, f'deltaT did not vary: {steps}'
written = float(pathlib.Path(sys.argv[2]).name)
assert math.isclose(written, times[-1], rel_tol=2e-6, abs_tol=1e-12), \
    f'final directory {written} does not match last computed time {times[-1]}'
PY

# Function-object cadence: dictionary timeStep/1 means exactly one force row for every completed step.  The
# terminal finalisation must not append a duplicate when that last step was already sampled.
python3 - "$LOG" "$WORK/postProcessing/forceCoeffs/0/coefficient.dat" <<'PY'
import math, pathlib, re, sys
log = pathlib.Path(sys.argv[1]).read_text()
times = [float(x) for x in re.findall(r'^Time = ([^ ]+)$', log, re.M)]
rows = [line.split() for line in pathlib.Path(sys.argv[2]).read_text().splitlines()
        if line.strip() and not line.startswith('#')]
assert len(rows) == len(times), f'force rows {len(rows)} != completed steps {len(times)}'
sample_times = [float(row[0]) for row in rows]
assert len(sample_times) == len(set(sample_times)), f'duplicate force sample times: {sample_times}'
assert math.isclose(sample_times[-1], times[-1], rel_tol=2e-6, abs_tol=1e-12), \
    f'last force time {sample_times[-1]} != last computed time {times[-1]}'
PY

# phi output: the face flux must be written as a surfaceScalarField (OF writes phi).
if [ ! -f "$OUT/phi" ] || ! grep -q "surfaceScalarField" "$OUT/phi"; then
    echo "FAIL: phi not written as a surfaceScalarField"; exit 1
fi

# restart: startFrom latestTime must resume from the just-written time dir and advance.
sed -i 's/startFrom       startTime;/startFrom       latestTime;/; s/endTime         3e-4;/endTime         5e-4;/' "$WORK/system/controlDict"
# Exercise the other required function-object clock on restart.  This is dictionary-only selection; no
# BRAE_FORCE_INTERVAL override participates.
sed -i '/coeffs/,/}/{s/writeControl    timeStep;/writeControl    runTime;/; s/writeInterval   1;/writeInterval   5e-5;/;}' "$WORK/system/controlDict"
RLOG="$WORK/restart.log"
"$BIN" "$WORK" > "$RLOG" 2>&1
if [ "$?" -ne 0 ]; then echo "FAIL: restart run exited nonzero"; tail -20 "$RLOG"; exit 1; fi
# The message comes from the SHARED resolveStartTime now, not this driver's own copy: startFrom moved
# into brae::Time, where OF puts it (Time::setControls, Time.C:149). Behaviour verified unchanged --
# the restart still resolves to the written time dir, reads its phi, and advances past it. This driver's
# private copy also only handled `latestTime`, silently ignoring `firstTime`; the shared one does both.
if ! grep -q "starting from time" "$RLOG"; then echo "FAIL: startFrom latestTime did not restart"; tail -20 "$RLOG"; exit 1; fi
# seamless restart: the resume must READ the written phi (surfaceScalarField) rather than recompute it from U. The fresh
# run started at 0/ (no phi) so it did NOT read; the restart's time dir has phi, so this fires only on the restart.
if ! grep -q "read phi from" "$RLOG"; then
    echo "FAIL: restart did not read phi (surfaceScalarField restart path not taken)"; tail -20 "$RLOG"; exit 1; fi

python3 - "$RLOG" "$WORK" <<'PY'
import math, pathlib, re, sys
log = pathlib.Path(sys.argv[1]).read_text()
times = [float(x) for x in re.findall(r'^Time = ([^ ]+)$', log, re.M)]
histories = sorted(pathlib.Path(sys.argv[2]).glob('postProcessing/forceCoeffs/*/coefficient.dat'),
                   key=lambda p: float(p.parent.name))
assert times and len(histories) >= 2, 'runTime cadence restart evidence missing'
rows = [line.split() for line in histories[-1].read_text().splitlines()
        if line.strip() and not line.startswith('#')]
sample_times = [float(row[0]) for row in rows]
assert 1 < len(sample_times) < len(times), \
    f'runTime cadence did not subsample completed steps: samples={len(sample_times)} steps={len(times)}'
assert math.isclose(sample_times[-1], times[-1], rel_tol=2e-6, abs_tol=1e-12), \
    f'runTime terminal sample {sample_times[-1]} != last computed time {times[-1]}'
PY

echo "PASS: brae_pimpleFoam transient kEpsilon stable (contLocal=$cont), wrote $OUT/{U,p,phi}, restart read phi OK"
exit 0
