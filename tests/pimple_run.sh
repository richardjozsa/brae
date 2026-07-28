#!/usr/bin/env bash
# Driver ctest for brae_pimpleFoam: end-to-end on a transient variant of the COMMITTED pitzDaily (kEpsilon URANS). Copies
# the case to a work dir, patches it to transient (pimpleFoam + Euler ddt + PIMPLE 2x2 correctors), runs 15 time steps,
# and checks the run is STABLE (bounded continuity, no NaN) and writes a valid OpenFOAM time directory with U + p. This
# exercises the whole driver: controlDict/fvSchemes(ddt)/fvSolution(PIMPLE) parse, turbulence model + field read/write,
# the transient time loop (pimpleStep with fvm::ddt on momentum + k/epsilon), and the OF-format writer.
# Skips (exit 125) if the fixture is absent (SKIP_RETURN_CODE), like ami_oracle.
set -u
BIN="${1:?brae_pimpleFoam binary}"
SRC="${2:?committed pitzDaily case dir}"
WORK="${3:?work dir}"

if [ ! -f "$SRC/constant/polyMesh/points" ] || [ ! -f "$SRC/0/U" ] || [ ! -f "$SRC/system/fvSchemes" ]; then
    echo "SKIP: fixture '$SRC' not present"; exit 125
fi

rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$SRC/constant" "$SRC/system" "$SRC/0" "$WORK/"

# transient controlDict (physical time): 15 steps of 2e-5 s -> endTime 3e-4, CFL ~ 0.2 on pitzDaily (U~10, cell~1e-3).
cat > "$WORK/system/controlDict" <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application     pimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         3e-4;
deltaT          2e-5;
writeControl    timeStep;
writeInterval   15;
writePrecision  8;
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
OUT=$(grep -oE "written [^ ]+" "$LOG" | tail -1 | awk '{print $2}')
if [ -z "$OUT" ] || [ ! -f "$OUT/U" ] || [ ! -f "$OUT/p" ]; then
    echo "FAIL: no written U/p time directory"; tail -25 "$LOG"; exit 1
fi
if grep -qiE "\bnan\b|\binf\b" "$OUT/U" "$OUT/p"; then echo "FAIL: NaN/inf in written fields"; exit 1; fi

# phi output: the face flux must be written as a surfaceScalarField (OF writes phi).
if [ ! -f "$OUT/phi" ] || ! grep -q "surfaceScalarField" "$OUT/phi"; then
    echo "FAIL: phi not written as a surfaceScalarField"; exit 1
fi

# restart: startFrom latestTime must resume from the just-written time dir and advance.
sed -i 's/startFrom       startTime;/startFrom       latestTime;/; s/endTime         3e-4;/endTime         5e-4;/' "$WORK/system/controlDict"
RLOG="$WORK/restart.log"
"$BIN" "$WORK" > "$RLOG" 2>&1
if [ "$?" -ne 0 ]; then echo "FAIL: restart run exited nonzero"; tail -20 "$RLOG"; exit 1; fi
if ! grep -q "resuming from time" "$RLOG"; then echo "FAIL: startFrom latestTime did not restart"; tail -20 "$RLOG"; exit 1; fi
# seamless restart: the resume must READ the written phi (surfaceScalarField) rather than recompute it from U. The fresh
# run started at 0/ (no phi) so it did NOT read; the restart's time dir has phi, so this fires only on the restart.
if ! grep -q "read phi from" "$RLOG"; then
    echo "FAIL: restart did not read phi (surfaceScalarField restart path not taken)"; tail -20 "$RLOG"; exit 1; fi

echo "PASS: brae_pimpleFoam transient kEpsilon stable (contLocal=$cont), wrote $OUT/{U,p,phi}, restart read phi OK"
exit 0
