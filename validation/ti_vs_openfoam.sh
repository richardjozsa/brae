#!/bin/bash
# brae_rhoSimpleFoam vs OpenFOAM with TURBULENT INLET BCs on a flowRateInletVelocity inlet.
#
# OF re-evaluates these in updateCoeffs every outer iteration:
#     turbulentIntensityKineticEnergyInlet      refValue = 1.5*I^2*|Up|^2
#     turbulentMixingLengthFrequencyInlet       refValue = sqrt(k_p)/(Cmu^0.25 * L)
#     turbulentMixingLengthDissipationRateInlet refValue = (Cmu^0.75/L) * k_p^1.5
# brae computed them ONCE, on the host, at set-up. The formulas were right; the timing was not.
#
# That is exact for a fixedValue U inlet -- Up never moves, so one evaluation is the same as a thousand.
# It is wrong for flowRateInletVelocity, where OF rebuilds Up each iteration from the LIVE boundary
# density: at set-up brae only has the seed density (`rhoInlet`, or 1.0 when absent), so the frozen inlet
# |U| is off by the ratio of seed to converged density, and k_inlet is off by its square.
#
# The case makes that gap deliberate: massFlowRate 0.5885 kg/s through a 0.2 x 0.05 m inlet with
# `rhoInlet 1.0`, while the converged density is ~1.18 kg/m^3. Seed U = 58.9 m/s vs converged ~50 m/s,
# so a frozen inlet k is ~1.39x too large.
#
# Measured on this case (brae vs OF, converged):
#     refreshed (correct)   U 3.4e-06   k 4.4e-06   omega 3.6e-06   nut 1.7e-06
#     frozen (the bug)      U 1.6e-03   k 2.0e-01   omega 4.0e-02   nut 1.5e-01
# i.e. k improves by a factor of ~44,000. A case with a fixedValue inlet would pass either way and prove
# nothing -- flowRateInletVelocity is the whole point of this gate.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/rhoTI" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_ti_vs_of}
TOL=${TOL:-1e-4}
# k must be far better than the frozen value (1.95e-01), so loosening TOL cannot hide a regression.
KMAX=${KMAX:-1e-3}

if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true

rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$SRC"/* "$WORK/"
mkdir -p "$WORK/0" && cp "$WORK"/0.orig/* "$WORK/0/"
cd "$WORK"
blockMesh > log.blockMesh 2>&1
rhoSimpleFoam > log.rhoSimpleFoam 2>&1
OFLAST=$(ls -d [0-9]* | grep -vx 0 | sort -g | tail -1)

rm -rf "$WORK.brae"; cp -r "$SRC" "$WORK.brae"
cp -r "$WORK/constant/polyMesh" "$WORK.brae/constant/"
mkdir -p "$WORK.brae/0" && cp "$WORK.brae"/0.orig/* "$WORK.brae/0/"
"$BUILD/brae_rhoSimpleFoam" -case "$WORK.brae" > "$WORK.brae/log.brae" 2>&1
BRLAST=$(ls -d "$WORK.brae"/[0-9]* | grep -v '/0$' | sort -g | tail -1)

python3 - "$WORK/$OFLAST" "$BRLAST" "$TOL" "$KMAX" <<'PY'
import re, sys, math
ofd, brd, tol, kmax = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])

def rd(path):
    try:
        t = open(path).read()
    except OSError:
        return None
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*boundaryField', t, re.S)
    if not m:
        return None
    return [float(x) for x in re.findall(r'-?\d+\.?\d*[eE]?[-+]?\d*', m.group(1))]

bad = 0
checked = 0
scores = {}
for f in ("U", "T", "k", "omega", "nut"):
    of, br = rd(f"{ofd}/{f}"), rd(f"{brd}/{f}")
    if of is None or br is None:
        print(f"  {f}: SKIP (not written on both sides)")
        continue
    n = min(len(of), len(br))
    denom = sum(a*a for a in of[:n])
    if denom <= 0.0:
        print(f"  {f}: FAIL OF field is all zeros")
        bad += 1
        continue
    l2 = math.sqrt(sum((a-b)**2 for a, b in zip(of[:n], br[:n])) / denom)
    spread = (max(of[:n]) - min(of[:n])) / (abs(sum(of[:n])/n) + 1e-300)
    ok = l2 <= tol and spread > 1e-6
    scores[f] = l2
    print(f"  {f}: L2rel {l2:.4e}  tol {tol:.0e}  spread {spread:.3e}  {'OK' if ok else 'FAIL'}")
    checked += 1
    if not ok:
        bad += 1

# Name the failure mode rather than leaving it to a tolerance: a frozen inlet shows up in k first.
kk = scores.get("k")
if kk is None:
    print("  FAIL k not compared -- that is the field this gate is FOR")
    bad += 1
elif kk > kmax:
    print(f"  FAIL k {kk:.4e} > {kmax:.0e}: the turbulent-inlet BCs are likely frozen at set-up again.")
    print( "       OF re-evaluates them every updateCoeffs; brae must call deviceUpdateTurbulentInletK /")
    print( "       deviceUpdateTurbulentInletSecond each outer iteration (device_simple_foam.cu).")
    bad += 1

if checked < 4:
    print(f"  FAIL only {checked} fields compared -- the gate is not testing what it claims")
    bad += 1

print(f"ti_vs_openfoam: {bad} failures over {checked} fields")
sys.exit(1 if bad else 0)
PY
