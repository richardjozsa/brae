#!/bin/bash
# brae_rhoSimpleFoam vs OpenFOAM with `linearUpwind` on the KINETIC energy term, sensibleInternalEnergy.
#
# OF's EEqn carries div(phi,K|Ekp) as fvc::div -- fully EXPLICIT -- so the scheme chosen for it sets the
# face value directly. Two defects lived here, and the second was only exposed by fixing the first:
#
#   1. deviceEnergyKineticSource had no linearUpwind parameter at all. ctl_.luHe was parsed and never
#      forwarded, so `div(phi,Ekp) bounded Gauss linearUpwind` silently ran upwind. Cost: T 2.0e-02.
#   2. grad(K) took its boundary values from deviceBCValue(dbHe, K, ...) -- the ENERGY field's BC
#      descriptor applied to the K array, which at a fixedValue he patch returns he's refValue rather
#      than K's boundary value. Harmless-looking under limitedLinear, whose gradient only feeds a limiter
#      clamped to [0,1]; fatal under linearUpwind, where it goes straight into the face value. Enabling
#      linearUpwind on the wrong gradient made T FOUR TIMES WORSE (9.3e-02) than plain upwind.
#
# K's boundary is now built from the boundary U (and, for Ekp, the boundary p and rho) -- the same
# expression as the cell value, which is what OF's constructed volScalarField("Ekp", ...) carries.
#
# Measured on this case (brae vs OF, converged):
#     upwind, as shipped before   T 2.0e-02   U 8.6e-04
#     linearUpwind, bad gradient  T 9.3e-02   U 1.3e-02
#     linearUpwind, fixed         T 1.9e-06   U 3.8e-06
# The case is sensibleInternalEnergy on purpose: Ekp = 0.5|U|^2 + p/rho, and p/rho is a large share of e,
# so the kinetic term is big enough for its scheme to matter.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/rhoKE2" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_ke2_vs_of}
TOL=${TOL:-1e-4}
# k must be far better than the frozen value (1.95e-01), so loosening TOL cannot hide a regression.
TMAX=${TMAX:-1e-3}

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

python3 - "$WORK/$OFLAST" "$BRLAST" "$TOL" "$TMAX" <<'PY'
import re, sys, math
ofd, brd, tol, tmax = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])

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
for f in ("T", "p", "U"):
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
tt = scores.get("T")
if tt is None:
    print("  FAIL T not compared -- that is the field this gate is FOR")
    bad += 1
elif tt > tmax:
    print(f"  FAIL T {tt:.4e} > {tmax:.0e}. Two regressions look like this:")
    print( "       (a) div(phi,K|Ekp) ignoring linearUpwind and running upwind    -> ~2.0e-02")
    print( "       (b) grad(K) taking its boundary from the ENERGY field's BC     -> ~9.3e-02")
    print( "       See deviceEnergyKineticSource in device_energy.cu.")
    bad += 1

if checked < 3:
    print(f"  FAIL only {checked} fields compared -- the gate is not testing what it claims")
    bad += 1

print(f"ke2_vs_openfoam: {bad} failures over {checked} fields")
sys.exit(1 if bad else 0)
PY
