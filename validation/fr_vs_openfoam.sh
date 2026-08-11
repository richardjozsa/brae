#!/bin/bash
# Gate 3: brae_rhoSimpleFoam vs OpenFOAM rhoSimpleFoam on the same heated-duct case.
#
# The case matters as much as the comparison. An earlier version of this gate used a duct with a fixed
# inlet temperature and zeroGradient everywhere else, whose exact steady answer is a UNIFORM field --
# it passed at 1e-11 while testing none of the compressible coupling. This case runs a 700 K wall
# against a 300 K wall so there is ~72 K of real spread, noSlip boundary layers, and density varying
# across the duct: rho*rAU in the pressure equation, the phiHbyA rho-weighting (including the boundary
# rho evaluated from boundary p and T), muEff and the rho relaxation are all exercised.
#
# OF stops on residualControl (~104 iters); brae runs its endTime. Both reach steady state, which is
# what is compared -- not the iteration count.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/rhoFR" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_fr_vs_of}
# 1e-5: both fields land at ~2e-7. The old 1e-2 was three orders too loose and let the kinetic-energy
# sign bug through unnoticed for two phases.
TOL=${TOL:-1e-5}

if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true

rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$SRC"/* "$WORK/"
mkdir -p "$WORK/0" && cp "$WORK"/0.orig/* "$WORK/0/"
cd "$WORK"
blockMesh > log.blockMesh 2>&1
rhoSimpleFoam > log.rhoSimpleFoam 2>&1
OFLAST=$(ls -d [0-9]* | grep -vx 0 | sort -g | tail -1)

# brae runs the same case in a clean copy, sharing only the mesh OF generated
rm -rf "$WORK.brae"; cp -r "$SRC" "$WORK.brae"
cp -r "$WORK/constant/polyMesh" "$WORK.brae/constant/"
mkdir -p "$WORK.brae/0" && cp "$WORK.brae"/0.orig/* "$WORK.brae/0/"
"$BUILD/brae_rhoSimpleFoam" -case "$WORK.brae" > "$WORK.brae/log.brae" 2>&1
BRLAST=$(ls -d "$WORK.brae"/[0-9]* | grep -v '/0$' | sort -g | tail -1)

python3 - "$WORK/$OFLAST" "$BRLAST" "$TOL" <<'PY'
import re, sys, math
ofd, brd, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])
def rd(p):
    t = open(p).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\(([^)]*)\)', t, re.S)
    return [float(x) for x in m.group(1).split()] if m else None
bad = 0
for f in ("T", "p"):
    of, br = rd(f"{ofd}/{f}"), rd(f"{brd}/{f}")
    if of is None or br is None:
        print(f"  {f}: FAIL uniform/missing field -- the case is degenerate, fix the case not the tolerance")
        bad += 1
        continue
    n = min(len(of), len(br))
    l2 = math.sqrt(sum((a-b)**2 for a, b in zip(of[:n], br[:n])) / sum(a*a for a in of[:n]))
    spread = (max(of[:n]) - min(of[:n])) / (abs(sum(of[:n])/n) + 1e-300)
    ok = l2 <= tol and spread > 1e-6            # spread guard: refuse to pass on a flat field
    print(f"  {f}: L2rel {l2:.4e}  tol {tol:.0e}  spread {spread:.3e}  {'OK' if ok else 'FAIL'}")
    if not ok: bad += 1
print(f"rho_vs_openfoam: {bad} failures")
sys.exit(1 if bad else 0)
PY
