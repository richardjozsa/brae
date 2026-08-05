#!/bin/bash
# brae_rhoSimpleFoam vs OpenFOAM with a REGEX-KEYED boundary entry.
#
# OpenFOAM resolves a boundaryField key against a patch by exact name, then literal group membership, then
# REGEX against the patch name or a group name (last pattern wins). Cases use this constantly. brae's
# buildField always did it correctly, but several other lookups compared `entry.name == patch.name` and
# silently fell back to a default when the key was a pattern:
#
#   - the per-face Prt for alphatWallFunction reverted to the MODEL default 1.0 instead of the wall
#     function's 0.85, so wall alphat and the wall heat flux ran ~15% low. squareBend and squareBendLiq
#     both key that entry as "(?i).*walls" against a patch literally named `walls`.
#   - the turbulent-inlet BCs kept their written `value` placeholder instead of the computed inlet value.
#
# This case keys the alphat wall entry as "(?i).*wall" against patches named hotWall/coldWall.
#
# Measured on this case (brae vs OF, converged):
#     regex-aware (correct)   T 2.7e-07   k 1.9e-06   nut 1.2e-06
#     exact-name (the bug)    T 5.3e-03   k 2.7e-03   nut 1.3e-03
# A case whose keys are all literal patch names would pass either way and prove nothing.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/rhoRX" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_rx_vs_of}
TOL=${TOL:-1e-4}
# k must be far better than the frozen value (1.95e-01), so loosening TOL cannot hide a regression.
TMAX=${TMAX:-1e-4}

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
tt = scores.get("T")
if tt is None:
    print("  FAIL T not compared -- that is the field this gate is FOR")
    bad += 1
elif tt > tmax:
    print(f"  FAIL T {tt:.4e} > {tmax:.0e}: the regex-keyed alphatWallFunction entry is probably being")
    print( "       missed again, so Prt fell back to the model default 1.0 instead of the patch's 0.85.")
    print( "       Patch entries must resolve via findPatchEntry (exact -> group -> regex), never by")
    print( "       comparing entry.name == patch.name. See patch_entry_lookup.cuh.")
    bad += 1

if checked < 4:
    print(f"  FAIL only {checked} fields compared -- the gate is not testing what it claims")
    bad += 1

print(f"rx_vs_openfoam: {bad} failures over {checked} fields")
sys.exit(1 if bad else 0)
PY
