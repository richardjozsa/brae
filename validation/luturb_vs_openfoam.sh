#!/bin/bash
# brae_rhoSimpleFoam vs OpenFOAM rhoSimpleFoam with linearUpwind on the TURBULENCE scalars.
#
# Every other gate specifies `div(phi,k|omega) bounded Gauss upwind`, so none of them can see what scheme
# the turbulence scalars actually run. That blindness hid a real error: both steady drivers used to force
# ctl.luK = ctl.luEps = false, silently running UPWIND on any case that asked for linearUpwind, on the
# strength of an unmeasured claim that linearUpwind "degrades turbulence accuracy vs OF".
#
# Re-measured on this case (identical fvSchemes on both sides, converged fields):
#     honoured (now the default) : k 2.0e-06   omega 3.2e-06   nut 8.1e-07
#     downgraded to upwind       : k 6.8e-03   omega 8.9e-03   nut 1.6e-02
# The downgrade was the error, costing 1.6% on nut. This gate exists so that cannot come back silently.
#
# It runs BOTH directions on purpose. Checking only that the honoured run matches OF would still pass if
# the downgrade were reinstated AND the tolerance loosened; asserting that the downgraded run is clearly
# WORSE proves the case actually discriminates between the two schemes. A case where both agree with OF
# is a case that is not testing the scheme at all -- which is exactly how this bug survived.
#
# NOTE the incompressible driver still gates this OFF as a cold-start stability guard (pitzDaily SST
# diverges from a uniform initial state with linearUpwind on k, where OF converges). That is a separate,
# open, convergence-path problem -- the discretisation is validated to 1.6e-06 by a single-iteration
# reproducer from OF's own converged state. See gpuSimpleFoam.cu.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/rhoLUturb" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_luturb_vs_of}
# Fields land at 2-3.2e-06 honoured; 2e-5 keeps ~6x margin and is still 300x below the 6.8e-03 the
# downgrade produces, so the discrimination check below has enormous headroom.
TOL=${TOL:-2e-5}
# The downgraded run must be at least this much worse on k, or the case is not scheme-sensitive.
MINGAP=${MINGAP:-50}

if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true

rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$SRC"/* "$WORK/"
mkdir -p "$WORK/0" && cp "$WORK"/0.orig/* "$WORK/0/"
cd "$WORK"
blockMesh > log.blockMesh 2>&1
rhoSimpleFoam > log.rhoSimpleFoam 2>&1
OFLAST=$(ls -d [0-9]* | grep -vx 0 | sort -g | tail -1)

# brae twice on the same mesh: default (honours the scheme) and forced-upwind.
for MODE in honour downgrade; do
    B="$WORK.$MODE"
    rm -rf "$B"; cp -r "$SRC" "$B"
    cp -r "$WORK/constant/polyMesh" "$B/constant/"
    mkdir -p "$B/0" && cp "$B"/0.orig/* "$B/0/"
    if [ "$MODE" = downgrade ]; then
        BRAE_SCALAR_LINEARUPWIND=0 "$BUILD/brae_rhoSimpleFoam" -case "$B" > "$B/log.brae" 2>&1
    else
        "$BUILD/brae_rhoSimpleFoam" -case "$B" > "$B/log.brae" 2>&1
    fi
done

python3 - "$WORK/$OFLAST" "$WORK.honour" "$WORK.downgrade" "$TOL" "$MINGAP" <<'PY'
import re, sys, math, os, glob
ofd, hon, dwn, tol, mingap = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), float(sys.argv[5])

def rd(path):
    try:
        t = open(path).read()
    except OSError:
        return None
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*)\)\s*;', t, re.S)
    if not m:
        return None
    return [float(x) for x in re.findall(r'-?\d+\.?\d*[eE]?[-+]?\d*', m.group(1).replace('(', ' ').replace(')', ' '))]

def latest(d):
    ts = [x for x in glob.glob(d + "/[0-9]*")
          if re.fullmatch(r'[\d.]+', os.path.basename(x)) and os.path.basename(x) != "0"]
    if not ts:
        return None
    return sorted(ts, key=lambda x: float(os.path.basename(x)))[-1]

def l2(a, b):
    n = min(len(a), len(b))
    den = sum(x*x for x in a[:n])
    if den <= 0.0:
        return None
    return math.sqrt(sum((x-y)**2 for x, y in zip(a[:n], b[:n])) / den)

FIELDS = ("T", "p", "U", "k", "omega", "nut")
bad = 0
checked = 0
scores = {}
for tag, d in (("honour", hon), ("downgrade", dwn)):
    ld = latest(d)
    if ld is None:
        print(f"  FAIL {tag}: brae wrote no time directory")
        bad += 1
        continue
    print(f"  --- brae {tag} (t={os.path.basename(ld)}) vs OF ---")
    for f in FIELDS:
        of, br = rd(f"{ofd}/{f}"), rd(f"{ld}/{f}")
        if of is None or br is None:
            print(f"    {f}: SKIP (not written on both sides)")
            continue
        e = l2(of, br)
        if e is None:
            print(f"    {f}: FAIL OF field is all zeros -- fix the case, not the tolerance")
            bad += 1
            continue
        scores[(tag, f)] = e
        if tag == "honour":
            ok = e <= tol
            checked += 1
            if not ok:
                bad += 1
            print(f"    {f}: L2rel {e:.4e}  tol {tol:.0e}  {'OK' if ok else 'FAIL'}")
        else:
            print(f"    {f}: L2rel {e:.4e}  (expected WORSE -- this run ignores the requested scheme)")

if checked < 5:
    print(f"  FAIL only {checked} fields compared -- the gate is not testing what it claims")
    bad += 1

# Discrimination: the downgraded run must be markedly worse, else the case cannot see the scheme at all.
for f in ("k", "nut"):
    h, d = scores.get(("honour", f)), scores.get(("downgrade", f))
    if h is None or d is None:
        print(f"  FAIL cannot compare {f} across both modes")
        bad += 1
        continue
    gap = d / max(h, 1e-300)
    ok = gap >= mingap
    print(f"  discrimination {f}: downgraded/honoured = {gap:.1f}x  (need >= {mingap:.0f}x)  {'OK' if ok else 'FAIL'}")
    if not ok:
        bad += 1
        print(f"       -> the two schemes give nearly the same answer here, so this case does NOT test")
        print(f"          the turbulence convection scheme. Fix the case, not the tolerance.")

print(f"luturb_vs_openfoam: {bad} failures over {checked} fields")
sys.exit(1 if bad else 0)
PY
