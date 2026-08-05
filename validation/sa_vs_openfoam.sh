#!/bin/bash
# brae vs OpenFOAM simpleFoam on airFoil2D: SpalartAllmaras, external aero, farfield + nutUSpaldingWallFunction.
#
# SA had NO ctest coverage at all before this gate. That is not a small hole: SA is what the delta-wing and
# F-16 demos run (SA-IDDES), and the gap it hid was large -- brae's nuTilda was 34% off OF on this very case.
#
# The bug the gate exists to catch: OF's DnuTildaEff() is a volScalarField, (nuTilda + nu)/sigmaNut
# (SpalartAllmarasBase.C:381-390), so the laplacian's WALL-FACE coefficient uses the PATCH value of nuTilda.
# At an SA wall nuTilda is fixedValue 0, so OF's wall diffusivity is exactly nu/sigmaNut. brae passed no
# per-face boundary diffusivity and fell back to the adjacent CELL's value, too large by (1 + nuTilda_P/nu)
# -- a median factor of ~133 here. That is a spurious implicit sink on every wall-adjacent cell worth ~23%
# of the whole destruction term, so it over-damped nuTilda exactly at the wall.
#
# Why nothing else caught it: the boundary laplacian weight is ZERO on a zeroGradient patch, so the k and
# omega wall treatments never exercise this path -- the bug is reachable only through a fixedValue scalar
# wall BC, which among the turbulence models is SA's nuTilda alone.
#
# Measured, converged, vs OF v2412:
#     before : U 1.36e-02   p 3.44e-02   nuTilda 3.42e-01
#     after  : U 4.88e-05   p 2.45e-04   nuTilda 3.63e-03
# Fields land at ~4e-3 on nuTilda, so 2e-2 keeps ~5x margin while still catching a regression of the bug
# above by two orders of magnitude.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_sa_vs_of}
TOL=${TOL:-2e-2}
# nuTilda must be at least this much better than the known-broken value (3.4e-1), so a regression that
# merely loosens the tolerance cannot pass.
NUTMAX=${NUTMAX:-2e-2}

if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true
TUT="$WM_PROJECT_DIR/tutorials/incompressible/simpleFoam/airFoil2D"
[ -d "$TUT" ] || TUT=/usr/lib/openfoam/openfoam2412/tutorials/incompressible/simpleFoam/airFoil2D
if [ ! -d "$TUT" ]; then echo "airFoil2D tutorial not found -- skipping"; exit 77; fi

rm -rf "$WORK"; mkdir -p "$WORK"
cp -r "$TUT"/* "$WORK/"
cd "$WORK"
cp -rf constant/polyMesh.orig constant/polyMesh    # the tutorial ships the mesh packed
rm -rf 0; cp -r 0.orig 0
simpleFoam > log.simpleFoam 2>&1
OFLAST=$(ls -d [0-9]* | grep -vx 0 | sort -g | tail -1)

# brae runs the same case in a clean copy, sharing only the mesh
rm -rf "$WORK.brae"; mkdir -p "$WORK.brae"
cp -r "$TUT"/* "$WORK.brae/"
cp -rf "$WORK/constant/polyMesh" "$WORK.brae/constant/"
rm -rf "$WORK.brae/0"; cp -r "$WORK.brae/0.orig" "$WORK.brae/0"
# airFoil2D asks for `div(phi,nuTilda) bounded Gauss linearUpwind grad(nuTilda)`. The STEADY driver gates
# turbulence-scalar linearUpwind off by default as a cold-start stability guard (see gpuSimpleFoam.cu), so
# without this the gate would compare brae-on-upwind against OF-on-linearUpwind and fail for the wrong
# reason. Measured here: running upwind against this case costs nuTilda 1.67 vs 3.6e-03 -- a factor of 460,
# which is by far the sharpest evidence yet that the downgrade is expensive on external aero.
BRAE_SCALAR_LINEARUPWIND=1 "$BUILD/brae" -case "$WORK.brae" > "$WORK.brae/log.brae" 2>&1
BRLAST=$(ls -d "$WORK.brae"/[0-9]* | grep -v '/0$' | sort -g | tail -1)

python3 - "$WORK/$OFLAST" "$BRLAST" "$TOL" "$NUTMAX" <<'PY'
import re, sys, math
ofd, brd, tol, nutmax = sys.argv[1], sys.argv[2], float(sys.argv[3]), float(sys.argv[4])

def rd(path):
    try:
        t = open(path).read()
    except OSError:
        return None
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*boundaryField', t, re.S)
    if not m:
        return None
    return [float(x) for x in re.findall(r'-?\d+\.?\d*[eE]?[-+]?\d*', m.group(1))]

FIELDS = ("U", "p", "nuTilda", "nut")
bad = 0
checked = 0
scores = {}
for f in FIELDS:
    of, br = rd(f"{ofd}/{f}"), rd(f"{brd}/{f}")
    if of is None or br is None:
        print(f"  {f}: SKIP (not written on both sides: OF={of is not None} brae={br is not None})")
        continue
    n = min(len(of), len(br))
    denom = sum(a*a for a in of[:n])
    if denom <= 0.0:
        print(f"  {f}: FAIL OF field is all zeros -- fix the case, not the tolerance")
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

if checked < 3:
    print(f"  FAIL only {checked} fields compared -- the gate is not testing what it claims")
    bad += 1

# The wall-diffusivity bug shows up as an over-damped nuTilda specifically. Assert it directly, so the
# gate still fails if someone reintroduces it and widens TOL to accommodate.
nt = scores.get("nuTilda")
if nt is None:
    print("  FAIL nuTilda not compared -- that is the field this gate is FOR")
    bad += 1
elif nt > nutmax:
    print(f"  FAIL nuTilda {nt:.4e} > {nutmax:.0e}: the wall-face diffusivity is likely back to the")
    print(f"       adjacent-cell value (that produced 3.4e-01 here). See SpalartAllmarasBase.C:381-390.")
    bad += 1

print(f"sa_vs_openfoam: {bad} failures over {checked} fields")
sys.exit(1 if bad else 0)
PY
