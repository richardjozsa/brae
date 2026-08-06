#!/bin/bash
# brae_rhoSimpleFoam vs OpenFOAM rhoSimpleFoam with `inletOutlet` on TEMPERATURE.
#
# This gate exists because none of the other nine compressible gates uses inletOutlet on T -- they all
# specify `outlet { type zeroGradient; }` -- and that blindness hid a real bug for the whole port:
#
#   dbHe_ (the ENERGY boundary) is built from T's boundary, so it inherited T's inletOutlet mask. But
#   deviceUpdateInletOutlet was called on dbU_, dbP_, dbK_, dbEps_, dbReThetat_ and dbGammaInt_ -- and
#   never on dbHe_. The mask was set once at build time and never consulted, so the outlet enthalpy stayed
#   clamped at fixedValue(inletValue) with full convection AND laplacian coupling, where OF switches to
#   zeroGradient on outflow. It then propagated: rhoBnd_ is built from dbHe_, so the outlet density, the
#   outlet mass flux and hence the pressure field all inherited the pinned temperature.
#
# All six stock rhoSimpleFoam tutorials use inletOutlet on T, so this was not an exotic path.
#
# The case is the heated duct with ONE change: the outlet T becomes
#     inletOutlet; inletValue uniform 300;
#
# The discriminator is the INTERNAL field, not the outlet face value. Measured by reverting the dbHe_
# update on this exact case:
#     with the fix     T 2.7e-07   rho 1.5e-06   U 2.6e-06   p 5.3e-07
#     without it       T 2.8e+00   rho 3.6e-02   U 1.5e-02   p 1.7e-03
# i.e. a factor of ten million on T. The pinned boundary does not stay on the boundary: it is coupled in
# through both convection and the laplacian, and rhoBnd_ is derived from it, so the whole field moves.
#
# (The outlet face values themselves only differ by ~5.7 K here, and brae's WRITTEN boundary is a
# pass-through of the input entry regardless -- see the note further down. Neither is the witness.)
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/rhoIO" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_io_vs_of}
TOL=${TOL:-2e-5}

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

python3 - "$WORK/$OFLAST" "$BRLAST" "$TOL" <<'PY'
import re, sys, math
ofd, brd, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])

def internal(path):
    try:
        t = open(path).read()
    except OSError:
        return None
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*boundaryField', t, re.S)
    if not m:
        return None
    return [float(x) for x in re.findall(r'-?\d+\.?\d*[eE]?[-+]?\d*', m.group(1))]

def patchValues(path, patch):
    """The boundary values actually written on a patch -- this gate's real subject."""
    try:
        t = open(path).read()
    except OSError:
        return None
    b = re.search(r'boundaryField(.*)$', t, re.S)
    if not b:
        return None
    pm = re.search(r'\n\s*' + patch + r'\s*\n?\s*\{(.*?)\n\s*\}', b.group(1), re.S)
    if not pm:
        return None
    blk = pm.group(1)
    vm = re.search(r'value\s+nonuniform[^(]*\((.*?)\)\s*;', blk, re.S)
    if vm:
        return [float(x) for x in re.findall(r'-?\d+\.?\d*[eE]?[-+]?\d*', vm.group(1))]
    um = re.search(r'value\s+uniform\s+([-\d.eE+]+)', blk)
    return [float(um.group(1))] if um else None

bad = 0
checked = 0
for f in ("T", "p", "U", "rho"):
    of, br = internal(f"{ofd}/{f}"), internal(f"{brd}/{f}")
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
    print(f"  {f}: L2rel {l2:.4e}  tol {tol:.0e}  spread {spread:.3e}  {'OK' if ok else 'FAIL'}")
    checked += 1
    if not ok:
        bad += 1

# The direct assertion. The internal field alone is a weak witness -- the pinned value lives ON the patch,
# and only leaks inward through the coupling. Compare the outlet face values themselves, and check they are
# nowhere near inletValue, so the failure mode is named rather than inferred from a tolerance.
ofT = patchValues(f"{ofd}/T", "outlet")
brT = patchValues(f"{brd}/T", "outlet")
if ofT is None or brT is None:
    print(f"  FAIL cannot read the outlet T values (OF={ofT is not None} brae={brT is not None})")
    bad += 1
else:
    ofMean = sum(ofT)/len(ofT)
    brMean = sum(brT)/len(brT)
    # Now an ASSERTION. This was informational while brae echoed the INPUT boundaryField instead of the
    # values it computed -- the written outlet T read 300 K (the input `value`) whether the solve was right
    # or not, so asserting on it would have failed for the wrong reason. The writer now emits the SOLVED
    # boundary, so the written outlet T must track OpenFOAM's.
    print(f"  outlet T: OF mean {ofMean:.2f} K   brae mean {brMean:.2f} K   inletValue 300 K")
    if abs(brMean - ofMean) > 0.5:
        print(f"  FAIL written outlet T differs from OpenFOAM by {abs(brMean-ofMean):.2f} K")
        bad += 1
    if abs(brMean - 300.0) < 1e-9:
        print( "  FAIL the written outlet T is exactly inletValue -- either the inletOutlet mask on the")
        print( "       ENERGY boundary is unresolved again, or the writer is back to echoing the input")
        print( "       boundaryField instead of the computed one (foam_field_writer.cuh).")
        bad += 1

if checked < 3:
    print(f"  FAIL only {checked} fields compared -- the gate is not testing what it claims")
    bad += 1

print(f"io_vs_openfoam: {bad} failures over {checked} fields")
sys.exit(1 if bad else 0)
PY
