#!/bin/bash
# E5b: OF's `RAS { turbulence off; }` switch, against OpenFOAM.
#
# Found by dict_audit only after E5 made the audit run on REFUSED cases. aerofoilNACA0012 sets this entry
# and aerofoilNACA0012 refuses on fvOptions, so the switch sat in the one file brae never audited. brae
# read `simulationType` and `RASModel` out of that dict and never looked at `turbulence` beside them, so a
# case asking to FREEZE the turbulence ran a fully live model.
#
# OF semantics (RASModel.C:70 getOrDefault<Switch>("turbulence", true); kEpsilon.C:216, kOmegaSSTBase.C:502
# `if (!turbulence_) return;`): correct() returns before anything is solved, so k, epsilon|omega and nut
# keep the values they were constructed with while momentum keeps using that frozen nut. It is NOT the same
# as `simulationType laminar`, where nut is zero and never read.
#
# What this asserts, and why in this order:
#   1. turbulence off -> k and omega are EXACTLY frozen (bit-identical to 0/). This is the claim, and it is
#      exact, not approximate: measured L2rel 0.000e+00 against OF on both fields.
#   2. NEGATIVE CONTROL: turbulence on -> the same fields MOVE. Without it, a brae that never solved
#      turbulence at all would pass (1) perfectly.
#   3. U still agrees with OF, i.e. momentum really is using the frozen nut rather than nothing.
#
# This gate also pins E7, which it is what FOUND. With the model frozen the STARTUP correctNut becomes the
# answer, so nut stopped being self-correcting and a startup discrepancy was visible for the first time:
# 8.41e-02 vs OF, all of it in the 40 inlet-column cells.
#
# Root cause, predicted arithmetically to six decimals BEFORE any code changed: OF constructs the thermo,
# THEN the turbulence model, THEN calls validate() -> correctNut(). brae validated inside the solver ctor,
# before the compressible driver had seeded the thermo -- so a flowRateInletVelocity patch was still on its
# `rhoInlet` fallback (1.0 here against a true 1.161), giving an inlet U of 58.85 m/s instead of 50.69
# against a uniform internal 50. That du/dx = 354 makes sqrt(S2) = 500.6 > a1*omega = 124, tripping the SST
# Bradshaw limiter, so nut/(k/omega) = 124/500.6 = 0.247687 -- against 0.247687 measured. OF, whose inlet
# still holds the file `value` at that moment, has no gradient there and so no limiting.
#
# Fixed by revalidateAfterThermo(): nut 8.41e-02 -> 2.41e-09 (3.5e7x), U 3.63e-05 -> 2.04e-05.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/rhoTI" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_turbswitch_vs_of}
NUT_TOL=${NUT_TOL:-1e-6}   # lands at 2.41e-09; 1e-6 keeps ~400x margin and still fails the 8.41e-02 regression

if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true

rm -rf "$WORK"; mkdir -p "$WORK"
MESH="$WORK/mesh"
cp -r "$SRC" "$MESH"
mkdir -p "$MESH/0" && cp "$MESH"/0.orig/* "$MESH/0/"
( cd "$MESH" && blockMesh > log.blockMesh 2>&1 )

setup()   # $1 = dir, $2 = on|off
{
    rm -rf "$1"; cp -r "$SRC" "$1"
    cp -r "$MESH/constant/polyMesh" "$1/constant/"
    mkdir -p "$1/0" && cp "$1"/0.orig/* "$1/0/"
    sed -i "s/turbulence  *[a-zA-Z]*;/turbulence $2;/" "$1/constant/turbulenceProperties"
    grep -q "turbulence  *$2;" "$1/constant/turbulenceProperties" \
        || { echo "  FAIL could not set 'turbulence $2' in the case -- the sed did not match"; exit 1; }
}

for SW in on off; do
    setup "$WORK/br_$SW" "$SW"
    "$BUILD/brae_rhoSimpleFoam" -case "$WORK/br_$SW" > "$WORK/br_$SW/log.brae" 2>&1
    setup "$WORK/of_$SW" "$SW"
    ( cd "$WORK/of_$SW" && rhoSimpleFoam > log.rhoSimpleFoam 2>&1 )
done

python3 - "$WORK" "$NUT_TOL" <<'PY'
import re, os, sys, math
work, nut_tol = sys.argv[1], float(sys.argv[2])

def rd(path):
    try:
        t = open(path).read()
    except OSError:
        return None
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*boundaryField', t, re.S)
    if m:
        return [float(x) for x in m.group(1).replace('(', ' ').replace(')', ' ').split()]
    m = re.search(r'internalField\s+uniform\s+\(?([-\d.eE+\s]+?)\)?\s*;', t)
    return [float(x) for x in m.group(1).split()] if m else None

def last(d):
    ts = [x for x in os.listdir(d) if x.replace('.', '', 1).isdigit() and x != '0']
    return os.path.join(d, sorted(ts, key=float)[-1]) if ts else None

def l2(a, b):
    n = min(len(a), len(b))
    den = sum(x*x for x in a[:n])
    if den <= 0:
        return None
    return math.sqrt(sum((x-y)**2 for x, y in zip(a[:n], b[:n])) / den)

bad = 0
for sw in ("off", "on"):
    br, of = last(f"{work}/br_{sw}"), last(f"{work}/of_{sw}")
    if not br or not of:
        print(f"  turbulence {sw}: FAIL no time directory written (brae={br} OF={of})")
        bad += 1
        continue
    print(f"  turbulence {sw}  (brae t={os.path.basename(br)}, OF t={os.path.basename(of)})")
    for f in ("k", "omega", "nut", "U"):
        a, b = rd(f"{of}/{f}"), rd(f"{br}/{f}")
        if a is None or b is None:
            print(f"    {f}: FAIL missing on one side")
            bad += 1
            continue
        e = l2(a, b)
        # k and omega must be EXACT with the model frozen. nut is the startup correctNut and is what
        # catches E7 regressing (8.41e-02 if validate() runs before the thermo again).
        tol = 1e-5 if sw == "on" else (1e-12 if f in ("k", "omega") else (nut_tol if f == "nut" else 1e-3))
        ok = e is not None and e <= tol
        print(f"    {f:6s}: L2rel {e:.3e}  tol {tol:.0e}  {'OK' if ok else 'FAIL'}")
        if not ok:
            bad += 1

    # NEGATIVE CONTROL: with the switch off the fields must be UNCHANGED from 0/, and with it on they must
    # have MOVED. Asserting only "brae matches OF" would pass for a brae that never solves turbulence.
    for f in ("k", "omega"):
        init, final = rd(f"{work}/br_{sw}/0/{f}"), rd(f"{br}/{f}")
        if not init or not final:
            continue
        moved = max(abs(x - init[0]) for x in final[:min(len(final), 500)]) > 1e-9
        want = (sw == "on")
        print(f"    {f} moved from its 0/ value: {moved}  (expected {want})  "
              f"{'OK' if moved == want else 'FAIL'}")
        if moved != want:
            print(f"       with the switch {sw}, OF's RASModel::correct() "
                  f"{'solves' if want else 'returns immediately'}, so brae must too.")
            bad += 1

print(f"turbswitch_vs_openfoam: {bad} failures")
sys.exit(1 if bad else 0)
PY
