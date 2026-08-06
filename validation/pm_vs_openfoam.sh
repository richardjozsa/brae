#!/bin/bash
# pressureControl's pMaxFactor/pMinFactor reference must come ONLY from pressure patches that FIX a value.
#
# OF: pressureControl.C:75-77 -- `if (pbf[patchi].fixesValue())` -- because the factors scale a KNOWN
# reference pressure. A zeroGradient or calculated patch carries whatever the field currently holds there,
# which is not a reference at all; it is just the interior value showing through.
#
# brae scanned EVERY patch. Identical on a uniform initial p (every patch reads the same number), which is
# why all the gates agreed -- and wrong the moment p is non-uniform. Measured on this case, whose initial p
# ramps 1.0e5 -> 9.0e5 while the only fixedValue patch (the outlet) sits at 1.0e5:
#     fixesValue only (OF)  ->  [50000, 150000]
#     every patch (the bug) ->  [50000, 1.34962e+06]      pMax 9x too large; the limiter does nothing
#
# Asserted on the solver's own printed limits rather than on a field, because that is the quantity that
# went wrong -- a field comparison would only show it once the limiter actually had to clip something.
set -e
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
SRC=${SRC:-$(cd "$(dirname "$0")/rhoPM" && pwd)}
WORK=${WORK:-/tmp/brae_pm_limits}
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found -- skipping (blockMesh needed)"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true

rm -rf "$WORK"; cp -r "$SRC" "$WORK"
mkdir -p "$WORK/0" && cp "$WORK"/0.orig/* "$WORK/0/"
cd "$WORK" && blockMesh > log.blockMesh 2>&1

# ramp the initial p so the zeroGradient walls sit far from the fixedValue outlet
python3 - "$WORK" <<'PY'
import pathlib, re, sys
w = sys.argv[1]
own = pathlib.Path(w + "/constant/polyMesh/owner").read_text()
body = re.search(r'\n(\d+)\s*\(\s*(.*?)\n\)', own, re.S).group(2)
nc = max(int(x) for x in re.findall(r'^\d+$', body, re.M)) + 1
p = pathlib.Path(w + "/0/p"); s = p.read_text()
vals = "\n".join(f"{100000.0 + 800000.0*i/nc:.6g}" for i in range(nc))
s = re.sub(r'internalField[^;]*;', f'internalField   nonuniform List<scalar>\n{nc}\n(\n{vals}\n)\n;', s, count=1)
p.write_text(s)
print(f"nCells={nc}")
PY

OUT=$("$BUILD/brae_rhoSimpleFoam" -case "$WORK" 2>&1 | grep -i "pressureControl limits" | head -1)
echo "  $OUT"
python3 - "$OUT" <<'PY'
import re, sys
m = re.search(r'\[([-\d.eE+]+),\s*([-\d.eE+]+)\]', sys.argv[1])
if not m:
    print("  FAIL solver printed no pressureControl limits"); sys.exit(1)
lo, hi = float(m.group(1)), float(m.group(2))
bad = 0
# the only fixedValue p patch is the outlet at 1.0e5, so the limits must be 0.5x and 1.5x of THAT
if abs(lo - 50000.0) > 1.0:
    print(f"  FAIL pMin {lo:.6g} != 50000 (0.5 x the fixedValue outlet)"); bad += 1
if abs(hi - 150000.0) > 1.0:
    print(f"  FAIL pMax {hi:.6g} != 150000 (1.5 x the fixedValue outlet).")
    print( "       A value near 1.35e+06 means the reference scan is including zeroGradient patches,")
    print( "       which read the interior field rather than a reference. OF scans only patches where")
    print( "       fixesValue() is true (pressureControl.C:75-77).")
    bad += 1
print(f"pm_vs_openfoam: {bad} failures")
sys.exit(1 if bad else 0)
PY
