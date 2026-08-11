#!/bin/bash
# Gate: endTime is an ABSOLUTE time (OF semantics), and a restarted run reproduces a continuous one.
#
# WHY THIS GATE EXISTS. OF's Time::run() tests `value() < endTime - 0.5*deltaT`, so endTime is where the
# run STOPS, not how long it lasts:
#     case at latestTime 10, endTime 20   ->   OF runs 10 -> 20   (ten steps)
# brae's steady drivers looped `for (iter = 1; iter <= endTime; ++iter)`, i.e. endTime as a run LENGTH:
#     the same case                       ->   brae ran 10 -> 30  (twenty steps)
# That is only correct when startTime is 0, which every fresh-start case is -- so the whole suite was
# blind to it while it silently changed iteration counts, write times, and any comparison of a restarted
# run against a continuous one. Both steady drivers had it (gpuSimpleFoam and gpuRhoSimpleFoam);
# gpuPimpleFoam was already absolute.
#
# A SECOND BUG IN THE SAME AREA, fixed with it: both steady drivers took the time-value origin from
# controlDict's `startTime` rather than the RESOLVED start, so a `startFrom latestTime` restart from 10
# named its output 1, 2, 3... -- overwriting the case's own early history instead of continuing it.
#
# WHAT IT ASSERTS
#   1. OF's own behaviour, as the oracle: OF restarted at 10 with endTime 20 stops at 20.
#   2. brae matches it, for BOTH the continuous and the restarted run.
#   3. The round-trip identity: continuous@20 == restart(0->10, 10->20)@20, field by field.
#
# WHY rhoBox AND NOT angledDuct. (3) needs a QUIET state. Measured on angledDuct's cold start, where the
# pressureControl ceiling is saturated for dozens of iterations (OF's own `p max` starts at 1.47598e+06
# and decays), continuous-vs-restart came out at 1.08e-01 and deleting phi changed it by 1.0x -- the
# trajectory there is chaotic, so the test measures the transient rather than the restart. rhoBox is a
# smooth heated duct that never engages the limiter; it converges at ~105 iterations, so 20 is well
# inside its runway.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/rhoBox" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_roundtrip_vs_of}
T0=${T0:-10}    # restart point
T1=${T1:-20}    # stop time, absolute
# Measured: p 5.46e-09  U 8.70e-09  T 3.69e-06  rho 2.90e-06  phi 1.92e-05. These sit at brae's
# compressible run-to-run floor (the path is not bit-reproducible), not at a structural difference, so
# the bar is set an order above the worst of them rather than at zero.
TOL=${TOL:-1e-3}

if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true

# rhoBox's controlDict packs several entries per line, so line-anchored sed silently does nothing here.
# (That cost one confusing run: the case ignored the edit and went to convergence at 105.) Rewrite by key.
ctl() {   # $1 = controlDict, $2 = startTime, $3 = endTime, $4 = writeInterval
    python3 - "$@" <<'PY'
import re, sys
p, st, en, wi = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
s = open(p).read()
s = re.sub(r'startFrom\s+\w+;',             'startFrom startTime;', s)
s = re.sub(r'startTime\s+[0-9.eE+-]+;',     f'startTime {st};',      s)
s = re.sub(r'endTime\s+[0-9.eE+-]+;',       f'endTime {en};',        s)
s = re.sub(r'writeControl\s+\w+;',          'writeControl timeStep;', s)
s = re.sub(r'writeInterval\s+[0-9.eE+-]+;', f'writeInterval {wi};',   s)
open(p, 'w').write(s)
PY
}
last() { ls -d "$1"/[0-9]* 2>/dev/null | sed 's:.*/::' | sort -g | tail -1; }

rm -rf "$WORK" "$WORK.br" "$WORK.rst" "$WORK.of"
cp -r "$SRC" "$WORK"; mkdir -p "$WORK/0"; cp "$WORK"/0.orig/* "$WORK/0/"
( cd "$WORK" && blockMesh > log.blockMesh 2>&1 )
sed -i 's/relTol *[0-9.eE-]*;/relTol 0;/g' "$WORK/system/fvSolution"

bad=0

# ---------------------------------------------------------------- 1. the oracle: what does OF do?
cp -r "$WORK" "$WORK.of"
ctl "$WORK.of/system/controlDict" 0 "$T0" "$T0"
( cd "$WORK.of" && rhoSimpleFoam > log.of1 2>&1 || true )
ctl "$WORK.of/system/controlDict" "$T0" "$T1" "$T0"
( cd "$WORK.of" && rhoSimpleFoam > log.of2 2>&1 || true )
OFLAST=$(last "$WORK.of")
if [ "$OFLAST" != "$T1" ]; then
    echo "  OF restarted at $T0 with endTime $T1 stopped at $OFLAST, expected $T1 -- the oracle is not what this gate assumes"
    exit 1
fi
echo "  OF     restart $T0 -> endTime $T1 stops at $OFLAST  OK (endTime is absolute)"

# ---------------------------------------------------------------- 2. brae, continuous and restarted
cp -r "$WORK" "$WORK.br"
ctl "$WORK.br/system/controlDict" 0 "$T1" "$T0"
"$BUILD/brae_rhoSimpleFoam" -case "$WORK.br" > "$WORK.br/log" 2>&1 || true
BRLAST=$(last "$WORK.br")
[ "$BRLAST" = "$T1" ] || { echo "  brae   continuous 0 -> $T1 stopped at $BRLAST, expected $T1  FAIL"; bad=$((bad+1)); }
[ "$BRLAST" = "$T1" ] && echo "  brae   continuous 0 -> $T1 stops at $BRLAST  OK"

cp -r "$WORK" "$WORK.rst"
ctl "$WORK.rst/system/controlDict" 0 "$T0" "$T0"
"$BUILD/brae_rhoSimpleFoam" -case "$WORK.rst" > "$WORK.rst/log1" 2>&1 || true
ctl "$WORK.rst/system/controlDict" "$T0" "$T1" "$T0"
"$BUILD/brae_rhoSimpleFoam" -case "$WORK.rst" > "$WORK.rst/log2" 2>&1 || true
RSTLAST=$(last "$WORK.rst")
if [ "$RSTLAST" != "$T1" ]; then
    echo "  brae   restart $T0 -> endTime $T1 stopped at $RSTLAST, expected $T1  FAIL"
    echo "         (endTime read as a run LENGTH rather than an absolute time)"
    bad=$((bad+1))
fi
[ "$RSTLAST" = "$T1" ] && echo "  brae   restart $T0 -> endTime $T1 stops at $RSTLAST  OK"

[ "$bad" -eq 0 ] || { echo "roundtrip_vs_openfoam: $bad failures"; exit 1; }

# ---------------------------------------------------------------- 3. the round-trip identity
python3 - "$WORK.br/$T1" "$WORK.rst/$T1" "$TOL" <<'PY'
import re, sys, os, math
cont, rst, tol = sys.argv[1], sys.argv[2], float(sys.argv[3])

def rd(path):
    if not os.path.exists(path):
        return None
    t = open(path).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\(([^)]*)\)', t, re.S)
    if m:
        return [float(x) for x in m.group(1).replace('(', ' ').replace(')', ' ').split()]
    return None      # uniform internalField carries no per-cell signal; reported, not silently passed

bad, checked = 0, 0
for f in ("p", "T", "rho", "phi", "U"):
    a, b = rd(f"{cont}/{f}"), rd(f"{rst}/{f}")
    if a is None or b is None:
        print(f"  {f:<4} uniform or missing on one side -- not compared")
        continue
    if len(a) != len(b):
        print(f"  {f:<4} length {len(a)} vs {len(b)}  FAIL")
        bad += 1
        continue
    den = math.sqrt(sum(x * x for x in a))
    if den <= 0:
        print(f"  {f:<4} continuous field is identically zero -- not compared")
        continue
    l2 = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b))) / den
    ok = l2 <= tol
    print(f"  {f:<4} continuous@T1 vs restart@T1  L2rel {l2:.3e}  tol {tol:.0e}  ({len(a)} values)  {'OK' if ok else 'FAIL'}")
    checked += 1
    if not ok:
        bad += 1

if checked == 0:
    print("  round-trip compared NOTHING -- every field was uniform/missing, the gate asserts nothing")
    bad += 1
print(f"roundtrip_vs_openfoam: {bad} failures")
sys.exit(1 if bad else 0)
PY
