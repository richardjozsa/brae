#!/usr/bin/env bash
# One stock simpleFoam TUTORIAL, run by real OpenFOAM and by brae on the same mesh, converged fields
# compared. The case is read straight from the OpenFOAM installation, so nothing has to be committed --
# this gate costs a tutorial name and a set of bounds.
#
# WHY THESE TUTORIALS. The OF-mirror rebuild's whole point is running the real simpleFoam tutorials
# faithfully, and nine of the seventeen already have a dedicated gate built around whatever defect they
# exposed. The ones here had none: they were run once, agreed, and nothing kept them agreeing.
#
#   rotatingCylinders      6400 cells, laminar, MRF, `bounded Gauss linearUpwind grad(U)`
#   backwardFacingStep2D  20540 cells, kOmegaSST, SIMPLEC, `bounded Gauss LUST grad(U)`
#
# The second is the only simpleFoam tutorial that names LUST, so it is the one thing standing behind
# brae's internal LUST weights -- worth a gate on its own account.
#
# FUNCTION OBJECTS ARE STRIPPED, and not for convenience: they compute no field this gate reads, and
# backwardFacingStep2D's stressComponents+pressureCoefficient pair hangs OpenFOAM in this build (thirty
# minutes at 100% CPU on 20540 cells without writing a single `Time =`).
#
# CONVERGENCE IS ASSERTED, NOT ASSUMED. Comparing two codes at a fixed iteration count compares
# TRAJECTORIES unless both have arrived, and the difference is not small: squareBend at its tutorial
# endTime of 500 reads U 2.759e-02 against OpenFOAM, and at 8000 -- where both codes sit on the same
# 2.5e-05 residual plateau -- it reads 1.773e-03. Same code, same case, a factor of 16 from nothing but
# where the comparison was taken. So this refuses to report a number unless both runs finished at a
# residual worth comparing, or stopped because their own residualControl said to.
set -u
CASE="${1:?tutorial name under incompressible/simpleFoam}"
BOUNDS="${2:?field bounds, e.g. U=2e-03,p=5e-03}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BRAE="${BRAE_BIN:-$ROOT/build/brae}"
TUT=/usr/lib/openfoam/openfoam2412/tutorials/incompressible/simpleFoam
OFBIN=/usr/lib/openfoam/openfoam2412/platforms/linuxARM64GccDPInt32Opt
[ -x "$BRAE" ]                 || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
[ -x "$OFBIN/bin/simpleFoam" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
[ -d "$TUT/$CASE" ]            || { echo "SKIP: no $CASE tutorial in this OpenFOAM"; exit 77; }

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
set +u
# shellcheck disable=SC1091
source /usr/lib/openfoam/openfoam2412/etc/bashrc > /dev/null 2>&1 || true
set -u

cp -r "$TUT/$CASE" "$W/of" || exit 1
python3 - "$W/of" <<'PY'
import re, sys, os
c = os.path.join(sys.argv[1], 'system/controlDict')
s = open(c).read()
s = re.sub(r'functions\s*\{.*?\n\}', 'functions\n{\n}', s, flags=re.S)
s = re.sub(r'^writeFormat .*',    'writeFormat     ascii;', s, flags=re.M)
s = re.sub(r'^writePrecision .*', 'writePrecision  15;', s, flags=re.M)
open(c, 'w').write(s)
PY
# Allrun's EXIT STATUS is not the test -- rotatingCylinders ends with a `./plot` step that fails when
# gnuplot is absent, long after simpleFoam has run and written its result. What matters is whether
# OpenFOAM produced a solution, which is checked below by looking for one.
( cd "$W/of" && timeout 2400 ./Allrun > allrun.log 2>&1 ) || true
[ -f "$W/of/log.simpleFoam" ] \
    || { echo "FAIL: OpenFOAM never ran simpleFoam for $CASE"; tail -5 "$W/of/allrun.log"; exit 1; }
grep -q '^End' "$W/of/log.simpleFoam" \
    || { echo "FAIL: OpenFOAM's simpleFoam did not finish for $CASE"; tail -5 "$W/of/log.simpleFoam"; exit 1; }

# brae runs the SAME mesh and the same dictionaries -- only the solver differs.
cp -r "$W/of" "$W/brae"
rm -rf "$W"/brae/[1-9]* "$W"/brae/0.[0-9]* "$W"/brae/log.* "$W"/brae/postProcessing "$W"/brae/allrun.log
( cd "$W/brae" && timeout 2400 "$BRAE" . > run.log 2>&1 ) \
    || { echo "FAIL: brae refused or crashed on $CASE"; tail -15 "$W/brae/run.log"; exit 1; }

GATE_BOUNDS="$BOUNDS" GATE_CASE="$CASE" python3 - "$W" <<'PY'
import os, re, sys
import numpy as np
W = sys.argv[1]
BOUND = {k: float(v) for k, v in (kv.split('=') for kv in os.environ['GATE_BOUNDS'].split(','))}

def lastTime(d):
    ts = [x for x in os.listdir(d) if re.fullmatch(r'[0-9]+(\.[0-9]+)?', x) and x != '0']
    return None if not ts else max(ts, key=float)

def read(d, f):
    b = open(os.path.join(d, f), 'rb').read()
    m = re.search(rb'internalField\s+nonuniform\s+List<(scalar|vector)>\s*\n(\d+)\s*\n\(', b)
    if not m:
        m2 = re.search(rb'internalField\s+uniform\s+([^;]+);', b)
        return None if not m2 else np.array([[float(x) for x in re.findall(rb'[-+0-9.eE]+', m2.group(1))]])
    typ = m.group(1).decode(); n = int(m.group(2)); nc = 3 if typ == 'vector' else 1
    fm = re.search(r'format\s+(\w+)', b[:1024].decode('latin-1'))
    if fm and fm.group(1) == 'binary':
        return np.frombuffer(b[m.end():m.end()+n*nc*8], dtype='<f8').reshape(n, nc)
    txt = b[m.end():].decode('latin-1')
    vals = re.findall(r'[-+0-9.eE]+', txt.split(')\n;')[0] if ')\n;' in txt else txt)
    return np.array([float(x) for x in vals[:n*nc]]).reshape(n, nc)

def rel(a, b):
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))

rc = 0
to, tb = lastTime(os.path.join(W, 'of')), lastTime(os.path.join(W, 'brae'))
if to is None or tb is None:
    print("  FAIL: %s wrote no result" % ('OpenFOAM' if to is None else 'brae')); sys.exit(1)
od, bd = os.path.join(W, 'of', to), os.path.join(W, 'brae', tb)
print("  %s: OpenFOAM t=%s, brae t=%s" % (os.environ['GATE_CASE'], to, tb))

# 1. Both must have arrived, or everything below compares trajectories. A run that satisfied the case's
#    own residualControl counts as arrived; one that hit endTime must be at a residual worth comparing.
print("  1. both codes reached a steady state")
for k, log in (('OpenFOAM', os.path.join(W, 'of', 'log.simpleFoam')),
               ('brae',     os.path.join(W, 'brae', 'run.log'))):
    if not os.path.exists(log):
        print("     %-9s FAIL: no solver log" % k); rc = 1; continue
    txt = open(log, errors='ignore').read()
    stopped = 'solution converged' in txt
    res = re.findall(r'Solving for Ux, Initial residual = ([0-9.eE+-]+)', txt)
    if not res:
        print("     %-9s FAIL: no Ux residual in the log" % k); rc = 1; continue
    v = float(res[-1])
    ok = stopped or v < 1e-04
    print("     %-9s Ux %.3e  (%s)   %s" % (k, v, "residualControl" if stopped else "ran to endTime",
                                            "ok" if ok else "FAIL: still moving"))
    if not ok: rc = 1

# 2. The fields.
print("  2. brae vs OpenFOAM")
fields = [f for f in BOUND if os.path.exists(os.path.join(od, f)) and os.path.exists(os.path.join(bd, f))]
if not fields:
    print("     FAIL: none of the named fields exist in both results"); sys.exit(1)
for f in fields:
    e = rel(read(bd, f), read(od, f))
    ok = e < BOUND[f]
    print("     %-8s L2 rel %.3e   %s" % (f, e, "ok" if ok else "FAIL (> %.0e)" % BOUND[f]))
    if not ok: rc = 1

# 3. THE CONTROL, and it is what makes the bounds above mean anything: the INITIAL field must FAIL them.
#    Without it a bound loose enough to pass a broken solver also passes a solver that did nothing.
print("  3. control -- the initial field must NOT pass those bounds")
z = os.path.join(W, 'of', '0')
worst = 0.0
for f in fields:
    if not os.path.exists(os.path.join(z, f)):
        continue
    try:
        e = rel(read(z, f), read(od, f))
    except Exception:
        continue
    worst = max(worst, e / BOUND[f])
    print("     %-8s L2 rel %.3e  (%.0fx its bound)" % (f, e, e / BOUND[f]))
if worst <= 1.0:
    print("     FAIL: the initial field already meets the bounds, so they discriminate nothing"); rc = 1
sys.exit(rc)
PY
rc=$?
[ $rc -eq 0 ] && echo "  ok:   brae reproduces the $CASE tutorial"
exit $rc
