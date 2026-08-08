#!/bin/bash
# Gate: brae_rhoSimpleFoam vs OpenFOAM rhoSimpleFoam on a RESTART, not a fresh start.
#
# WHY THIS GATE EXISTS. Every other compressible gate starts from 0/ and only ever reads p, T and U.
# OpenFOAM's createFields.H reads two MORE fields when they are on disk, and brae read neither:
#
#   compressibleCreatePhi.H : surfaceScalarField phi(IOobject(..., READ_IF_PRESENT, ...), fvc::flux(rho*U))
#   createFields.H:12       : volScalarField    rho(IOobject(..., READ_IF_PRESENT, ...), thermo.rho())
#
# On a fresh start neither file exists, both codes take the fallback, and the whole suite is blind. On a
# RESTART OpenFOAM resumes the stored fields -- which carry the history of rho.relax() and of the
# pressure correction -- while brae recomputed them from p, T and U. Measured on the angledDuct restart
# before the fix: phi differed on 80778 of 80800 faces (3.11e-03), and the recomputed rho sat 1.9e-03
# from the stored one at the outlet. That is a different initial condition, not round-off.
#
# WHY naca0012, AND WHY NOT ANY OF THE DUCTS. Density relaxation is what makes the stored rho differ
# from thermo.rho() at all. Every other compressible case in validation/ sets
#     relaxationFactors { fields { rho 1.0; } }
# and with relaxRho = 1 the stored and recomputed densities are IDENTICAL -- the read is a provable
# no-op and the gate would assert nothing. This was measured, not assumed: pointed at rhoBox (rho 1.0)
# the negative control below PASSED, i.e. deleting phi and rho changed the answer by less than the
# tolerance. naca0012 is the only case in the suite with rho 0.01, so it is the only one where a port
# that recomputes rho instead of resuming it is exposed. (naca_vs_openfoam's own header says the same
# thing about the same case, for the fresh-start half of the problem.)
#
# THE DISCRIMINATOR is that both codes restart from the SAME OpenFOAM-written time directory. Any
# disagreement after that is brae's, not the case's.
#
# SELF-CHECK. The gate runs a negative control: the identical brae restart with T0/phi and T0/rho
# deleted, which forces brae onto the fallback the fix removed. That leg MUST fail the tolerance. If it
# passes, the gate is not discriminating anything and exits non-zero saying so -- a green gate that
# cannot go red is worse than no gate.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/naca0012" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_restart_vs_of}
# Measured, both legs, 5 iterations restarted from OF's 100/:
#     field   with the reads   negative control   ratio
#     p         4.79e-05         6.61e-02         1381x
#     T         1.75e-04         2.72e-02          155x
#     U         5.66e-05         3.30e-02          583x
#     rho       4.76e-06         1.97e-02         4131x
# 1e-3 sits in the middle of a two-order gap: ~5.7x margin above the worst passing field (T), and the
# negative control still fails by 20-66x. Tightening to 1e-5 would fail on brae's ordinary transient-path
# difference from OF -- which on this same case is p 3.25e-04 / T 1.60e-04 even at CONVERGENCE
# (naca_vs_openfoam) -- i.e. it would be measuring the port's known floor, not the restart reads.
TOL=${TOL:-1e-3}
T0=${T0:-100}    # restart from here: far enough in that rho 0.01 has accumulated real relaxation history
T1=${T1:-105}    # compare here

if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true

setctl() {   # $1 = controlDict, $2 = startTime, $3 = endTime
    python3 - "$1" "$2" "$3" <<'PY'
import re, sys
p, st, end = sys.argv[1], sys.argv[2], sys.argv[3]
s = open(p).read()
s = re.sub(r'startFrom\s+\w+;',           'startFrom      startTime;', s)
s = re.sub(r'startTime\s+[0-9.eE+-]+;',   f'startTime      {st};',      s)
s = re.sub(r'endTime\s+[0-9.eE+-]+;',     f'endTime        {end};',     s)
s = re.sub(r'writeControl\s+\w+;',        'writeControl   timeStep;',   s)
s = re.sub(r'writeInterval\s+[0-9.eE+-]+;', f'writeInterval  {int(end)-int(st)};', s)
open(p, 'w').write(s)
PY
}

# ---------------------------------------------------------------- 1. OF runs 0 -> T0 and writes it
rm -rf "$WORK" "$WORK.of2" "$WORK.br" "$WORK.neg"
cp -r "$SRC" "$WORK"
cd "$WORK"
# The tutorial projects the block mesh onto OF's shipped NACA0012 geometry (see naca_vs_openfoam.sh).
mkdir -p constant/geometry
cp -f "$FOAM_TUTORIALS"/resources/geometry/NACA0012.obj.gz constant/geometry/
mkdir -p 0 && cp 0.orig/* 0/
blockMesh                        > log.blockMesh       2>&1
transformPoints -scale '(1 0 1)' > log.transformPoints 2>&1
extrudeMesh                      > log.extrudeMesh     2>&1
topoSet                          > log.topoSet         2>&1
[ -f constant/polyMesh/owner ] || { echo "restart_vs_openfoam: mesh generation FAILED"; exit 1; }
# brae REFUSES a compressible case carrying fvOptions rather than dropping the constraint silently, so
# the gate removes limitTemperature from BOTH sides -- exactly as naca_vs_openfoam does.
rm -f system/fvOptions
setctl system/controlDict 0 "$T0"
rhoSimpleFoam > log.rhoSimpleFoam 2>&1 || true
[ -d "$WORK/$T0" ] || { echo "restart_vs_openfoam: OF did not write $T0/"; exit 1; }
for f in phi rho; do
    [ -f "$WORK/$T0/$f" ] || { echo "restart_vs_openfoam: OF wrote no $T0/$f -- nothing to restart-read, gate is void"; exit 1; }
done

# ---------------------------------------------------------------- 2. three restarts from the same T0/
for d in of2 br neg; do
    cp -r "$WORK" "$WORK.$d"
    rm -f "$WORK.$d"/log.*
    setctl "$WORK.$d/system/controlDict" "$T0" "$T1"
done
# the negative control: deny brae the two stored fields, i.e. the pre-fix behaviour
rm -f "$WORK.neg/$T0/phi" "$WORK.neg/$T0/rho"

( cd "$WORK.of2" && rhoSimpleFoam > log.of 2>&1 || true )
"$BUILD/brae_rhoSimpleFoam" -case "$WORK.br"  > "$WORK.br/log.brae"  2>&1 || true
"$BUILD/brae_rhoSimpleFoam" -case "$WORK.neg" > "$WORK.neg/log.brae" 2>&1 || true
for d in of2 br neg; do
    [ -d "$WORK.$d/$T1" ] || { echo "restart_vs_openfoam: $d wrote no $T1/ -- see $WORK.$d/log.*"; exit 1; }
done

# ---------------------------------------------------------------- 3. compare, and check the check
python3 - "$WORK.of2/$T1" "$WORK.br/$T1" "$WORK.neg/$T1" "$TOL" <<'PY'
import re, sys, math
ofd, brd, negd, tol = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])

def rd(p):
    t = open(p).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\(([^)]*)\)', t, re.S)
    if not m:
        return None
    v = [float(x) for x in m.group(1).replace('(', ' ').replace(')', ' ').split()]
    return v

def l2(a, b):
    n = min(len(a), len(b))
    den = sum(x*x for x in a[:n])
    return math.sqrt(sum((x-y)**2 for x, y in zip(a[:n], b[:n])) / den) if den > 0 else float('nan')

bad, blind = 0, []
for f in ("p", "T", "U", "rho"):
    of, br, ng = rd(f"{ofd}/{f}"), rd(f"{brd}/{f}"), rd(f"{negd}/{f}")
    if of is None or br is None or ng is None:
        print(f"  {f}: FAIL uniform/missing field -- degenerate case, fix the case not the tolerance")
        bad += 1
        continue
    n = min(len(of), len(br))
    spread = (max(of[:n]) - min(of[:n])) / (abs(sum(of[:n])/n) + 1e-300)
    r_br, r_ng = l2(of, br), l2(of, ng)
    ok = r_br <= tol and spread > 1e-6           # spread guard: refuse to pass on a flat field
    print(f"  {f:<4} restart-read L2rel {r_br:.4e}  tol {tol:.0e}  spread {spread:.3e}  {'OK' if ok else 'FAIL'}")
    print(f"  {f:<4} negative ctl L2rel {r_ng:.4e}  (must exceed tol)")
    if not ok:
        bad += 1
    if r_ng <= tol:
        blind.append(f)

if blind:
    print(f"restart_vs_openfoam: NEGATIVE CONTROL PASSED for {blind} -- deleting phi/rho changed nothing,")
    print("  so this gate cannot detect the defect it exists for. Treat as failure.")
    bad += len(blind)
print(f"restart_vs_openfoam: {bad} failures")
sys.exit(1 if bad else 0)
PY
