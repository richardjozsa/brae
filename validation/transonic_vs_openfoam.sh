#!/bin/bash
# squareBend (rhoSimpleFoam, transonic + consistent, Mach ~0.96) vs OpenFOAM v2412, CONVERGED.
#
# This gate is the evidence that lifted the `transonic yes` refusal, so it is the one test standing
# between brae and shipping a branch that was refused for months. What it covers that nothing else does:
#
#   - TRANSONIC. `transonic yes` makes the pressure equation gain an implicit fvm::div(phid,p) with
#     phid = (psi_f/rho_f)*phiHbyA, which turns the pressure matrix ASYMMETRIC. AMG-PCG cannot converge
#     on it (it burns the iteration cap every step), so the branch also selects BiCGStab -- OF confirms
#     the same constraint independently by refusing to start on a solver/matrix mismatch in BOTH
#     directions (DIC on transonic, DILU on subsonic). No other gate exercises an asymmetric p matrix.
#   - SIMPLEC as well (`consistent yes`): rAtU = 1/(1/rAU - H1), and pcEqn.H's ordering, where the
#     phid and the interp(psi*p)*phiHbyA/interp(rho) subtraction both use the ORIGINAL phiHbyA rather
#     than the SIMPLEC-corrected one. Getting that order wrong gave contGlobal -283 at iteration 1.
#   - heRhoThermo, so `rho = thermo.rho()` returns the STORED field and thermo.correct() legitimately
#     moves rho_ -- the opposite of hePsiThermo. That distinction is what naca_vs_openfoam cannot see.
#   - Mach ~0.96 with T from 884 K to 1045 K, so compressibility is not a perturbation.
#
# WHY IT WAS REFUSED, and what actually fixed it. squareBend diverged (contGlobal -2.83e+02 at
# iteration 1, NaN by 50) and the transonic assembly was blamed. It was not the cause: the defects were
# in the SHARED compressible path, and the last one -- forcing updateRho=false at thermo.correct() for
# every thermo type, when heRhoThermo's calculate() legitimately sets rho_ -- was introduced while
# fixing aerofoilNACA0012 and took this case from converged-in-136 to NaN. Fixing the chain fixed the
# branch. Do not read this gate as "transonic was broken and got fixed".
#
# Measured brae vs OF at convergence (brae 160 iterations, OF 156):
#     p 1.77e-03  U 1.43e-03  T 6.00e-04  rho 1.62e-03  k 7.37e-03  epsilon 8.32e-03  nut 4.93e-03
# Tolerances keep ~3x margin on the mean flow and ~4x on the turbulence.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
TUT=${TUT:-compressible/rhoSimpleFoam/squareBend}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_transonic}
TOL_MEAN=${TOL_MEAN:-5e-3}      # p, U, T, rho
TOL_TURB=${TOL_TURB:-3e-2}      # k, epsilon, nut

if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true
SRC="$FOAM_TUTORIALS/$TUT"
if [ ! -d "$SRC" ]; then echo "tutorial $TUT not found -- skipping"; exit 77; fi

rm -rf "$WORK" "$WORK.brae"
cp -r "$SRC" "$WORK"
cd "$WORK"
if [ -f Allrun.pre ]; then ./Allrun.pre > log.pre 2>&1; else blockMesh > log.pre 2>&1; fi
[ -d 0.orig ] && { mkdir -p 0; cp -r 0.orig/* 0/ 2>/dev/null || true; }
if [ ! -f constant/polyMesh/owner ]; then echo "transonic_vs_openfoam: mesh generation FAILED"; exit 1; fi

# The case must actually BE transonic, else this gate silently degrades into a second subsonic test.
if ! grep -qE '^[[:space:]]*transonic[[:space:]]+yes' system/fvSolution; then
    echo "transonic_vs_openfoam: FAIL $TUT is not 'transonic yes' -- this gate would test nothing"
    exit 1
fi

sed -i 's/^writeInterval.*/writeInterval   5000;/' system/controlDict
rhoSimpleFoam > log.rhoSimpleFoam 2>&1 || true
OFLAST=$(ls -d [0-9]* 2>/dev/null | grep -vx 0 | sort -g | tail -1)
if [ -z "$OFLAST" ]; then echo "transonic_vs_openfoam: OF wrote no time directory"; exit 1; fi
if ! grep -q "SIMPLE solution converged" log.rhoSimpleFoam; then
    echo "transonic_vs_openfoam: OF did NOT converge -- no converged oracle to compare against"
    exit 1
fi

cp -r "$SRC" "$WORK.brae"
cp -r "$WORK/constant/polyMesh" "$WORK.brae/constant/"
[ -d "$WORK.brae/0.orig" ] && { mkdir -p "$WORK.brae/0"; cp -r "$WORK.brae"/0.orig/* "$WORK.brae/0/" 2>/dev/null || true; }
sed -i 's/^writeInterval.*/writeInterval   5000;/' "$WORK.brae/system/controlDict"
# NO BRAE_TRANSONIC here, deliberately: the point of this gate is that the branch runs by DEFAULT.
# If the refusal is ever reinstated, brae exits and the check below fails, which is the intended signal.
"$BUILD/brae_rhoSimpleFoam" -case "$WORK.brae" > "$WORK.brae/log.brae" 2>&1 || true
if grep -q "NOT VALIDATED, so it is refused" "$WORK.brae/log.brae"; then
    echo "transonic_vs_openfoam: brae REFUSED the case -- the transonic refusal is back in place"
    exit 1
fi
BRLAST=$(ls -d "$WORK.brae"/[0-9]* 2>/dev/null | grep -v '/0$' | sort -g | tail -1)
if [ -z "$BRLAST" ]; then echo "transonic_vs_openfoam: brae wrote no time directory"; tail -1 "$WORK.brae/log.brae"; exit 1; fi
if ! grep -q "converged" "$WORK.brae/log.brae"; then
    echo "transonic_vs_openfoam: brae did NOT converge -- FAIL"
    grep -E "^Time = " "$WORK.brae/log.brae" | tail -1
    exit 1
fi

OFIT=$(grep -c "^Time = " log.rhoSimpleFoam || true)
BRIT=$(grep -oE "converged in [0-9]+ iterations" "$WORK.brae/log.brae" | grep -oE "[0-9]+" | head -1)
echo "  transonic + consistent: OF ${OFIT} iterations, brae ${BRIT} iterations"

python3 - "$WORK/$OFLAST" "$BRLAST" "$TOL_MEAN" "$TOL_TURB" <<'PY'
import re, sys, math
ofd, brd = sys.argv[1], sys.argv[2]
tol_mean, tol_turb = float(sys.argv[3]), float(sys.argv[4])

def rd(path):
    try:
        t = open(path).read()
    except OSError:
        return None
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*boundaryField', t, re.S)
    if not m:
        return None
    body = m.group(1).replace('(', ' ').replace(')', ' ')
    return [float(x) for x in re.findall(r'-?\d+\.?\d*(?:[eE][-+]?\d+)?', body)]

FIELDS = (("p", tol_mean), ("U", tol_mean), ("T", tol_mean), ("rho", tol_mean),
          ("k", tol_turb), ("epsilon", tol_turb), ("nut", tol_turb))
bad = checked = 0
for f, tol in FIELDS:
    of, br = rd(f"{ofd}/{f}"), rd(f"{brd}/{f}")
    if of is None or br is None:
        print(f"  {f}: SKIP (not on both sides: OF={of is not None} brae={br is not None})")
        continue
    n = min(len(of), len(br))
    denom = sum(a * a for a in of[:n])
    if denom <= 0.0:
        print(f"  {f}: FAIL OF field is all zeros -- nothing to compare")
        bad += 1
        continue
    l2 = math.sqrt(sum((a - b) ** 2 for a, b in zip(of[:n], br[:n])) / denom)
    mean = abs(sum(of[:n]) / n) + 1e-300
    spread = (max(of[:n]) - min(of[:n])) / mean
    ok = l2 <= tol and spread > 1e-6          # a uniform field agrees for free and tests nothing
    checked += 1
    bad += 0 if ok else 1
    why = "" if spread > 1e-6 else "  (field is UNIFORM -- tests nothing)"
    print(f"  {f:8s} L2rel {l2:.4e}  tol {tol:.0e}  OF range [{min(of[:n]):.6g}, {max(of[:n]):.6g}]"
          f"  {'ok' if ok else 'FAIL'}{why}")

if checked < 6:
    print(f"  FAIL only {checked} fields compared -- must cover the mean flow AND the turbulence")
    bad += 1
print(f"transonic_vs_openfoam: {bad} failures")
sys.exit(1 if bad else 0)
PY
