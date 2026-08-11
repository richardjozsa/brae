#!/bin/bash
# aerofoilNACA0012 (rhoSimpleFoam, kOmegaSST, subsonic external aero) vs OpenFOAM v2412, CONVERGED.
#
# WHY THIS CASE. It is the only case in reach that exercises the compressible SIMPLE loop where the
# things that hid every other defect are absent:
#
#   - HIGH SPEED. |U| ~ 280 m/s, so the kinetic term Ekp = 0.5|U|^2 + p/rho is ~57% of e. On the ducts
#     it is a rounding error, which is why a wrong energy definition converged there for months.
#   - A NEARLY SINGULAR PRESSURE. freestreamPressure is a mixed BC: the WHOLE boundary anchors the
#     pressure level with sum|iC| ~ 2e-3 against a comparable diagonal in each of 16000 cells. The
#     solution's LEVEL is therefore set by the net mass balance and amplified by ~1/sum|iC|, so a 7%
#     error in sum(pSrc) moves p by tens of kPa. A duct with a fixedValue outlet cannot show this.
#   - HEAVY DENSITY RELAXATION. relaxationFactors/fields/rho = 0.01, so OF's rho barely moves per
#     iteration. Any port that RECOMPUTES rho instead of relaxing the stored field is exposed here and
#     nowhere else -- with relaxRho = 1 the two are identical.
#   - REAL TURBULENCE with wall functions, and a wake where the wall distance matters.
#
# Six defects were found on this case, and each of them converged happily on some other case:
#   1  alphat not set by validate()          -> alphaEff 46x too small in the FIRST energy solve
#   2  he = Cv*T instead of Cp*(T-Tref)-R*T  -> he offset by Cp*Tstd = 3.0e5 J/kg
#   3  bound() applied to he                 -> OF's sensible energy is NEGATIVE; T pinned at 417.7 K
#   4  linearUpwind's gradient arg ignored   -> unlimited grad where the case asked cellLimited
#   5  thermo.correct() overwrote rho        -> discarded rho.relax(); rho ratio 0.085..2.07
#   6  the BOUNDARY half of the same rho     -> inlet mass influx 12.4% high, p pinned at pMax
# The method that found them is written up in cudafoam/solver-porting-dissection.md.
#
# WHY CONVERGED, not one iteration. Defects 1-4 show at iteration 1 and a cheap gate would catch them.
# Defects 5 and 6 do NOT: every stage of iteration 2 agreed to 1e-3 or better while the run was still
# heading for the pMax ceiling, because the error lived in the net mass balance rather than in any
# field norm. Only running to convergence distinguishes "matches OF" from "converges to something".
# That is what this gate is for, and it is why it is slow and labelled so.
#
# The gate also asserts brae actually CONVERGED (residualControl satisfied) rather than merely reaching
# endTime -- reaching endTime with a plausible field is exactly what the pinned-pressure bug looked like.
#
# WHAT THIS GATE DOES *NOT* COVER, measured by mutation rather than assumed:
#   - reverting defect 6 (boundary rho unrelaxed): FAILS, "brae did NOT converge ... Ux nan"  <- caught
#   - reverting defect 2 (he = Cv*T, no Tref offset): PASSES, 739 iterations, every field in tolerance.
#     That is not a hole to paper over -- it is the mechanism working as understood. The offset injects
#     a spurious source C*div(phi), and div(phi) -> 0 at convergence, so with the other five defects
#     fixed the run lands on the same answer. F2 is therefore covered by the `sensible_energy` UNIT
#     test (9 assertions go red on the old formula), NOT here. A converged-field gate structurally
#     cannot see a defect that vanishes at convergence; do not claim it does.
#
# Measured brae vs OF at convergence (brae 736 iterations, OF 668):
#     p 3.25e-04   U 5.86e-04   T 1.60e-04   rho 3.73e-04   k 1.22e-02   omega 3.22e-02   nut 4.19e-02
# Tolerances below keep ~6x margin on the mean flow and ~2.4x on the turbulence.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/naca0012" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_naca}
TOL_MEAN=${TOL_MEAN:-2e-3}      # p, U, T, rho
TOL_TURB=${TOL_TURB:-1e-1}      # k, omega, nut

if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true

rm -rf "$WORK" "$WORK.brae"
cp -r "$SRC" "$WORK"
cd "$WORK"

# The tutorial projects the block mesh onto the shipped NACA0012 geometry, so it comes from OF's own
# resources rather than being vendored here (a copy would drift from the tutorial silently).
mkdir -p constant/geometry
cp -f "$FOAM_TUTORIALS"/resources/geometry/NACA0012.obj.gz constant/geometry/
mkdir -p 0 && cp 0.orig/* 0/
blockMesh                        > log.blockMesh      2>&1
transformPoints -scale '(1 0 1)' > log.transformPoints 2>&1
extrudeMesh                      > log.extrudeMesh    2>&1
topoSet                          > log.topoSet        2>&1
if [ ! -f constant/polyMesh/owner ]; then echo "naca_vs_openfoam: mesh generation FAILED"; exit 1; fi

# fvOptions: the tutorial carries `limitTemperature min 101 max 1000`. brae REFUSES a compressible case
# with an fvOptions file rather than silently dropping the constraint, which is the correct behaviour --
# so the gate removes it from BOTH sides. That is only legitimate if the constraint never actually binds,
# and "never binds" is asserted below against OF's own converged T rather than assumed. If someone
# changes the case so the limiter engages, this gate FAILS instead of quietly comparing two different
# problems.
TMIN=$(sed -n 's/^ *min *\([0-9.]*\) *;/\1/p' system/fvOptions | head -1)
TMAX=$(sed -n 's/^ *max *\([0-9.]*\) *;/\1/p' system/fvOptions | head -1)
rm -f system/fvOptions
sed -i 's/^writeInterval.*/writeInterval   5000;/' system/controlDict   # only the converged field is compared

rhoSimpleFoam > log.rhoSimpleFoam 2>&1 || true
OFLAST=$(ls -d [0-9]* 2>/dev/null | grep -vx 0 | sort -g | tail -1)
if [ -z "$OFLAST" ]; then echo "naca_vs_openfoam: OF wrote no time directory"; exit 1; fi
if ! grep -q "SIMPLE solution converged" log.rhoSimpleFoam; then
    echo "naca_vs_openfoam: OF did NOT converge -- the oracle is not a converged field, nothing to compare"
    exit 1
fi

# brae runs the same case in a clean copy, sharing only the mesh OF generated.
cp -r "$SRC" "$WORK.brae"
cp -r "$WORK/constant/polyMesh" "$WORK.brae/constant/"
mkdir -p "$WORK.brae/0" && cp "$WORK.brae"/0.orig/* "$WORK.brae/0/"
rm -f "$WORK.brae/system/fvOptions"
sed -i 's/^writeInterval.*/writeInterval   5000;/' "$WORK.brae/system/controlDict"
"$BUILD/brae_rhoSimpleFoam" -case "$WORK.brae" > "$WORK.brae/log.brae" 2>&1 || true
BRLAST=$(ls -d "$WORK.brae"/[0-9]* 2>/dev/null | grep -v '/0$' | sort -g | tail -1)
if [ -z "$BRLAST" ]; then echo "naca_vs_openfoam: brae wrote no time directory"; sed -n '$p' "$WORK.brae/log.brae"; exit 1; fi

# The regression this gate exists for: the pinned-pressure bug reached endTime with a plausible-looking
# field. Reaching endTime is a FAILURE here, not a pass.
if ! grep -q "converged" "$WORK.brae/log.brae"; then
    echo "naca_vs_openfoam: brae did NOT converge (residualControl never satisfied) -- FAIL"
    grep -E "^Time = " "$WORK.brae/log.brae" | tail -1
    exit 1
fi

OFIT=$(grep -c "^Time = " log.rhoSimpleFoam || true)
BRIT=$(grep -oE "converged in [0-9]+ iterations" "$WORK.brae/log.brae" | grep -oE "[0-9]+" | head -1)
echo "  converged: OF ${OFIT} iterations, brae ${BRIT} iterations"

python3 - "$WORK/$OFLAST" "$BRLAST" "$TOL_MEAN" "$TOL_TURB" "$TMIN" "$TMAX" <<'PY'
import re, sys, math
ofd, brd = sys.argv[1], sys.argv[2]
tol_mean, tol_turb = float(sys.argv[3]), float(sys.argv[4])
tmin, tmax = float(sys.argv[5]), float(sys.argv[6])

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

bad = 0

# The fvOptions removal is only legitimate while the temperature constraint never binds. Assert it
# against OF's own converged field, so a case change that engages the limiter fails loudly.
T_of = rd(f"{ofd}/T")
if T_of is None:
    print("  T: FAIL OF wrote no T -- cannot verify the fvOptions removal was inert")
    bad += 1
else:
    lo, hi = min(T_of), max(T_of)
    if lo <= tmin or hi >= tmax:
        print(f"  fvOptions: FAIL limitTemperature [{tmin}, {tmax}] BINDS on OF's converged T "
              f"[{lo:.2f}, {hi:.2f}] -- removing it compares a different problem")
        bad += 1
    else:
        print(f"  fvOptions: limitTemperature [{tmin}, {tmax}] never binds "
              f"(OF T [{lo:.2f}, {hi:.2f}]) -- removal from both sides is inert")

FIELDS = (("p", tol_mean), ("U", tol_mean), ("T", tol_mean), ("rho", tol_mean),
          ("k", tol_turb), ("omega", tol_turb), ("nut", tol_turb))
checked = 0
for f, tol in FIELDS:
    of, br = rd(f"{ofd}/{f}"), rd(f"{brd}/{f}")
    if of is None or br is None:
        print(f"  {f}: SKIP (not written on both sides: OF={of is not None} brae={br is not None})")
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
    # A uniform field agrees to machine precision while testing nothing.
    ok = l2 <= tol and spread > 1e-6
    checked += 1
    if not ok:
        bad += 1
    why = "" if spread > 1e-6 else "  (field is UNIFORM -- tests nothing)"
    print(f"  {f:6s} L2rel {l2:.4e}  tol {tol:.0e}  OF range [{min(of[:n]):.6g}, {max(of[:n]):.6g}]"
          f"  {'ok' if ok else 'FAIL'}{why}")

if checked < 6:
    print(f"  FAIL only {checked} fields compared -- the gate must cover the mean flow AND the turbulence")
    bad += 1
print(f"naca_vs_openfoam: {bad} failures")
sys.exit(1 if bad else 0)
PY
