#!/bin/bash
# heRhoThermo vs hePsiThermo: WHEN rho sees the just-solved pressure.
#
# OF's rhoSimpleFoam ends pEqn.H/pcEqn.H with `rho = thermo.rho()` and NO thermo.correct(), so T, psi, mu
# and alpha keep the values EEqn's thermo.correct() gave them and only rho moves. What `thermo.rho()`
# RETURNS depends on the thermo type:
#
#     psiThermo::rho()  ->  p_*psi_    recomputed with the JUST-SOLVED p       (psiThermo.C:150)
#     rhoThermo::rho()  ->  rho_       the STORED field, from BEFORE the solve (rhoThermo.C:233)
#
# So a heRhoThermo case carries a rho that LAGS the pressure by one outer iteration; a hePsiThermo case
# does not. brae accepted heRhoThermo on the grounds that rho == psi*p exactly for perfectGas -- true of
# the arithmetic, false of the timing -- and recomputed rho with the new p in both cases.
#
# ONE ITERATION is compared, not a converged field, for two reasons: the effect IS a one-iteration lag, and
# squareBend (the only heRhoThermo tutorial in reach) does not yet converge in brae for unrelated reasons
# (B1). Waiting for convergence would mean never gating this at all.
#
# A low-Mach duct cannot test this: rho barely depends on p there, so both timings agree. This uses
# squareBend (Mach 0.958), where one pressure solve moves rho by percent.
#
# Measured at iteration 1, brae vs OF:
#     recompute with new p (hePsiThermo timing) : rho L2rel 7.02e-02
#     keep the stored rho  (heRhoThermo timing) : rho L2rel 9.52e-04     <- 74x
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/sbMatched" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_rhotiming}
TOL=${TOL:-5e-3}

if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true

# The oracle is tools/dumpPEqn: OF's own rhoSimpleFoam, writing the pressure equation's inputs at
# iteration 1. Without it there is nothing to compare against, and passing would be meaningless.
DUMP="$(command -v dumpPEqn || true)"
[ -z "$DUMP" ] && [ -n "${FOAM_USER_APPBIN:-}" ] && [ -x "$FOAM_USER_APPBIN/dumpPEqn" ] && DUMP="$FOAM_USER_APPBIN/dumpPEqn"
if [ -z "$DUMP" ]; then
    echo "dumpPEqn not built -- build it with: (cd tools/dumpPEqn && wmake)"
    echo "skipping rather than passing without an oracle"
    exit 77
fi

setup()
{
    rm -rf "$1"; mkdir -p "$(dirname "$1")"; cp -r "$SRC" "$1"
    ( cd "$1" && blockMesh > log.blockMesh 2>&1 )
    mkdir -p "$1/0" && cp "$1"/0.orig/* "$1/0/" 2>/dev/null || true
    sed -i 's/^endTime.*/endTime 1;/' "$1/system/controlDict"
    sed -i 's/writeInterval.*/writeInterval 1;/' "$1/system/controlDict"
}

setup "$WORK/of"
( cd "$WORK/of" && "$DUMP" > log.dumpPEqn 2>&1 )
setup "$WORK/br"
cp -r "$WORK/of/constant/polyMesh" "$WORK/br/constant/"
BRAE_TRANSONIC=1 "$BUILD/brae_rhoSimpleFoam" -case "$WORK/br" > "$WORK/br/log.brae" 2>&1

if [ ! -f "$WORK/of/1/rho_" ]; then
    echo "  FAIL dumpPEqn wrote no rho_ -- no oracle"; tail -20 "$WORK/of/log.dumpPEqn"; exit 1
fi

python3 - "$WORK/of/1/rho_" "$WORK/br/1/rho" "$TOL" <<'PY'
import re, sys, math
def rd(p):
    t = open(p).read()
    m = re.search(r'internalField\s+nonuniform[^(]*\((.*?)\)\s*;\s*boundaryField', t, re.S)
    return [float(x) for x in m.group(1).split()] if m else None

of, br, tol = rd(sys.argv[1]), rd(sys.argv[2]), float(sys.argv[3])
if of is None or br is None:
    print(f"  FAIL rho missing (OF={of is not None} brae={br is not None})"); sys.exit(1)
n = min(len(of), len(br))
l2 = math.sqrt(sum((a-b)**2 for a, b in zip(of[:n], br[:n])) / sum(a*a for a in of[:n]))
print(f"  rho @ iteration 1: L2rel {l2:.4e}  tol {tol:.0e}  "
      f"OF mean {sum(of)/len(of):.6g}  brae mean {sum(br)/len(br):.6g}")

# The case must actually EXERCISE the timing: if one pressure solve barely moves rho, both timings agree
# and this gate proves nothing. OF's own rho spread across the domain stands in for that.
spread = (max(of) - min(of)) / (sum(of)/len(of))
print(f"  OF rho spread {spread:.4f} (needs > 0.01, else the case cannot distinguish the two timings)")
bad = 0
if l2 > tol:
    print("       ~7e-02 here means brae is recomputing rho with the JUST-SOLVED p (hePsiThermo timing)")
    print("       on a heRhoThermo case. OF's rhoThermo::rho() returns the STORED rho_, which lags the")
    print("       pressure by one outer iteration (ThermoCoeffs::rhoThermoType).")
    bad += 1
if spread <= 0.01:
    print("  FAIL the case stopped exercising a compressible pressure-density coupling")
    bad += 1
print(f"rhotiming_vs_openfoam: {bad} failures")
sys.exit(1 if bad else 0)
PY
