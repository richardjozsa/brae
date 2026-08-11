#!/bin/bash
# D2: boundary COEFFICIENTS vs OpenFOAM, for the BC types that actually caused bugs.
#
# D1 verified the assembled diagonal and source against OF's fvScalarMatrix to 1.76e-06, but only with
# plain fixedValue/zeroGradient patches -- diag_compare.cu builds its fields with a local loader that
# understands nothing else. So the numbers that decide how a BC enters the matrix were never compared for
# any BC that is not trivial.
#
# That is exactly where group A lived. internalCoeffs goes to the adjacent cell's DIAGONAL and
# boundaryCoeffs to its SOURCE, and four separate defects were "the boundary value exists but never reaches
# the discretisation": A1 (inletOutlet on T never resolved -> T 276% off), A12 (grad(K) taking the energy
# field's descriptor), B5 (fixedGradient discretised as zeroGradient -- an adiabatic wall), and the
# face-diffusivity laplacian variant that B5's first attempt missed. Every one was found by inspection,
# after the fact. This compares the coefficients directly, so the next one is a number instead.
#
# One scalar field carries FOUR real BC types at once, and OF's own dump gives them four DISTINCT
# signatures -- so a comparison that accidentally tested nothing could not come out clean:
#
#     inlet    fixedValue      sum|IC| = 0.0004   sum|BC| = 0.3012
#     outlet   inletOutlet     sum|IC| = 0.1      sum|BC| = 0
#     gradWall fixedGradient   sum|IC| = 0        sum|BC| = 0.025   (= 40 faces * gamma*|Sf|*g)
#     zeroWall zeroGradient    sum|IC| = 0        sum|BC| = 0
#
# Measured: every patch agrees to MACHINE PRECISION (worst 1.3e-15), not to a tolerance -- these are the
# same closed-form expressions on both sides, so anything above rounding is a real discrepancy. Hence the
# 1e-10 threshold in bcoeff_compare: five orders above the noise and still far below any real defect.
#
# Both sides derive phi from the same U via fvc::flux(U) rather than reading a phi from disk, and brae
# builds its fields through the PRODUCTION factory (buildField -> makePatchField), so inletOutlet and
# fixedGradient take the path a real run takes. A harness that reconstructs the BCs itself tests the harness.
set -e
OFBASHRC=${OFBASHRC:-/usr/lib/openfoam/openfoam2412/etc/bashrc}
SRC=${SRC:-$(cd "$(dirname "$0")/bcoeffBox" && pwd)}
BUILD=${BUILD:-$(cd "$(dirname "$0")/../build" && pwd)}
WORK=${WORK:-/tmp/brae_bc_vs_of}
GAMMA=${GAMMA:-1e-3}

if [ ! -f "$OFBASHRC" ]; then echo "OpenFOAM not found at $OFBASHRC -- skipping"; exit 77; fi
source "$OFBASHRC" 2>/dev/null || true

# The OF side needs dumpScalarMatrix, which is built from tools/. Without it there is nothing to compare
# against, and silently passing would be the exact failure mode D5 was about.
DUMP="$(command -v dumpScalarMatrix || true)"
if [ -z "$DUMP" ] && [ -n "${FOAM_USER_APPBIN:-}" ] && [ -x "$FOAM_USER_APPBIN/dumpScalarMatrix" ]; then
    DUMP="$FOAM_USER_APPBIN/dumpScalarMatrix"
fi
if [ -z "$DUMP" ]; then
    echo "dumpScalarMatrix not built -- build it with: (cd tools/dumpScalarMatrix && wmake)"
    echo "skipping rather than passing without an oracle"
    exit 77
fi

rm -rf "$WORK"
cp -r "$SRC" "$WORK"
mkdir -p "$WORK/0" && cp "$WORK"/0.orig/* "$WORK/0/"
( cd "$WORK" && blockMesh > log.blockMesh 2>&1 )

# OF assembles fvm::div(phi,psi) - fvm::laplacian(gamma,psi) and writes ICoeff/BCoeff per patch.
"$DUMP" -case "$WORK" -field psi -gamma "$GAMMA" > "$WORK/log.dump" 2>&1
if [ ! -f "$WORK/0/ICoeff" ] || [ ! -f "$WORK/0/BCoeff" ]; then
    echo "  FAIL dumpScalarMatrix wrote no ICoeff/BCoeff -- no oracle to compare against"
    tail -20 "$WORK/log.dump"
    exit 1
fi

# Sanity: the four patches must carry DIFFERENT coefficients, else "they agree" is vacuous.
if ! grep -q "fixedGradient" "$WORK/log.dump" || ! grep -q "inletOutlet" "$WORK/log.dump"; then
    echo "  FAIL the case stopped exercising inletOutlet/fixedGradient -- OF reported:"
    grep "patch " "$WORK/log.dump" || true
    exit 1
fi
grep "patch " "$WORK/log.dump" | sed 's/^/  OF: /'

"$BUILD/bcoeff_compare" "$WORK" 0 "$GAMMA"
