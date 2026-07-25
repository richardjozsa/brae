#!/usr/bin/env bash
# Regression lock for the single-GPU hardening fail-loud guards. Each sub-case mutates a valid case to trigger one
# guard and asserts brae exits NON-ZERO with the expected message. Fails if any guard stops firing (so a future edit
# that silently removes a guard is caught here). All guards fire at SETUP (before solving), so this is fast.
#   argv: <brae binary> <base case dir> <work dir>
set -u
BRAE="$1"; BASE="$2"; WORK="$3"
fail=0
CASE="$WORK/case"

setup() { rm -rf "$CASE"; cp -r "$BASE" "$CASE"; rm -rf "$CASE"/[1-9]* 2>/dev/null
          sed -i 's/endTime .*/endTime 1;/' "$CASE/system/controlDict"; }

expect() {   # $1=label  $2=expected substring (must appear in output, and exit must be non-zero)
    local out rc
    out=$(CUDA_VISIBLE_DEVICES=0 "$BRAE" -case "$CASE" 2>&1); rc=$?
    if [ "$rc" -eq 0 ]; then echo "FAIL [$1]: expected non-zero exit, got 0"; fail=1; return; fi
    if ! printf '%s' "$out" | grep -qF -- "$2"; then
        echo "FAIL [$1]: expected message '$2' not found"; printf '%s\n' "$out" | tail -2; fail=1; return; fi
    echo "PASS [$1]"
}

mkdir -p "$WORK"

# #15 non-Newtonian transportModel -> Newtonian-only guard
setup; sed -i 's/transportModel .*/transportModel CrossPowerLaw;/' "$CASE/constant/transportProperties"
expect "non-Newtonian(#15)" "transportModel"

# #18 endTime < 1 (fractional truncation footgun)
setup; sed -i 's/endTime .*/endTime 0;/' "$CASE/system/controlDict"
expect "endTime<1(#18)" "< 1)"

# #17 #calc directive
setup; sed -i 's/endTime .*/endTime #calc "2*3";/' "$CASE/system/controlDict"
expect "calc-directive(#17)" "#calc"

# #10 no explicit div(phi,U)
setup; sed -i '/div(phi,U)/d' "$CASE/system/fvSchemes"
expect "no-div(phi,U)(#10)" "div(phi,U)"

# #9 internalField size mismatch
setup; sed -i 's/internalField.*/internalField nonuniform List<scalar> 2 (1 2);/' "$CASE/0/p"
expect "field-size(#9)" "size mismatch"

# #3 unsupported fvOptions source (scalarSemiImplicitSource is parsed but never applied)
setup; printf 's { type scalarSemiImplicitSource; active yes; scalarSemiImplicitSourceCoeffs { selectionMode all; injectionRateSuSp { k (1 0); } } }\n' > "$CASE/system/fvOptions"
expect "fvOptions-scalar(#3)" "cannot apply"

# #4 nutUSpaldingWallFunction on a kEpsilon (non-SA) case
setup; sed -i 's/nutkWallFunction/nutUSpaldingWallFunction/' "$CASE/0/nut"
expect "wallfn-Spalding(#4)" "nutUSpaldingWallFunction"

rm -rf "$CASE"
if [ "$fail" -ne 0 ]; then echo "HARDENING GUARDS: FAIL"; exit 1; fi
echo "HARDENING GUARDS: all guards fired"; exit 0
