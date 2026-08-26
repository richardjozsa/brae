#!/usr/bin/env bash
# Direct Brae-vs-OpenFOAM normalization gate. Every input is caller-supplied and read-only.
set -eu

BIN="${1:?test_force_coeffs binary}"
CASE="${BRAE_OF_RETAINED_CASE:-}"
TIME="${BRAE_OF_RETAINED_TIME:-}"
COEFF="${BRAE_OF_RETAINED_COEFFICIENT:-}"
if [ -z "$CASE" ] || [ -z "$TIME" ] || [ -z "$COEFF" ]; then
    echo "SKIP: set BRAE_OF_RETAINED_CASE, BRAE_OF_RETAINED_TIME, and BRAE_OF_RETAINED_COEFFICIENT for the direct OpenFOAM comparison"
    exit 125
fi
field_present() {
    [ -f "$CASE/$TIME/$1" ] || [ -f "$CASE/$TIME/$1.gz" ]
}
if [ ! -d "$CASE/constant/polyMesh" ] || [ ! -f "$COEFF" ]; then
    echo "SKIP: retained OpenFOAM case/fields/coefficient file is not available"
    exit 125
fi
for field in U p k nut; do
    if ! field_present "$field"; then
        echo "SKIP: retained OpenFOAM field '$field' is not available"
        exit 125
    fi
done

exec "$BIN" "$CASE" "$TIME" "$COEFF"
