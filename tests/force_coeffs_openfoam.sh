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
if [ ! -d "$CASE/constant/polyMesh" ] || [ ! -f "$CASE/$TIME/U" ] || [ ! -f "$CASE/$TIME/p" ] || [ ! -f "$COEFF" ]; then
    echo "SKIP: retained OpenFOAM case/fields/coefficient file is not available"
    exit 125
fi

exec "$BIN" "$CASE" "$TIME" "$COEFF"
