#!/usr/bin/env bash
# Fresh extraction wrapper for the no-solve retained OpenFOAM yPlus comparison.
set -eu

BIN="${1:?comparison binary}"
ARCHIVE37="${2:-}"
ARCHIVE42="${3:-}"
if [ -z "$ARCHIVE37" ] || [ -z "$ARCHIVE42" ] || [ ! -f "$ARCHIVE37" ] || [ ! -f "$ARCHIVE42" ]; then
    echo "SKIP: retained yPlus archive variable is empty or archive is absent"
    exit 125
fi

WORK="$(mktemp -d "${TMPDIR:-/tmp}/brae-yplus-retained.XXXXXX")"
trap 'rm -rf "$WORK"' EXIT
mkdir "$WORK/issue37" "$WORK/issue42"
tar -xf "$ARCHIVE37" -C "$WORK/issue37"
tar -xf "$ARCHIVE42" -C "$WORK/issue42"
printf 'fresh extraction from %s\n' "$ARCHIVE37" > "$WORK/issue37/.fresh-extraction-marker"
printf 'fresh extraction from %s\n' "$ARCHIVE42" > "$WORK/issue42/.fresh-extraction-marker"

"$BIN" "$WORK/issue37" "$WORK/issue42"
