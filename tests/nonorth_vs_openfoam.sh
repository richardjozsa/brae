#!/usr/bin/env bash
# NON-ORTHOGONAL CORRECTION vs REAL OPENFOAM.
#
# The correction is invisible on a near-orthogonal mesh -- on pitzDaily every brae path agrees to 4 digits
# whether or not it is applied, so a gate there would pass with the term deleted. shearedChannel is
# genuinely non-orthogonal AND uses only schemes the rebuilt path implements (`bounded Gauss upwind`,
# `Gauss linear corrected`, laminar, steady), so it isolates this one term.
#
# The oracle is generated HERE by running real simpleFoam, not checked in: the point is agreement with
# OpenFOAM, and a stored reference cannot be re-derived if the case changes.
#
# The control is the whole test. Without the correction the same solver is 8.5e-02 on U; with it, 6.9e-04.
# Asserting only the second number would pass on a mesh where the term does not matter.
set -u
SRC="${1:?case dir}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DIAG="${DIAG_BIN:-$ROOT/build/diag_simple_loop}"
OFBIN=/usr/lib/openfoam/openfoam2412/platforms/linuxARM64GccDPInt32Opt
[ -x "$DIAG" ]            || { echo "SKIP: no diag_simple_loop at $DIAG"; exit 77; }
[ -x "$OFBIN/bin/simpleFoam" ] || { echo "SKIP: real OpenFOAM not available"; exit 77; }
SRC="$(cd "$SRC" && pwd)"

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
cp -r "$SRC" "$W/of"
# OpenFOAM resolves etc/controlDict through WM_PROJECT_DIR; without it simpleFoam aborts with
# "Could not find mandatory etc entry 'controlDict'" before reading the case at all.
export WM_PROJECT_DIR=/usr/lib/openfoam/openfoam2412
export FOAM_ETC="$WM_PROJECT_DIR/etc"
export PATH="$OFBIN/bin:$PATH"
export LD_LIBRARY_PATH="$OFBIN/lib:$OFBIN/lib/dummy:${LD_LIBRARY_PATH:-}"
( cd "$W/of" && timeout 600 simpleFoam > log.of 2>&1 ) || { echo "FAIL: OpenFOAM did not run"; tail -3 "$W/of/log.of"; exit 1; }
grep -q converged "$W/of/log.of" || { echo "FAIL: OpenFOAM did not converge"; exit 1; }
OFT=$(ls -d "$W/of"/[0-9]* | grep -vE '/0$' | sort -t/ -k99 -n | tail -1)
echo "  ok:   OpenFOAM reference generated -- $(grep converged "$W/of/log.of" | head -1)"

OUT=$("$DIAG" "$SRC" 0 500 "$OFT" 2>/dev/null | tail -3)
echo "$OUT" | sed 's/^/  /'
# The U figure is the field after the literal "U" on each line -- indexed by the marker, not by a column
# number, so a change in the label's wording cannot silently shift which number is read.
CORR=$(echo "$OUT" | awk '/WITH non-orth/{for(i=1;i<=NF;i++) if($i=="U"){print $(i+1); exit}}')
UNCO=$(echo "$OUT" | awk '/WITHOUT the correction/{for(i=1;i<=NF;i++) if($i=="U"){print $(i+1); exit}}')
[ -n "$CORR" ] && [ -n "$UNCO" ] || { echo "FAIL: could not parse the comparison"; exit 1; }

python3 - "$CORR" "$UNCO" <<'PY'
import sys
corr, unco = float(sys.argv[1]), float(sys.argv[2])
fails = 0
# The correction must bring U close to OpenFOAM. 5e-3 is an order above the measured 6.9e-04.
if corr <= 5e-3: print("  ok:   corrected U error %.3e <= 5e-03" % corr)
else:            print("  FAIL: corrected U error %.3e > 5e-03" % corr); fails += 1
# CONTROL: and the mesh must be non-orthogonal enough that omitting it is clearly worse. Without this the
# test would pass on a mesh where the term does nothing -- i.e. it would not be testing the term.
if unco >= 20*corr: print("  ok:   uncorrected is %.0fx worse (%.3e) -- the case discriminates (control)" % (unco/corr, unco))
else:               print("  FAIL: uncorrected only %.1fx worse -- this case cannot test the correction" % (unco/max(corr,1e-30))); fails += 1
print("PASS" if not fails else "FAIL"); sys.exit(1 if fails else 0)
PY
