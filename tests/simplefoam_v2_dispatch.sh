#!/usr/bin/env bash
# DISPATCH GATE for the rebuilt simpleFoam (BRAE_SIMPLEFOAM_V2).
#
# The rebuilt path covers a strict subset of what the existing solver runs, so the thing that has to be
# true is not "it produces the right answer on the cases it supports" -- the component tests cover that --
# but "it never runs a case it does not support". A path that quietly degrades is indistinguishable from a
# correct one in the output, and brae has already shipped a solver that ignored MRFProperties, converged,
# and said nothing.
#
# So this asserts, on real cases through the real binary:
#   1. off  -> nothing changes, the existing solver runs;
#   2. on + unsupported -> REFUSES, names the reason, exits non-zero;
#   3. on + supported   -> runs and writes;
#   4. a substitution brae makes on purpose (GAMG -> AMG-PCG) is ANNOUNCED, not hidden.
#
# usage: simplefoam_v2_dispatch.sh <pitzDailyCase>
set -u
SRC="${1:?case dir}"
BRAE="${BRAE_BIN:-$(cd "$(dirname "$0")/.." && pwd)/build/brae}"
[ -x "$BRAE" ] || { echo "SKIP: no brae binary at $BRAE"; exit 77; }
SRC="$(cd "$SRC" && pwd)"

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
fails=0
ok()   { echo "  ok:   $1"; }
bad()  { echo "  FAIL: $1"; fails=$((fails+1)); }

mkcase() {   # mkcase <dir> ; copies SRC and strips time dirs
    cp -r "$SRC" "$1"
    find "$1" -mindepth 1 -maxdepth 1 -type d ! -name 0 ! -name constant ! -name system -exec rm -rf {} + 2>/dev/null
    rm -rf "$1/postProcessing"
    python3 - "$1/system/controlDict" <<'PY'
import re, sys
p = sys.argv[1]; s = open(p).read()
s = re.sub(r'^endTime\s+\S+;', 'endTime         3;', s, flags=re.M)
open(p, 'w').write(s)
PY
}

# Make a case the rebuilt path SUPPORTS: laminar, upwind, plain SIMPLE.
supported() {
    mkcase "$1"
    python3 - "$1" <<'PY'
import re, sys, os
d = sys.argv[1]
s = open(d + '/system/fvSchemes').read()
s = re.sub(r'div\(phi,U\)[^;]*;', 'div(phi,U)      bounded Gauss upwind;', s)
open(d + '/system/fvSchemes', 'w').write(s)
s = open(d + '/system/fvSolution').read()
s = re.sub(r'consistent\s+\S+;', 'consistent      no;', s)
open(d + '/system/fvSolution', 'w').write(s)
s = open(d + '/constant/turbulenceProperties').read()
s = re.sub(r'simulationType\s+\S+;', 'simulationType  laminar;', s)
open(d + '/constant/turbulenceProperties', 'w').write(s)
PY
}

echo "== 1. NOT selected: the existing solver runs =="
mkcase "$W/off"
( cd "$W/off" && "$BRAE" > log 2>&1 )
if [ -d "$W/off/3" ]; then ok "off: the case ran and wrote a time directory"; else bad "off: no output"; fi
if grep -q "simpleFoam v2" "$W/off/log"; then bad "off: the v2 path announced itself"; else ok "off: v2 stayed silent"; fi

echo "== 2. selected + supported: the rebuilt path runs =="
supported "$W/on"
( cd "$W/on" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > log 2>&1 )
rc=$?
[ $rc -eq 0 ] && ok "supported: exit 0" || bad "supported: exit $rc"
[ -d "$W/on/3" ] && ok "supported: wrote a time directory" || bad "supported: no output"
grep -q "Time = 3" "$W/on/log" && ok "supported: ran the requested iterations" \
                                || bad "supported: did not reach the end time"

echo "== 3. the GAMG substitution is ANNOUNCED, not hidden =="
if grep -q "NOTICE (simpleFoam v2).*GAMG" "$W/on/log"; then
    ok "GAMG -> AMG-PCG substitution is stated"
else
    bad "GAMG substitution was silent"
fi

echo "== 4. selected + unsupported: REFUSES, with the reason =="
# Each of these is a component the rebuilt path does not implement, and each would otherwise produce a
# converged, plausible, wrong answer.
try_refusal() {   # try_refusal <label> <mutator-python> <expected-substring>
    local d="$W/ref_$1"
    supported "$d"
    python3 - "$d" <<PY
import re, sys, os
d = sys.argv[1]
$2
PY
    ( cd "$d" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > log 2>&1 )
    local rc=$?
    if [ $rc -eq 0 ]; then bad "$1: ran anyway (exit 0)"; return; fi
    if grep -q "$3" "$d/log"; then ok "$1: refused, and named it"; else
        bad "$1: refused but did not name the reason ($3)"; sed -n '1,4p' "$d/log"; fi
}

try_refusal mrf \
  "open(d+'/constant/MRFProperties','w').write('// test\n')" \
  "MRFProperties"

try_refusal fvoptions \
  "open(d+'/constant/fvOptions','w').write('// test\n')" \
  "fvOptions"

try_refusal simplec \
  "s=open(d+'/system/fvSolution').read(); s=re.sub(r'consistent\s+\S+;','consistent      yes;',s); open(d+'/system/fvSolution','w').write(s)" \
  "SIMPLEC"

try_refusal divscheme \
  "s=open(d+'/system/fvSchemes').read(); s=re.sub(r'div\(phi,U\)[^;]*;','div(phi,U)      bounded Gauss linearUpwind grad(U);',s); open(d+'/system/fvSchemes','w').write(s)" \
  "implicit weights only"

try_refusal transient \
  "s=open(d+'/system/fvSchemes').read(); s=re.sub(r'default\s+steadyState;','default         Euler;',s); open(d+'/system/fvSchemes','w').write(s)" \
  "steadyState"

try_refusal turbulence \
  "s=open(d+'/constant/turbulenceProperties').read(); s=re.sub(r'simulationType\s+\S+;','simulationType  RAS;',s); open(d+'/constant/turbulenceProperties','w').write(s)" \
  "RAS"

echo "== 5. NEGATIVE CONTROL: the guard is not refusing everything =="
# The supported case above ran. If it had not, every "refused" line would be meaningless -- a guard that
# blocks unconditionally passes every refusal test.
[ -d "$W/on/3" ] && ok "the guard admits a supported case (control)" \
                 || bad "the guard refuses everything (control)"

echo
[ $fails -eq 0 ] && { echo "PASS"; exit 0; } || { echo "FAIL ($fails)"; exit 1; }
