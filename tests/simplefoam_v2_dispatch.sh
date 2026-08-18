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
s = re.sub(r'div\(phi,U\)[^;]*;', 'div(phi,U)      Gauss upwind;', s)
# laplacianSchemes is deliberately LEFT ALONE at `Gauss linear corrected`, which is what pitzDaily ships
# and what OpenFOAM defaults to. It used to be rewritten to `orthogonal` here because the rebuilt
# fvm::laplacian was orthogonal only; both halves of the correction are now ported on both paths, so the
# supported case exercises the flag end to end instead of dodging it.
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

# `bounded` is SUPPORTED: -fvm::Sp(fvc::div(phi),U) is implemented on both paths and matched to 2.9e-16.
# It used to be a refusal; assert it RUNS, since the term vanishes at convergence and a converged field
# comparison could not tell whether it was applied.
echo "== 4c. bounded div(phi,U) is supported =="
supported "$W/bnd"
python3 - "$W/bnd" <<'PYEOF'
import re, sys
d = sys.argv[1]
s = open(d + '/system/fvSchemes').read()
s = re.sub(r'div\(phi,U\)[^;]*;', 'div(phi,U)      bounded Gauss upwind;', s)
open(d + '/system/fvSchemes', 'w').write(s)
PYEOF
( cd "$W/bnd" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > log 2>&1 )
brc=$?
if [ $brc -eq 0 ]; then ok "bounded: exit 0"; else bad "bounded: exit $brc"; sed -n '1,4p' "$W/bnd/log"; fi
grep -q "is bounded" "$W/bnd/log" && ok "bounded: the solver states it is applying the term" \
                                  || bad "bounded: applied silently or not at all"

# `limited <coeff>` is limitedSnGrad, which caps the correction -- a DIFFERENT scheme from `corrected`,
# and the one place where accepting the whole `corrected`/`limited` family would silently over-correct.
try_refusal limitedlaplacian \
    "import re,sys;p=sys.argv[1]+'/system/fvSchemes';s=open(p).read();open(p,'w').write(re.sub(r'laplacianSchemes\s*\{[^}]*\}','laplacianSchemes\n{\n    default         Gauss linear limited 0.33;\n}',s))" \
    "limited"

try_refusal divscheme \
  "s=open(d+'/system/fvSchemes').read(); s=re.sub(r'div\(phi,U\)[^;]*;','div(phi,U)      bounded Gauss linearUpwind grad(U);',s); open(d+'/system/fvSchemes','w').write(s)" \
  "implicit weights only"

try_refusal transient \
  "s=open(d+'/system/fvSchemes').read(); s=re.sub(r'default\s+steadyState;','default         Euler;',s); open(d+'/system/fvSchemes','w').write(s)" \
  "steadyState"

# RAS/kEpsilon is SUPPORTED: the driver's turbulence hook is wired to the device k-epsilon. This used to
# be a refusal, and flipping it is the point of that work -- so assert it RUNS and writes the turbulence
# fields, not merely that it is accepted. A case that ran without writing k/epsilon/nut would mean the
# hook was never called.
echo "== 4b. RAS/kEpsilon is supported: the turbulence hook runs =="
supported "$W/ras"
python3 - "$W/ras" <<'PYEOF'
import re, sys
d = sys.argv[1]
s = open(d + '/constant/turbulenceProperties').read()
s = re.sub(r'simulationType\s+\S+;', 'simulationType  RAS;', s)
open(d + '/constant/turbulenceProperties', 'w').write(s)
PYEOF
( cd "$W/ras" && BRAE_SIMPLEFOAM_V2=1 "$BRAE" > log 2>&1 )
rasrc=$?
if [ $rasrc -eq 0 ]; then ok "RAS/kEpsilon: exit 0"; else bad "RAS/kEpsilon: exit $rasrc"; sed -n '1,4p' "$W/ras/log"; fi
for fld in k epsilon nut; do
    if [ -f "$W/ras/3/$fld" ]; then ok "RAS/kEpsilon: wrote $fld (the hook ran)"
    else bad "RAS/kEpsilon: no $fld written"; fi
done
# The turbulence must have CHANGED nut -- a hook that ran but did nothing would still write the file.
if [ -f "$W/ras/3/nut" ] && [ -f "$W/ras/0/nut" ]; then
    if cmp -s "$W/ras/3/nut" "$W/ras/0/nut"; then bad "RAS/kEpsilon: nut is unchanged (control)"
    else ok "RAS/kEpsilon: nut changed (control)"; fi
fi

echo "== 5. NEGATIVE CONTROL: the guard is not refusing everything =="
# The supported case above ran. If it had not, every "refused" line would be meaningless -- a guard that
# blocks unconditionally passes every refusal test.
[ -d "$W/on/3" ] && ok "the guard admits a supported case (control)" \
                 || bad "the guard refuses everything (control)"

echo
[ $fails -eq 0 ] && { echo "PASS"; exit 0; } || { echo "FAIL ($fails)"; exit 1; }
