#!/usr/bin/env bash
# PIMPLE/momentumPredictor -- OF pimpleControl's switch, default true:
#
#   #include "UEqn.H"                       // ALWAYS assembled and relaxed: rAU and HbyA come from it
#   if (pimple.momentumPredictor()) { solve(UEqn == -fvc::grad(p)); }
#
# so `off` does not remove the momentum equation, it removes only its SOLVE -- U is then updated by the
# pressure corrector alone. laminar/planarPoiseuille turns it off, and solving anyway made brae's first
# step 56% fast (0.00535 against OpenFOAM's 0.00343 m/s) and the 20-step field 1.2e-01 out.
#
# THIS TEST EXISTS BECAUSE THE OBVIOUS TEST DOES NOT WORK. The first version of the fix read the entry
# into the controls AFTER the solver had been constructed -- and the solver takes a COPY. The run printed
# "momentumPredictor off" and then solved the predictor anyway. A test that checked the log line, or the
# parsed flag, would have passed on that. So this one runs the same case twice and requires the two U
# fields to DIFFER: the only evidence that the switch reached the solve.
#
# Skips (exit 125) if the fixture is absent, like pimple_run.
set -u
BIN="${1:?brae_pimpleFoam binary}"
SRC="${2:?committed pitzDaily case dir}"
WORK="${3:?work dir}"

if [ ! -f "$SRC/constant/polyMesh/points" ] || [ ! -f "$SRC/0/U" ]; then
    echo "SKIP: fixture '$SRC' not present"; exit 125
fi

fail=0
run_one()   # run_one <tag> <momentumPredictor entry, or empty for the default>
{
    local tag="$1" entry="${2-}" d="$WORK/$1"
    rm -rf "$d"; mkdir -p "$d"
    cp -r "$SRC/constant" "$SRC/system" "$SRC/0" "$d/"
    cat > "$d/system/controlDict" <<'EOF'
FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }
application     pimpleFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         6e-5;
deltaT          2e-5;
writeControl    timeStep;
writeInterval   3;
writePrecision  12;
EOF
    # the committed fixture is STEADY: make it transient, exactly as pimple_run.sh does
    sed -i 's/^ddtSchemes.*/ddtSchemes { default Euler; }/' "$d/system/fvSchemes" 2>/dev/null || true
    python3 - "$d/system/fvSchemes" <<'PYX'
import re, sys
p = sys.argv[1]; s = open(p).read()
s = re.sub(r'ddtSchemes\s*\{[^}]*\}', 'ddtSchemes { default Euler; }', s, flags=re.S)
open(p, 'w').write(s)
PYX
    cat > "$d/system/fvSolution" <<EOF
FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }
solvers
{
    p { solver PCG; preconditioner DIC; tolerance 1e-8; relTol 0; }
    "(U|k|epsilon)" { solver PBiCGStab; preconditioner DILU; tolerance 1e-9; relTol 0; }
}
PIMPLE { nOuterCorrectors 2; nCorrectors 2; nNonOrthogonalCorrectors 0; $entry }
relaxationFactors { equations { ".*" 1; } }
EOF
    ( cd "$d" && "$BIN" > log 2>&1 ) || { echo "FAIL [$tag]: run failed"; tail -3 "$d/log"; fail=1; return 1; }
    [ -f "$d/6e-05/U" ] || { echo "FAIL [$tag]: no output written"; fail=1; return 1; }
    return 0
}

# The same case three ways: default, explicitly on, explicitly off.
run_one default ""                        || true
run_one on      "momentumPredictor yes;"  || true
run_one off     "momentumPredictor no;"   || true

cmp_fields()   # cmp_fields <a> <b> -> prints the max |difference| over the internal field
{
    python3 - "$1" "$2" <<'PY'
import re, sys
def read(fn):
    b = open(fn).read()
    m = re.search(r'internalField\s+nonuniform\s+List<vector>\s*\n(\d+)\s*\n\(', b)
    if not m: return None
    n = int(m.group(1)); st = m.end(); en = b.index('\n)', st)
    return [[float(x) for x in t.split()] for t in re.findall(r'\(([^)]*)\)', b[st:en])]
a, c = read(sys.argv[1]), read(sys.argv[2])
if a is None or c is None or len(a) != len(c):
    print("nan"); raise SystemExit
print(max(max(abs(p-q) for p, q in zip(u, v)) for u, v in zip(a, c)))
PY
}

if [ $fail -eq 0 ]; then
    same=$(cmp_fields "$WORK/default/6e-05/U" "$WORK/on/6e-05/U")
    diff=$(cmp_fields "$WORK/default/6e-05/U" "$WORK/off/6e-05/U")
    echo "  default vs explicitly-on : max|dU| = $same"
    echo "  default vs off           : max|dU| = $diff"

    # 1. The default IS on: writing it explicitly must change nothing BEYOND brae's own run-to-run
    #    noise (GPU reduction order; measured ~1e-10 on this case, against a field of order 10).
    #    Bit-identity is not available here and asking for it would test the hardware, not the switch.
    if [ "$(python3 -c "print(1 if float('$same') < 1e-6 else 0)")" != "1" ]; then
        echo "FAIL: 'momentumPredictor yes' differs from the default, so the default is not 'on'"; fail=1
    fi
    # 2. ...and off must reach the SOLVE, not just the log. Identical fields mean the flag was parsed
    #    into a copy nobody reads -- the exact bug this test was written for.
    #    The threshold is four orders above the noise floor leg 1 just measured, so "different" here
    #    cannot be the same nondeterminism read twice.
    if [ "$(python3 -c "print(1 if float('$diff') > 1e4*max(float('$same'), 1e-14) else 0)")" != "1" ]; then
        echo "FAIL: 'momentumPredictor no' produced the SAME field as 'on' -- the switch never reached the solve"; fail=1
    fi
    # 3. ...and it must still be a sane run, not a broken one: no NaN in the written field.
    if grep -qi "nan" "$WORK/off/6e-05/U"; then
        echo "FAIL: the momentumPredictor-off run wrote NaN"; fail=1
    fi
fi

if [ $fail -eq 0 ]; then echo "momentum_predictor: PASSED"; else echo "momentum_predictor: FAILED"; fi
exit $fail
