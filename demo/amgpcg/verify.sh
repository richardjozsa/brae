#!/usr/bin/env bash
# Verification gate for the AMG/PCG fixes applied 2026-07-23 (see BUGS.md).
#
# Everything in BUGS.md is compile-checked but NOT GPU-verified, because the fixes were written
# with the machine's GPU off. Run this the moment the GPU is back, BEFORE trusting any fix. It
# gates each applied change with a concrete pass/fail:
#
#   1. regression   ctest -R "gpu_amg|gamg|pcg"     was 14/14 before the edits; catches any
#                                                    regression in F3/F5/D2 (all on the default path)
#   2. chebyshev    BRAE_CHEBYSHEV converges at 1M   the fix's CORRECTNESS, not just non-regression:
#                                                    was 5000 iters (stalled 1.6e-3), expect ~25-40
#   3. d2-capture   PCG graph captures ONCE          D2's new key comparison must not trigger
#                                                    spurious re-captures when controls are constant
#
# Usage:  ./demo/amgpcg/verify.sh            # all three gates
#         ./demo/amgpcg/verify.sh chebyshev  # one gate by name
#
# Exit code is the number of FAILED gates (0 = all pass).

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD="$ROOT/build"
DEMO="$ROOT/demo/amgpcg"
WHICH="${1:-all}"

pass=0
fail=0

say()  { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
ok()   { printf '  \033[32mPASS\033[0m  %s\n' "$1"; pass=$((pass+1)); }
bad()  { printf '  \033[31mFAIL\033[0m  %s\n' "$1"; fail=$((fail+1)); }
skip() { printf '  \033[33mSKIP\033[0m  %s\n' "$1"; }

# --------------------------------------------------------------------------------------------- #

need_gpu()
{
    if ! nvidia-smi >/dev/null 2>&1; then
        echo "no GPU visible (nvidia-smi failed) -- this gate needs the GPU powered on."
        return 1
    fi
    return 0
}

build_targets()
{
    say "build"
    if ! cmake --build "$BUILD" --target brae trace_brae -j 8 >/tmp/verify_build.log 2>&1; then
        bad "build (brae, trace_brae) -- see /tmp/verify_build.log"
        return 1
    fi
    ok "build (brae, trace_brae)"

    # The 1M case must exist for the Chebyshev gate; rebuild its mesh if missing.
    if [ ! -f "$DEMO/case1M/constant/polyMesh/owner" ]; then
        skip "case1M mesh absent -- run: (cd demo/amgpcg/case1M && blockMesh)"
    fi
    return 0
}

# --------------------------------------------------------------------------------------------- #

gate_regression()
{
    say "1. regression gate  (ctest gpu_amg|gamg|pcg)"
    need_gpu || { skip "regression (no GPU)"; return; }

    local out
    out="$(cd "$BUILD" && ctest -R 'gpu_amg|gamg|pcg' 2>&1)"
    echo "$out" | tail -3 | sed 's/^/    /'

    if echo "$out" | grep -q "100% tests passed"; then
        ok "regression: all AMG/PCG tests pass"
    else
        bad "regression: a test failed -- a fix regressed the default path, stop and investigate"
    fi
}

gate_chebyshev()
{
    say "2. chebyshev convergence  (BRAE_CHEBYSHEV at 1M)"
    need_gpu || { skip "chebyshev (no GPU)"; return; }

    if [ ! -f "$DEMO/case1M/constant/polyMesh/owner" ]; then
        skip "chebyshev: case1M mesh absent"
        return
    fi

    # The bug was DIVERGENCE: 5000 iters (= maxIter) stalled at 1.6e-3, i.e. it never converged.
    # The fix's job is to make it converge, not to make it fast -- Chebyshev on UNSMOOTHED
    # aggregation still tracks the hierarchy depth (~80 iters at 1M is expected; only BRAE_AMG_SA
    # fixes that). So pass = converged (final < 1e-8) in well under maxIter, not a tight count.
    local out iters final
    out="$(BRAE_CHEBYSHEV=1 "$BUILD/trace_brae" "$DEMO/case1M" 0 2>&1)"
    iters="$(echo "$out" | grep -oE 'iters=[0-9]+' | head -1 | cut -d= -f2)"
    final="$(echo "$out" | grep -oE 'final=[0-9.eE+-]+' | head -1 | cut -d= -f2)"
    echo "$out" | grep -E 'MEASURE|iters=' | sed 's/^/    /'

    if [ -z "$iters" ]; then
        bad "chebyshev: no iteration count parsed (did the solve crash?)"
    elif [ "$iters" -lt 500 ] && awk "BEGIN{exit !(${final:-1} < 1e-8)}"; then
        ok "chebyshev CONVERGES at 1M in $iters iters (final $final; was 5000 stalled at 1.6e-3)"
    else
        bad "chebyshev not converging at 1M ($iters iters, final $final) -- try Gershgorin bound or raise CHEB_UPPER_SAFETY"
    fi
}

gate_d2_capture()
{
    say "3. D2 device-path correctness  (graph PCG must match the host loop)"
    need_gpu || { skip "d2-capture (no GPU)"; return; }

    # case20 is a T-Laplacian case for the dump tools, NOT a simpleFoam case -- brae cannot run it.
    # D2 is exercised by a real simpleFoam solve: the device WHILE-graph PCG (BRAE_PCG_DEVICE=1,
    # which uses the graph cache whose key D2 extended) must converge in the same iteration count
    # and to the same field as the host loop (BRAE_PCG_DEVICE=0), which is documented bit-identical.
    local src="$ROOT/validation/matrixDumpGAMG"
    if [ ! -d "$src/282" ] && [ ! -d "$src/0" ]; then
        skip "d2-capture: validation/matrixDumpGAMG not found"
        return
    fi

    local work="/tmp/verify_d2"
    rm -rf "$work"; cp -r "$src" "$work"
    [ -d "$work/282" ] && mv "$work/282" "$work/0"
    sed -i 's/^startFrom.*/startFrom       startTime;/; s/^startTime.*/startTime       0;/; s/^endTime.*/endTime         30;/; s/^writeInterval.*/writeInterval   1000;/' "$work/system/controlDict"

    local on off itOn itOff
    on="$(cd "$work" && BRAE_PCG_DEVICE=1 "$BUILD/brae" 2>&1)"
    off="$(cd "$work" && BRAE_PCG_DEVICE=0 "$BUILD/brae" 2>&1)"
    itOn="$(echo "$on"  | grep -oE 'converged in [0-9]+' | grep -oE '[0-9]+')"
    itOff="$(echo "$off" | grep -oE 'converged in [0-9]+' | grep -oE '[0-9]+')"

    if [ -n "$itOn" ] && [ "$itOn" = "$itOff" ]; then
        ok "d2-capture: device and host both converge in $itOn iters (graph key correct)"
    else
        bad "d2-capture: device=$itOn host=$itOff iters -- D2 key logic changed the graph path"
    fi
}

# --------------------------------------------------------------------------------------------- #

build_targets || { echo; echo "build failed -- fix compile errors before gating."; exit 99; }

case "$WHICH" in
    all)        gate_regression; gate_chebyshev; gate_d2_capture ;;
    regression) gate_regression ;;
    chebyshev)  gate_chebyshev ;;
    d2*|capture) gate_d2_capture ;;
    *) echo "unknown gate '$WHICH' (use: all | regression | chebyshev | d2-capture)"; exit 2 ;;
esac

say "summary"
printf '  %d passed, %d failed\n' "$pass" "$fail"
exit "$fail"
