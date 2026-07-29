#!/usr/bin/env bash
# The transient solver's refusals: physics the STEADY driver applies and this one does not yet must stop the run,
# not be silently skipped (gpuPimpleFoam.cu, "Physics the STEADY driver applies"). A dropped MRF rotation or
# fvOptions source produces a plausible-looking field that is simply wrong, which is the failure mode brae exists
# to avoid -- so each refusal gets a test, and the control case proves the guards do not fire on a clean case.
#
# The guards run after the dictionary parse and BEFORE the mesh is read, so these fixtures are dicts only: no
# polyMesh, no GPU. The control case is expected to fail too -- but on the missing mesh, further in.
set -u
BIN="${1:?brae_pimpleFoam binary}"
WORK="${2:?work dir}"

fail=0
mkcase()   # mkcase <dir> [extra controlDict lines]
{
    mkdir -p "$1/system" "$1/constant"
    printf 'FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\napplication pimpleFoam;\nstartFrom startTime;\nstartTime 0;\nstopAt endTime;\nendTime 1;\ndeltaT 0.1;\n%s\n' "${2:-}" > "$1/system/controlDict"
    printf 'FoamFile { version 2.0; format ascii; class dictionary; object fvSchemes; }\nddtSchemes { default Euler; }\ngradSchemes { default Gauss linear; }\ndivSchemes { default none; div(phi,U) bounded Gauss upwind; }\nlaplacianSchemes { default Gauss linear corrected; }\n' > "$1/system/fvSchemes"
    printf 'FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\nsolvers { p { solver PCG; tolerance 1e-7; } U { solver PBiCGStab; tolerance 1e-8; } }\nPIMPLE { nOuterCorrectors 2; nCorrectors 2; nNonOrthogonalCorrectors 0; }\n' > "$1/system/fvSolution"
    printf 'FoamFile { version 2.0; format ascii; class dictionary; object transportProperties; }\ntransportModel Newtonian;\nnu 1e-05;\n' > "$1/constant/transportProperties"
    printf 'FoamFile { version 2.0; format ascii; class dictionary; object turbulenceProperties; }\nsimulationType laminar;\n' > "$1/constant/turbulenceProperties"
}
refuses()  # refuses <name> <case dir> <expected text>
{
    local log="$WORK/$1.log"
    "$BIN" -case "$2" > "$log" 2>&1
    local rc=$?
    if [ "$rc" -eq 0 ]; then
        echo "FAIL: $1 -- the solver accepted a case it cannot solve correctly"; fail=1; return
    fi
    if grep -qF -e "$3" "$log"; then
        echo "ok:   $1"
    else
        echo "FAIL: $1 -- refused, but not for the stated reason (wanted '$3')"; sed -n '1,8p' "$log"; fail=1
    fi
}

rm -rf "$WORK"; mkdir -p "$WORK"

# 1. adjustTimeStep: running at the fixed deltaT instead is a DIFFERENT simulation, not an approximation of the
#    requested one, so it cannot be quietly ignored.
mkcase "$WORK/adjustdt" "adjustTimeStep yes;
maxCo 0.9;"
refuses adjust_time_step "$WORK/adjustdt" "adjustTimeStep yes is not supported yet"

# 2. An active MRF zone: dropping the rotation leaves a converged-looking field of the wrong flow.
mkcase "$WORK/mrf"
printf 'FoamFile { version 2.0; format ascii; class dictionary; object MRFProperties; }\nMRF1\n{\n    cellZone rotor;\n    active yes;\n    origin (0 0 0);\n    axis (0 0 1);\n    omega 104.72;\n}\n' > "$WORK/mrf/constant/MRFProperties"
refuses active_mrf_zone "$WORK/mrf" "does not apply MRF yet"

# 3. fvOptions in either of the two places OpenFOAM looks.
mkcase "$WORK/fvopt_system"
printf 'FoamFile { version 2.0; format ascii; class dictionary; object fvOptions; }\nmomentumSource { type meanVelocityForce; }\n' > "$WORK/fvopt_system/system/fvOptions"
refuses fvoptions_in_system "$WORK/fvopt_system" "does not apply fvOptions yet"

mkcase "$WORK/fvopt_constant"
printf 'FoamFile { version 2.0; format ascii; class dictionary; object fvOptions; }\nporosity { type explicitPorositySource; }\n' > "$WORK/fvopt_constant/constant/fvOptions"
refuses fvoptions_in_constant "$WORK/fvopt_constant" "does not apply fvOptions yet"

# 4. An INACTIVE MRF zone is not a refusal -- OpenFOAM cases routinely ship one switched off, and refusing those
#    would be the mirror-image bug (stopping a case brae can solve perfectly well).
mkcase "$WORK/mrf_off"
printf 'FoamFile { version 2.0; format ascii; class dictionary; object MRFProperties; }\nMRF1\n{\n    cellZone rotor;\n    active no;\n    omega 104.72;\n}\n' > "$WORK/mrf_off/constant/MRFProperties"
"$BIN" -case "$WORK/mrf_off" > "$WORK/inactive_mrf.log" 2>&1
if grep -qF -e "does not apply MRF yet" "$WORK/inactive_mrf.log"; then
    echo "FAIL: inactive_mrf -- a switched-off MRF zone was treated as active"; fail=1
else
    echo "ok:   inactive_mrf"
fi

# 5. Control: a clean case must get PAST every guard. It still fails -- on the mesh these fixtures do not have --
#    which is exactly how we know the guards let it through rather than the run being stopped early.
mkcase "$WORK/clean"
"$BIN" -case "$WORK/clean" > "$WORK/clean.log" 2>&1
if grep -qE -e "does not apply (MRF|fvOptions) yet|adjustTimeStep yes is not supported" "$WORK/clean.log"; then
    echo "FAIL: clean_case -- a guard fired on a case with none of the unsupported features"
    sed -n '1,8p' "$WORK/clean.log"; fail=1
else
    echo "ok:   clean_case"
fi

# 6. --help is part of the CLI surface: it must work without a case and exit 0.
if "$BIN" --help > "$WORK/help.log" 2>&1 && grep -qF "brae_pimpleFoam" "$WORK/help.log"; then
    echo "ok:   help"
else
    echo "FAIL: help -- --help did not print usage and exit 0"; sed -n '1,5p' "$WORK/help.log"; fail=1
fi

[ "$fail" -eq 0 ] && echo "PASS: the transient solver refuses what it cannot solve, and only that"
exit "$fail"
