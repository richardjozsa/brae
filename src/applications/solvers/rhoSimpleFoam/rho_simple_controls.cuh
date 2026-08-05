#pragma once
// rho_simple_controls.cuh -- the controls rhoSimpleFoam needs on top of the incompressible ones.
//
// Deliberately small. Everything the steady incompressible solver already reads (relaxation factors,
// residualControl, schemes, linear-solver selection) is unchanged and stays in SolverControls; this holds
// only what has no incompressible meaning.
//
// The compressible loop composes the SAME phases the steady and PIMPLE solvers use
// (solveMomentumPredictor -> correctPressureVelocity -> correctTurbulence) with two inserted steps:
//
//     UEqn  ->  EEqn  ->  pEqn  ->  thermo.correct()  ->  rho.relax()  ->  turbulence
//
// which is OF's rhoSimpleFoam.C order. The energy equation sits between momentum and pressure because
// the pressure equation needs the density that the new temperature implies, not the previous one.

#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "thermo_types.cuh"

namespace brae {

struct RhoSimpleControls
{
    // Refused at start-up rather than silently ignored: a transonic case run down the subsonic branch
    // converges to a wrong answer without complaining. Phase 5 lifts this.
    bool transonic = false;

    // OF pressureControl: the cell whose pressure is pinned when the domain is closed. Unused when a
    // boundary fixes p, which is the common external-aero case.
    int pRefCell = 0;
    scalar pRefValue = 0.0;

    // OF pressureControl::limit(p) -- p is clamped to [pMin, pMax] AFTER the pressure solve and before
    // rho is recomputed (rhoSimpleFoam/pEqn.H:90). This is a real stabiliser, not a safety net: the
    // stock aerofoilNACA0012 tutorial runs at Mach 0.72 and ships pMinFactor/pMaxFactor precisely
    // because the pressure iterate overshoots early on. A case that specifies the factors and does not
    // get the clamp can diverge where OF converges.
    //
    // OF accepts either absolute pMax/pMin or pMaxFactor/pMinFactor, the latter scaled by a reference
    // pressure taken from the p boundary values (pressureControl.C). Both are read here.
    bool   limitMaxP = false;
    bool   limitMinP = false;
    scalar pMaxLimit = 0.0;
    scalar pMinLimit = 0.0;
    scalar pMaxFactor = 0.0;
    scalar pMinFactor = 0.0;

    // Energy equation relaxation. OF looks the factor up by the FIELD NAME, which is "h" for
    // sensibleEnthalpy and "e" for sensibleInternalEnergy -- so an "e" case that only lists "e" would be
    // read as unrelaxed if we always asked for "h".
    scalar relaxHe = 0.7;
};

inline RhoSimpleControls readRhoSimpleControls(
    const FoamDict& fvSolution,
    bool internalEnergy = false)
{
    RhoSimpleControls rc;

    const FoamDict* simple = fvSolution.subDict("SIMPLE");
    if (simple)
    {
        rc.transonic = (simple->wordOr("transonic", "no") == "yes");
        rc.pRefCell = simple->intOr("pRefCell", rc.pRefCell);
        rc.pRefValue = simple->scalarOr("pRefValue", rc.pRefValue);

        // Absolute limits win over the factors, matching OF's precedence (pMax/pMin read first).
        if (simple->found("pMax")) { rc.pMaxLimit = simple->scalarOr("pMax", 0.0); rc.limitMaxP = true; }
        else if (simple->found("pMaxFactor")) rc.pMaxFactor = simple->scalarOr("pMaxFactor", 0.0);
        if (simple->found("pMin")) { rc.pMinLimit = simple->scalarOr("pMin", 0.0); rc.limitMinP = true; }
        else if (simple->found("pMinFactor")) rc.pMinFactor = simple->scalarOr("pMinFactor", 0.0);
    }

    const FoamDict* relax = fvSolution.subDict("relaxationFactors");
    if (relax)
    {
        const FoamDict* eqns = relax->subDict("equations");
        // Try the case's own energy-field name first, then the other, so a case listing only one is
        // honoured either way rather than silently left at the default.
        const char* primary = internalEnergy ? "e" : "h";
        const char* fallback = internalEnergy ? "h" : "e";
        if (eqns) rc.relaxHe = eqns->scalarOr(primary, eqns->scalarOr(fallback, rc.relaxHe));
    }

    return rc;
}

} // namespace brae
