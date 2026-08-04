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

    // he equation relaxation, fvSolution relaxationFactors.equations.h (OF names it "h" even for
    // sensibleEnthalpy).
    scalar relaxHe = 0.7;
};

inline RhoSimpleControls readRhoSimpleControls(const FoamDict& fvSolution)
{
    RhoSimpleControls rc;

    const FoamDict* simple = fvSolution.subDict("SIMPLE");
    if (simple)
    {
        rc.transonic = (simple->wordOr("transonic", "no") == "yes");
        rc.pRefCell = simple->intOr("pRefCell", rc.pRefCell);
        rc.pRefValue = simple->scalarOr("pRefValue", rc.pRefValue);
    }

    const FoamDict* relax = fvSolution.subDict("relaxationFactors");
    if (relax)
    {
        const FoamDict* eqns = relax->subDict("equations");
        if (eqns) rc.relaxHe = eqns->scalarOr("h", rc.relaxHe);
    }

    return rc;
}

} // namespace brae
