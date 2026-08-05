#pragma once
// linear_solver_setup.cuh -- fvSolution -> DeviceSimpleControls, for EVERY solver driver.
//
// This exists because the compressible driver was ported by copying the parts of gpuSimpleFoam it needed
// to make the physics run, and fifteen controls were left behind. Each one then fell back to a struct
// default whose own comment said it should come from fvSolution:
//
//   relTol{P,U,KE}   unset -> 0     => every equation solved to ABSOLUTE tolerance every outer iteration,
//                                      where OF does a loose solve (all six rhoSimpleFoam tutorials set it)
//   consistent       unread         => "consistent yes" silently ran SIMPLE instead of SIMPLEC
//   nNonOrth         unset -> 0     => nNonOrthogonalCorrectors ignored
//   gs{U,K,Eps}      unset -> false => "solver smoothSolver" ignored, always BiCGStab
//   {bicg,pcg}CheckEvery, useGraph, corrScaling, bodyForce
//
// None of that is visible in a converged field on a case that happens not to use them, which is why it
// survived. Reading them in ONE place is the structural fix: a new driver gets the whole set or none.
//
// Everything here is read-only on the dicts and writes only into ctl.

#include "solver_controls.cuh"
#include "foam_dict.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace brae {

// fvSolution solvers.{p,U,<second>,...} tolerances/relTol + smoothSolver selection, and the SIMPLE
// sub-dict controls (consistent / nNonOrthogonalCorrectors / bodyForce).
//
// secondName is the 2nd turbulence scalar's FIELD name: "omega" on kOmegaSST, "epsilon" on kEpsilon,
// unused when ctl.sa (SA solves nuTilda only). Pass it explicitly -- deriving it here would re-hardcode
// the thing that already went wrong once.
//
// algorithmDict is the fvSolution sub-dict holding the algorithm controls: "SIMPLE" for the steady
// solvers, "PIMPLE" for the transient one. OF parameterises this the same way -- solutionControl reads
// consistent/nNonOrthogonalCorrectors from subOrEmptyDict(algorithmName_) (solutionControl.C:46,51,302)
// and the name is a constructor argument defaulting to "SIMPLE" in simpleControl.H:100 and "PIMPLE" in
// pimpleControl.H:135. Hardcoding "SIMPLE" here would read nNonOrthogonalCorrectors as 0 on every
// transient case, so this is required, not cosmetic.
inline void readLinearSolverControls(
    const FoamDict& fvSolution,
    const std::string& secondName,
    DeviceSimpleControls& ctl,
    const std::string& algorithmDict = "SIMPLE")
{
    const FoamDict* solvers = fvSolution.subDict("solvers");

    auto solverTol = [&](const std::string& f, scalar def)
    {
        const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
        return s ? s->scalarOr("tolerance", def) : def;
    };
    auto solverRelTol = [&](const std::string& f)   // SIMPLE only needs a loose per-step solve
    {
        const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
        return s ? s->scalarOr("relTol", 0.0) : 0.0;
    };
    // OF's selection, exactly: solver smoothSolver + a GaussSeidel-family smoother -> brae's symmetric
    // multicolor deviceSymGaussSeidel. Anything else (PBiCG[Stab]/GAMG/...) keeps BiCGStab.
    auto useSymGS = [&](const std::string& f)
    {
        const FoamDict* s = solvers ? solvers->subDict(f) : nullptr;
        if (!s || s->wordOr("solver", "") != "smoothSolver") return false;
        const std::string sm = s->wordOr("smoother", "");
        return sm == "symGaussSeidel" || sm == "GaussSeidel";
    };

    ctl.tolP = solverTol("p", 1e-6);
    ctl.tolU = solverTol("U", 1e-8);
    ctl.relTolP = solverRelTol("p");
    ctl.relTolU = solverRelTol("U");
    ctl.gsU = useSymGS("U");
    if (const char* gsuEnv = std::getenv("BRAE_GS_U"))
        ctl.gsU = (std::atoi(gsuEnv) != 0) && ctl.gsU;

    if (ctl.turbulent)
    {
        if (ctl.sa)
        {
            ctl.tolKE = solverTol("nuTilda", 1e-8);
            ctl.relTolKE = solverRelTol("nuTilda");
            ctl.gsK = useSymGS("nuTilda");
            ctl.gsEps = false;
        }
        else
        {
            ctl.tolKE = std::fmin(solverTol("k", 1e-8), solverTol(secondName, 1e-8));
            ctl.relTolKE = std::fmin(solverRelTol("k"), solverRelTol(secondName));
            ctl.gsK = useSymGS("k");
            ctl.gsEps = useSymGS(secondName);
        }
    }

    const FoamDict* algo = fvSolution.subDict(algorithmDict);
    {
        const std::string cons = algo ? algo->wordOr("consistent", "no") : "no";
        ctl.consistent = (cons == "yes" || cons == "true" || cons == "on" || cons == "1");   // SIMPLEC
    }
    ctl.nNonOrth = algo ? algo->intOr("nNonOrthogonalCorrectors", 0) : 0;
    {
        const std::vector<scalar> bf = algo ? algo->scalarListOr("bodyForce", {}) : std::vector<scalar>{};
        if (bf.size() >= 3) ctl.bodyForce = vector{bf[0], bf[1], bf[2]};   // constant momentum source
    }

    // Performance knobs (no effect on the converged answer). Kept here so a driver cannot get the
    // correctness controls and miss these, which is how they drifted apart before.
    ctl.pcgCheckEvery = 4;   // batched PCG residual read; OF-validated identical to K=1
    if (const char* ce = std::getenv("BRAE_PCG_CHECK_EVERY"))
    {
        const int k = std::atoi(ce);
        if (k >= 1) ctl.pcgCheckEvery = k;
    }
    if (const char* be = std::getenv("BRAE_BICG_CHECK_EVERY"))
    {
        const int k = std::atoi(be);
        if (k >= 1) ctl.bicgCheckEvery = k;
    }
    if (const char* cs = std::getenv("BRAE_CORR_SCALING")) ctl.corrScaling = (std::atoi(cs) != 0);
    if (const char* ug = std::getenv("BRAE_USE_GRAPH")) ctl.useGraph = (std::atoi(ug) != 0);
}

// relaxationFactors -> ctl.relax{U,P,K,Eps}. Third copy of this in the tree when it was written, each a
// different subset -- the steady driver had all of it, the compressible one had no alpha<=0 guard, and the
// transient one had neither the guard, nor the legacy form, nor the right key for the second scalar (it
// reused k's factor for epsilon/omega, so `omega 0.4` was ignored and omega ran at k's factor).
//
// Call AFTER the turbulence model is read: it branches on ctl.sa/ctl.sst to pick the field names.
inline void readRelaxationFactors(const FoamDict& fvSolution, DeviceSimpleControls& ctl)
{
    const FoamDict* rf  = fvSolution.subDict("relaxationFactors");
    const FoamDict* eqs = rf ? rf->subDict("equations") : nullptr;
    const FoamDict* fld = rf ? rf->subDict("fields") : nullptr;
    // OF accepts BOTH the modern nested {equations{} fields{}} and the legacy FLAT {p ..; U ..;} form. Fall
    // back to the flat keys when a sub-dict is absent, so a legacy case isn't silently left un-relaxed
    // (all factors 1.0 -> the steady SIMPLE loop typically diverges).
    const FoamDict* eqSrc  = eqs ? eqs : rf;
    const FoamDict* fldSrc = fld ? fld : rf;

    const char* kName    = ctl.sa ? "nuTilda" : "k";              // SA: relaxK carries the nuTilda relax
    const std::string sName = ctl.sst ? "omega" : "epsilon";

    ctl.relaxU   = eqSrc  ? eqSrc->scalarOr("U", 1.0) : 1.0;
    ctl.relaxK   = eqSrc  ? eqSrc->scalarOr(kName, 1.0) : 1.0;
    ctl.relaxEps = eqSrc  ? eqSrc->scalarOr(sName, 1.0) : 1.0;
    ctl.relaxP   = fldSrc ? fldSrc->scalarOr("p", 1.0) : 1.0;

    // A relaxation factor <= 0 divides by zero in the diagonal-relaxation kernel (Inf diag -> NaN). OF's
    // fvMatrix::relax skips relaxation for alpha <= 0; match that (treat as 1.0 = no under-relaxation) + warn.
    auto fixRelax = [](scalar& a, const char* nm)
    {
        if (a <= 0.0)
        {
            std::fprintf(stderr,
                "brae WARNING: relaxationFactors %s = %g <= 0; using 1.0 (no under-relaxation)\n", nm, (double)a);
            a = 1.0;
        }
    };
    fixRelax(ctl.relaxU, "U");
    fixRelax(ctl.relaxK, kName);
    fixRelax(ctl.relaxEps, sName.c_str());
    fixRelax(ctl.relaxP, "p");
}

// The energy equation's linear solver, which OF names "h" for sensibleEnthalpy and "e" for
// sensibleInternalEnergy. Was hardcoded tol=1e-10, relTol=0, BiCGStab regardless of the case.
struct EnergySolverControls
{
    scalar tol = 1e-10;
    scalar relTol = 0.0;
    bool   useGS = false;
};

inline EnergySolverControls readEnergySolverControls(
    const FoamDict& fvSolution,
    bool internalEnergy)
{
    EnergySolverControls e;
    const FoamDict* solvers = fvSolution.subDict("solvers");
    if (!solvers) return e;
    // Try the case's own energy-field name first, then the other, so a case listing only one is honoured.
    const char* primary = internalEnergy ? "e" : "h";
    const char* fallback = internalEnergy ? "h" : "e";
    const FoamDict* s = solvers->subDict(primary);
    if (!s) s = solvers->subDict(fallback);
    if (!s) return e;
    e.tol = s->scalarOr("tolerance", e.tol);
    e.relTol = s->scalarOr("relTol", e.relTol);
    if (s->wordOr("solver", "") == "smoothSolver")
    {
        const std::string sm = s->wordOr("smoother", "");
        e.useGS = (sm == "symGaussSeidel" || sm == "GaussSeidel");
    }
    return e;
}

}   // namespace brae
