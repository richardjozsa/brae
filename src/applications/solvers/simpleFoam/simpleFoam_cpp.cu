// _cpp REFERENCE DRIVER -- see simpleFoam_cpp.cuh for provenance and the ordering notes.
#include "simpleFoam_cpp.cuh"
#include "UEqn_cpp.cuh"
#include "pEqn_cpp.cuh"
#include "solve_vector.cuh"
#include "gamg.cuh"
#include "k_epsilon.cuh"
#include "kOmegaSST_cpp.cuh"
#include "SpalartAllmaras_cpp.cuh"

namespace brae {
namespace cpu {

Residuals simpleStep(
    SimpleFields&               f,
    SimpleControl&              ctl,
    const StepInput&            in,
    const PrimitiveMesh&        m,
    const FvGeometry&           g,
    const std::vector<FvPatch>& patches)
{
    Residuals res;

    // nuEff for THIS iteration comes from the nut left by the PREVIOUS one -- simpleFoam.C corrects the
    // turbulence at the END of the iteration, so the coupling is lagged. Boundary nuEff uses nut's own
    // boundary values (a wall function's nut_wall), not the owner cell's.
    std::vector<scalar> nuEffC;
    std::vector<std::vector<scalar>> nuEffB;
    const std::vector<scalar>* nuEffPtr = &in.nuEff;
    const std::vector<std::vector<scalar>>* nuEffBndPtr = &in.nuEffBnd;
    if (in.turb)
    {
        const GeometricField<scalar>& nut = *in.turb->nut;
        nuEffC.resize(nut.internal.size());
        for (std::size_t c = 0; c < nuEffC.size(); ++c) nuEffC[c] = in.nu + nut.internal[c];
        nuEffB.resize(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            const std::vector<scalar>& nb = nut.boundary[pi]->value();
            nuEffB[pi].resize(patches[pi].size);
            for (label i = 0; i < patches[pi].size; ++i) nuEffB[pi][i] = in.nu + nb[i];
        }
        nuEffPtr = &nuEffC;
        nuEffBndPtr = &nuEffB;
    }

    // storePrevIterFields(): OpenFOAM banks prevIter at the TOP of the iteration (simpleControl::loop),
    // so p.relax() below relaxes against the value p had before this iteration touched it.
    const std::vector<scalar> pPrev = f.p.internal;

    // OF's updateCoeffs is lazy: a patch recomputes its coefficients the first time the matrix asks for
    // them, which for the freestream family means recomputing the valueFraction from the CURRENT flow
    // angle. Doing it here, once at the top, is the same lag -- Up is still the previous iteration's
    // patch velocity either way. Non-mixed patches are untouched, so nothing else moves.
    {
        std::vector<std::vector<vector>> Ub(patches.size());
        for (std::size_t pi = 0; pi < patches.size(); ++pi)
        {
            Ub[pi] = f.U.boundary[pi]->value();
        }
        updateMixedFreestream(f.U.boundary, Ub, patches);
        // p too: the momentum equation reads p's boundary VALUE through -fvc::grad(p), and in OpenFOAM
        // that value is the blend the previous iteration's updateCoeffs left there. Seeding it on the
        // first iteration as well is what makes iteration 1 comparable rather than a special case.
        updateMixedFreestream(f.p.boundary, Ub, patches);
        f.p.evaluateBoundary();
    }

    // ---- UEqn.H ----------------------------------------------------------------------------
    // MRF.correctBoundaryVelocity(U) -- UEqn.H:3, the FIRST thing the momentum block does. The included
    // patch faces take the frame velocity, so the convection and the wall shear both see it.
    if (in.mrf) MRF::correctBoundaryVelocity(f.U, *in.mrf, patches);

    MomentumInput mi;
    mi.phi = &f.phi.internal;  mi.phiBnd = &f.phi.boundary;
    mi.nuEff = nuEffPtr;       mi.nuEffBnd = nuEffBndPtr;
    mi.relaxU = in.relaxU;
    mi.correctedLaplacian = in.correctedLaplacian;
    mi.snGradLimitCoeff   = in.snGradLimitCoeff;
    mi.bounded = in.bounded;
    mi.linearUpwind = in.linearUpwind;
    mi.gradULimitK  = in.gradULimitK;
    mi.scheme       = in.scheme;
    mi.schemeCoeff  = in.schemeCoeff;
    mi.hasMRF = in.hasMRF;     mi.hasFvOptions = in.hasFvOptions;
    mi.mrf = in.mrf;

    const FvVectorMatrix UEqn = assembleUEqn(f.U, mi, m, g, patches);

    if (ctl.momentumPredictor())
    {
        // solve(UEqn == -fvc::grad(p)) on a COPY: pEqn.H needs the original UEqn for A() and H().
        FvVectorMatrix Mp = UEqn;
        addPressureGradient(Mp, f.p, m, g, patches);
        const SolverPerformance up =
            solveVector(Mp, f.U, m, patches, in.tolU, in.relTolU, in.maxIter);
        res["U"] = up.initialResidual;
    }

    // ---- pEqn.H ----------------------------------------------------------------------------
    PressureInput pi;
    pi.relaxP = in.relaxP;
    pi.pRefCell = f.pRefCell;  pi.pRefValue = f.pRefValue;
    pi.consistent = ctl.consistent();
    pi.correctedLaplacian = in.correctedLaplacian;
    pi.snGradLimitCoeff   = in.snGradLimitCoeff;
    pi.hasMRF = in.hasMRF;     pi.hasFvOptions = in.hasFvOptions;
    pi.mrf = in.mrf;

    // p's turn. OF's updateCoeffs is called when the PRESSURE matrix asks for the coefficients, which is
    // after the momentum predictor -- so freestreamPressure sees the POST-predictor velocity, not the one
    // U's own valueFraction was built from at the top of the iteration.
    {
        std::vector<std::vector<vector>> Ub(patches.size());
        for (std::size_t pi2 = 0; pi2 < patches.size(); ++pi2)
        {
            Ub[pi2] = f.U.boundary[pi2]->value();
        }
        updateMixedFreestream(f.p.boundary, Ub, patches);
        f.p.evaluateBoundary();
    }

    const PressureStages st = pressurePredictor(UEqn, f.U, f.p, pi, m, g, patches);

    // Non-orthogonal corrector loop. Each pass re-assembles from the SAME phiHbyA and solves again; only
    // the final pass writes phi (simple.finalNonOrthogonalIter()).
    while (ctl.correctNonOrthogonal())
    {
        const FvScalarMatrix pEqn = assemblePEqn(st, f.p, pi, m, g, patches);
        const SolverPerformance pp =
            gamg(pEqn, f.p.internal, m, g, patches, in.tolP, in.relTolP, in.maxIter);
        if (res.find("p") == res.end()) res["p"] = pp.initialResidual;   // the FIRST solve's residual
        f.p.evaluateBoundary();

        if (ctl.finalNonOrthogonalIter())
        {
            f.phi = correctFlux(st, pEqn, f.p.internal, m, patches);
        }
    }

    // p.relax(), then the momentum corrector -- in that order, so phi came from the unrelaxed pressure
    // and U comes from the relaxed one.
    relaxField(f.p.internal, pPrev, in.relaxP);
    f.p.evaluateBoundary();

    f.U.internal = correctVelocity(st, f.p, m, g, patches);
    f.U.evaluateBoundary();

    // simpleFoam.C:93-94 -- laminarTransport.correct(); turbulence->correct();
    // A Newtonian transport model makes the first a no-op, which is why only the second appears here; a
    // non-Newtonian one is a separate manifest component and is not covered.
    if (in.turb)
    {
        if (in.turb->sa)
        {
            SA::DivScheme sch;
            sch.bounded       = in.turb->boundedTurb;
            sch.linearUpwind  = in.turb->linearUpwindTurb;
            sch.limitedLinear = in.turb->limitedLinearTurb;
            sch.coeff         = in.turb->turbLimiterCoeff;
            SA::correct(f.U, *in.turb->k, *in.turb->nut, f.phi, in.turb->y, in.nu,
                        m, g, patches, in.turb->relaxK, in.turb->tol, in.turb->relTol,
                        in.turb->maxIter, in.turb->saCoeffs, sch, in.turb->saRes);
        }
        else if (in.turb->sst)
        {
            kOmegaSST::correct(f.U, *in.turb->k, *in.turb->epsilon, *in.turb->nut, f.phi, in.turb->y,
                               in.nu, m, g, patches, in.turb->relaxEpsilon, in.turb->relaxK,
                               in.turb->tol, in.turb->relTol, in.turb->maxIter, in.turb->sstCoeffs,
                               /*res*/nullptr, in.turb->boundedTurb, in.turb->limitedLinearTurb,
                               in.turb->turbLimiterCoeff);
        }
        else
        {
            kepsilon::TurbDiv sch;
            sch.bounded       = in.turb->boundedTurb;
            sch.limitedLinear = in.turb->limitedLinearTurb;
            sch.coeff         = in.turb->turbLimiterCoeff;
            kepsilon::correct(f.U, *in.turb->k, *in.turb->epsilon, *in.turb->nut, f.phi, in.nu,
                              m, g, patches, in.turb->relaxEpsilon, in.turb->relaxK,
                              in.turb->tol, in.turb->relTol, in.turb->maxIter, in.turb->coeffs, sch);
        }
    }

    return res;
}

} // namespace cpu
} // namespace brae
