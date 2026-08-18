// _cpp REFERENCE DRIVER -- see simpleFoam_cpp.cuh for provenance and the ordering notes.
#include "simpleFoam_cpp.cuh"
#include "UEqn_cpp.cuh"
#include "pEqn_cpp.cuh"
#include "solve_vector.cuh"
#include "gamg.cuh"
#include "k_epsilon.cuh"

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

    // ---- UEqn.H ----------------------------------------------------------------------------
    MomentumInput mi;
    mi.phi = &f.phi.internal;  mi.phiBnd = &f.phi.boundary;
    mi.nuEff = nuEffPtr;       mi.nuEffBnd = nuEffBndPtr;
    mi.relaxU = in.relaxU;
    mi.correctedLaplacian = in.correctedLaplacian;
    mi.bounded = in.bounded;
    mi.linearUpwind = in.linearUpwind;
    mi.scheme       = in.scheme;
    mi.schemeCoeff  = in.schemeCoeff;
    mi.hasMRF = in.hasMRF;     mi.hasFvOptions = in.hasFvOptions;

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
    pi.hasMRF = in.hasMRF;     pi.hasFvOptions = in.hasFvOptions;

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
        kepsilon::correct(f.U, *in.turb->k, *in.turb->epsilon, *in.turb->nut, f.phi, in.nu,
                          m, g, patches, in.turb->relaxEpsilon, in.turb->relaxK,
                          in.turb->tol, in.turb->relTol, in.turb->maxIter, in.turb->coeffs);
    }

    return res;
}

} // namespace cpu
} // namespace brae
