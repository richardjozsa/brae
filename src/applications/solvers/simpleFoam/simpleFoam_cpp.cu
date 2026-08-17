// _cpp REFERENCE DRIVER -- see simpleFoam_cpp.cuh for provenance and the ordering notes.
#include "simpleFoam_cpp.cuh"
#include "UEqn_cpp.cuh"
#include "pEqn_cpp.cuh"
#include "solve_vector.cuh"
#include "gamg.cuh"

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

    // storePrevIterFields(): OpenFOAM banks prevIter at the TOP of the iteration (simpleControl::loop),
    // so p.relax() below relaxes against the value p had before this iteration touched it.
    const std::vector<scalar> pPrev = f.p.internal;

    // ---- UEqn.H ----------------------------------------------------------------------------
    MomentumInput mi;
    mi.phi = &f.phi.internal;  mi.phiBnd = &f.phi.boundary;
    mi.nuEff = &in.nuEff;      mi.nuEffBnd = &in.nuEffBnd;
    mi.relaxU = in.relaxU;
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

    return res;
}

} // namespace cpu
} // namespace brae
