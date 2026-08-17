#pragma once
// CUDA DRIVER -- one simpleFoam SIMPLE iteration, device-resident, composed of the ported components.
//
// provenance:
//   openfoam:  applications/solvers/incompressible/simpleFoam/simpleFoam.C:79-97 (the time-loop body)
//   reference: src/applications/solvers/simpleFoam/simpleFoam_cpp.cu
//   cuda:      src/applications/solvers/simpleFoam/simpleFoam.cu
//   tests:     tests/test_simple_step_cuda.cu
//
// THE DRIVER OWNS NO NUMERICS -- the same contract as the _cpp driver it mirrors:
//
//     gpu::assembleUEqn         UEqn.cu           (fvm::div + divDevReff + relax)
//     gpu::addPressureGradient  UEqn.cu           (-fvc::grad(p))
//     deviceJacobiBiCGStab      lduMatrix/solvers (the momentum solve, non-symmetric)
//     gpu::pressurePredictor    pEqn.cu           (rAU, HbyA, phiHbyA, adjustPhi)
//     gpu::assemblePEqn         pEqn.cu           (laplacian == div(phiHbyA), setReference)
//     deviceAMGPCG              GAMGPreconditioner(the pressure solve)
//     gpu::correctFlux          pEqn.cu           (phi = phiHbyA - pEqn.flux())
//     gpu::relaxField           pEqn.cu           (p.relax())
//     gpu::correctVelocity      pEqn.cu           (U = HbyA - rAU*grad(p))
//     correct()                 caller-supplied   (turbulence->correct(), simpleFoam.C:94)
//
// Compare with what it replaces: device_simple_foam.cu, 3578 lines, included by pimpleFoam,
// rhoSimpleFoam and five headers in solvers/common.
//
// ORDERING, taken from OpenFOAM and not rearranged:
//   * the momentum predictor solves a COPY of UEqn with -grad(p) added; pEqn.H needs the original for
//     A() and H(), and adding grad(p) to UEqn itself changes rAU and HbyA;
//   * p is relaxed AFTER the flux correction and BEFORE the velocity correction, so phi is built from
//     the unrelaxed pressure and U from the relaxed one;
//   * turbulence->correct() runs at the END, so the next iteration's UEqn uses this iteration's nut --
//     a lagged coupling, not a simultaneous one.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_amg.cuh"
#include "UEqn.cuh"
#include "pEqn.cuh"
#include <functional>
#include <map>
#include <string>

namespace brae {
namespace gpu {

// The device-resident solution state the loop carries between iterations.
struct SolverFields
{
    DeviceBuffer<scalar> Ux, Uy, Uz;
    DeviceBuffer<scalar> p;
    DeviceBuffer<scalar> phiInt, phiBnd;
};

struct StepInput
{
    // nuEff for THIS iteration. Supplied by the caller because it comes from the turbulence model, which
    // the caller owns (see `correct` below): nuEff = nu + nut from the PREVIOUS iteration.
    const DeviceBuffer<scalar>* nuEffCell = nullptr;
    const DeviceBuffer<scalar>* nuEffFace = nullptr;
    const DeviceBuffer<scalar>* nuEffBndFace = nullptr;

    scalar relaxU = 0.7;
    scalar relaxP = 0.3;
    scalar tolU = 1e-10, relTolU = 0.0;
    scalar tolP = 1e-10, relTolP = 0.0;
    int    maxIter = 2000;
    bool   momentumPredictor = true;
    label  nNonOrthogonalCorrectors = 0;

    label  pRefCell = -1;
    scalar pRefValue = 0.0;

    const DeviceBuffer<label>* adjustable = nullptr;        // adjustPhi mask   (!fixesValue)
    const DeviceBuffer<label>* takeUAtBoundary = nullptr;   // constrainHbyA mask (!assignable)

    bool hasMRF = false, hasFvOptions = false, consistent = false;   // all refused downstream

    // turbulence->correct(), called at the END of the iteration exactly where simpleFoam.C:94 calls it.
    // A hook rather than a hard dependency because the model is runtime-selected: the driver must not
    // know which one it is, and a driver that hard-codes kEpsilon is the god object starting over.
    std::function<void()> correct;
};

using Residuals = std::map<std::string, scalar>;

// Scratch that persists across iterations (the AMG hierarchy is built once per mesh, and `ones` is the
// unit vector deviceNormFactor needs). Kept out of StepInput so the loop cannot accidentally rebuild it.
struct SolverWorkspace
{
    AMGData amg;
    bool    amgBuilt = false;
    DeviceBuffer<scalar> ones;
};

// One SIMPLE iteration, in place on `f`.
Residuals simpleStep(
    SolverFields&                f,
    SolverWorkspace&             w,
    const DeviceMesh&            dm,
    const DeviceVectorBoundary&  dbU,
    const DeviceBoundary&        dbP,
    const StepInput&             in);

} // namespace gpu
} // namespace brae
