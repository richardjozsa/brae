#pragma once
// CUDA implementation of simpleFoam's pressure corrector -- the device twin of pEqn_cpp.
//
// provenance:
//   openfoam:  applications/solvers/incompressible/simpleFoam/pEqn.H:1-50
//   reference: src/applications/solvers/simpleFoam/pEqn_cpp.cu   (validated stage by stage vs OpenFOAM)
//   cuda:      src/applications/solvers/simpleFoam/pEqn.cu
//   tests:     tests/test_peqn_cuda.cu
//
// SAME STAGE DECOMPOSITION AS THE REFERENCE, for the same reason: pEqn.H has seven places it can be wrong
// -- rAU, HbyA, the flux, adjustPhi, the Laplacian, the reference cell and the flux correction -- and a
// fused device path can only be compared as one number. Every intermediate is a named buffer here so a
// disagreement names the stage. A past investigation ended at `phi = phiHbyA - pEqn.flux()`, stage 7.
//
// REFUSED, not ignored -- identical to the reference: MRF, fvOptions, and `consistent` (SIMPLEC).
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_ldu.cuh"
#include "UEqn.cuh"

namespace brae {
namespace gpu {

struct PressureInput
{
    scalar relaxP = 1.0;
    label  pRefCell = -1;        // -1 => the case needs no reference; adjustPhi is then skipped
    scalar pRefValue = 0.0;
    bool   consistent = false;   // refused
    bool   hasMRF = false;       // refused
    bool   hasFvOptions = false; // refused
    // adjustPhi's per-boundary-face mask: 1 where the U patch does NOT fix a value, i.e. where the flux
    // may be scaled. Built host-side from !U.boundary[pi]->fixesValue(), matching adjustPhi.C -- which
    // branches on fixesValue(), NOT on assignable(). constrainHbyA below branches on the other one.
    const DeviceBuffer<label>* adjustable = nullptr;
    // constrainHbyA's per-boundary-face mask: 1 where HbyA must take U's boundary value, i.e. where the
    // U patch is NOT assignable (constrainHbyA.C). fixedValue/noSlip/mixed/transform are not assignable;
    // zeroGradient is. The two masks differ on slip and inletOutlet, so they are two arguments.
    const DeviceBuffer<label>* takeUAtBoundary = nullptr;
};

// Every intermediate of pEqn.H, in the order OpenFOAM produces them.
struct PressureStages
{
    DeviceBuffer<scalar> rAU;              // 1/UEqn.A()
    DeviceBuffer<scalar> HbyA[3];          // cells
    DeviceBuffer<scalar> HbyAb[3];         // boundary faces, AFTER constrainHbyA
    DeviceBuffer<scalar> phiHbyAInt, phiHbyABnd;
    scalar massCorr = 1.0;
    bool   phiAdjusted = false;
};

// Stages 1-3: rAU, HbyA (constrained), phiHbyA (adjusted). Solves nothing.
void pressurePredictor(
    PressureStages&              st,
    const DeviceMesh&            dm,
    const DeviceVectorBoundary&  dbU,
    const MomentumMatrix&        UEqn,
    const DeviceBuffer<scalar>&  Ux,
    const DeviceBuffer<scalar>&  Uy,
    const DeviceBuffer<scalar>&  Uz,
    const PressureInput&         in);

// The assembled pressure equation, in the reference's decomposition.
struct PressureMatrix
{
    DeviceBuffer<scalar> diag, upper, lower;
    DeviceBuffer<scalar> source;
    DeviceBuffer<scalar> iC, bC;

    DeviceLduView view(const DeviceMesh& dm) const
    {
        DeviceLduView A{};
        A.nCells = dm.nCells; A.nInternalFaces = dm.nInternalFaces;
        A.diag = diag.data(); A.upper = upper.data(); A.lower = lower.data();
        A.owner = dm.owner.data(); A.nei = dm.nei.data();
        A.ownerStart = dm.ownerStart.data();
        A.losort = dm.losort.data(); A.losortStart = dm.losortStart.data();
        return A;
    }
};

// Stages 4-5: fvm::laplacian(rAU, p) == fvc::div(phiHbyA), then setReference.
void assemblePEqn(
    PressureMatrix&              P,
    const PressureStages&        st,
    const DeviceMesh&            dm,
    const DeviceBoundary&        dbP,
    const DeviceBuffer<scalar>&  rAUface,   // fvc::interpolate(rAU), internal faces
    const PressureInput&         in);

// Stage 7: phi = phiHbyA - pEqn.flux(), with the SOLVED pressure.
void correctFlux(
    DeviceBuffer<scalar>&        phiInt,
    DeviceBuffer<scalar>&        phiBnd,
    const PressureStages&        st,
    const PressureMatrix&        P,
    const DeviceMesh&            dm,
    const DeviceBoundary&        dbP,
    const DeviceBuffer<scalar>&  pSolved);

// GeometricField::relax -- p = pPrev + alpha*(p - pPrev)   (GeometricField.C:1094)
void relaxField(DeviceBuffer<scalar>& p, const DeviceBuffer<scalar>& pPrev, scalar alpha);

// Stage 9: U = HbyA - rAU*fvc::grad(p), with the RELAXED pressure.
void correctVelocity(
    DeviceBuffer<scalar>&        Ux,
    DeviceBuffer<scalar>&        Uy,
    DeviceBuffer<scalar>&        Uz,
    const PressureStages&        st,
    const DeviceBuffer<scalar>&  gradPx,
    const DeviceBuffer<scalar>&  gradPy,
    const DeviceBuffer<scalar>&  gradPz);

} // namespace gpu
} // namespace brae
