// CUDA implementation -- see pEqn.cuh for the provenance and the contract with the _cpp reference.
#include "pEqn.cuh"
#include "device_blas.cuh"
#include "device_simple.cuh"
#include <cuda_runtime.h>
#include <stdexcept>

namespace brae {
namespace gpu {

namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

void refuseUnsupported(const PressureInput& in)
{
    if (in.hasMRF)
        throw std::runtime_error(
            "pEqn(cuda): the case declares MRF, which pEqn.H applies via MRF.makeRelative(phiHbyA) "
            "(pEqn.H:5) and inside constrainPressure (pEqn.H:21). Not implemented on this path; refusing.");
    if (in.hasFvOptions)
        throw std::runtime_error(
            "pEqn(cuda): the case declares fvOptions, which pEqn.H applies as fvOptions.correct(U) "
            "(pEqn.H:49). Not implemented on this path; refusing.");
    if (in.consistent)
        throw std::runtime_error(
            "pEqn(cuda): SIMPLE/consistent (SIMPLEC) is set. pEqn.H:10-16 then replaces rAU with "
            "1/(1/rAU - UEqn.H1()) and corrects phiHbyA and HbyA with fvc::snGrad(p); neither is ported. "
            "Refusing rather than silently running plain SIMPLE.");
}

// constrainHbyA (constrainHbyA.C): on a patch whose U BC is NOT assignable, HbyA's boundary value is
// U's boundary value; elsewhere HbyA keeps the extrapolatedCalculated value it inherits from
// fvMatrix::H(), which is the owner cell's.
//
// `assignable` is not `fixesValue` -- slip and inletOutlet are non-assignable without fixing a value --
// so the mask is supplied by the caller rather than derived from bcType here.
__global__
void constrainHbyAKernel(
    int nB,
    const label* __restrict__ faceCell,
    const label* __restrict__ takeU,
    const scalar* __restrict__ Ub,        // U's boundary value
    const scalar* __restrict__ HbyACell,
    scalar* __restrict__ HbyAb)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nB) return;
    HbyAb[i] = takeU[i] ? Ub[i] : HbyACell[faceCell[i]];
}

// D += cmptAv(internalCoeffs) over the cell's boundary faces -- fvMatrix::D()'s
// addCmptAvBoundaryDiag, the numerator of A().
//
// GATHERED per cell via bndCellStart/bndPerm, not scattered with atomicAdd. A cell can own several
// boundary faces (any corner cell does), so the scatter form would make A() -- and therefore rAU, HbyA
// and the whole pressure equation -- depend on the order the faces happened to be scheduled in. That is
// the defect class the determinism work removed from the AMG restriction and the wall functions; adding
// a new instance of it here would reopen it.
__global__
void foldBoundaryDiagKernel(
    int nC,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const scalar* __restrict__ icAv,
    scalar* __restrict__ D)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar s = 0.0;
    for (label k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k) s += icAv[bndPerm[k]];
    D[c] += s;
}

} // namespace


void pressurePredictor(
    PressureStages&              st,
    const DeviceMesh&            dm,
    const DeviceVectorBoundary&  dbU,
    const MomentumMatrix&        UEqn,
    const DeviceBuffer<scalar>&  Ux,
    const DeviceBuffer<scalar>&  Uy,
    const DeviceBuffer<scalar>&  Uz,
    const PressureInput&         in)
{
    refuseUnsupported(in);
    if (!in.takeUAtBoundary)
        throw std::runtime_error("pEqn(cuda): constrainHbyA needs the per-face `assignable` mask.");

    // cmptAv(internalCoeffs) -- shared by A() and by H()'s boundary-diagonal term.
    DeviceBuffer<scalar> icAv;
    deviceCopy(icAv, UEqn.iC[0]);
    deviceAxpy(1.0, UEqn.iC[1], icAv);
    deviceAxpy(1.0, UEqn.iC[2], icAv);
    deviceScale(icAv, 1.0 / 3.0);

    // ---- rAU = 1/A() -------------------------------------------------------------------------
    // A() = D/V with D = diag + cmptAv(internalCoeffs). `diag` here is the RELAXED diagonal when the
    // matrix has been relaxed -- OpenFOAM's relax() writes diag_ in place and A() is taken afterwards, so
    // the boundary average is added ON TOP of the relaxed value, not instead of it.
    {
        DeviceBuffer<scalar> D;
        deviceCopy(D, UEqn.relaxed ? UEqn.relaxedDiag : UEqn.diag);
        foldBoundaryDiagKernel<<<nBlocks(dm.nCells), TPB>>>(
            dm.nCells, dm.bndCellStart.data(), dm.bndPerm.data(), icAv.data(), D.data());
        cudaCheck(cudaGetLastError(), "foldBoundaryDiag");
        deviceReciprocalV(dm, D, st.rAU);
    }

    // ---- HbyA = rAU * H() --------------------------------------------------------------------
    const DeviceBuffer<scalar>* U[3] = {&Ux, &Uy, &Uz};
    const DeviceLduView A = UEqn.view(dm);
    for (int k = 0; k < 3; ++k)
    {
        // bdDiagK = cmptAv(iC) - iC_k ; bdSrcK = bC_k   (fvMatrix::H's boundary-diagonal term)
        DeviceBuffer<scalar> bdDiagK;
        deviceCopy(bdDiagK, icAv);
        deviceAxpy(-1.0, UEqn.iC[k], bdDiagK);

        DeviceBuffer<scalar> Hk;
        deviceMatrixH(A, dm, *U[k], UEqn.source[k], bdDiagK, UEqn.bC[k], Hk);
        deviceHadamard(st.HbyA[k], st.rAU, Hk);
    }

    // ---- constrainHbyA(HbyA, U, p) -----------------------------------------------------------
    for (int k = 0; k < 3; ++k)
    {
        DeviceBuffer<scalar> Ub;
        deviceBCValue(dbU.comp[k], *U[k], Ub);
        st.HbyAb[k].resize(dm.nBndFaces);
        if (dm.nBndFaces > 0)
        {
            constrainHbyAKernel<<<nBlocks(dm.nBndFaces), TPB>>>(
                dm.nBndFaces, dm.bndCell.data(), in.takeUAtBoundary->data(),
                Ub.data(), st.HbyA[k].data(), st.HbyAb[k].data());
        }
    }
    cudaCheck(cudaGetLastError(), "constrainHbyA");

    // ---- phiHbyA = fvc::flux(HbyA) -----------------------------------------------------------
    deviceVectorFlux(dm, st.HbyA[0], st.HbyA[1], st.HbyA[2], st.phiHbyAInt);
    deviceBoundaryFlux(dm, st.HbyAb[0], st.HbyAb[1], st.HbyAb[2], st.phiHbyABnd);

    // ---- adjustPhi(phiHbyA, U, p) ------------------------------------------------------------
    // Only when p needs a reference. Without it a closed-outlet case hands the singular pressure operator
    // an inconsistent right-hand side, which has no solution at all.
    st.massCorr = 1.0;
    st.phiAdjusted = false;
    if (in.pRefCell >= 0)
    {
        if (!in.adjustable)
            throw std::runtime_error("pEqn(cuda): adjustPhi needs the per-face `adjustable` mask.");
        st.massCorr = deviceAdjustPhi(*in.adjustable, st.phiHbyABnd);
        st.phiAdjusted = (st.massCorr != 1.0);
    }
}


void assemblePEqn(
    PressureMatrix&              P,
    const PressureStages&        st,
    const DeviceMesh&            dm,
    const DeviceBoundary&        dbP,
    const DeviceBuffer<scalar>&  rAUface,
    const PressureInput&         in,
    const DeviceBuffer<scalar>*  p)
{
    refuseUnsupported(in);
    if (in.correctedLaplacian && !p)
        throw std::runtime_error("pEqn(cuda): the non-orthogonal correction needs grad(p), so the pressure "
                                 "field must be supplied to assemblePEqn.");

    // fvm::laplacian(rAU, p). The boundary diffusivity is rAU's own boundary value, which for the
    // extrapolatedCalculated field fvMatrix::A() produces IS the owner cell's -- hence the cell-gamma
    // kernel here rather than the face variant used for nuEff.
    deviceLaplacianCoeffs(dm, rAUface, P.diag, P.upper, P.lower, in.correctedLaplacian);
    deviceBCLaplacianCoeffs(dbP, st.rAU, P.iC, P.bC);

    // == fvc::div(phiHbyA), extensive: source = V*div(phiHbyA).
    {
        DeviceBuffer<scalar> divPhi;
        deviceDiv(dm, st.phiHbyAInt, st.phiHbyABnd, divPhi);
        deviceHadamard(P.source, divPhi, dm.V);
    }

    // The explicit non-orthogonal correction. The pressure laplacian keeps OpenFOAM's OWN sign here
    // (source += the laplacian's correction), unlike the momentum one -- device_simple_foam.cu:2649-2651.
    if (in.correctedLaplacian)
    {
        DeviceBuffer<scalar> pb, gx, gy, gz, lc;
        deviceBCValue(dbP, *p, pb);
        deviceGaussGrad(dm, *p, pb, gx, gy, gz);
        deviceLaplacianCorr(dm, rAUface, gx, gy, gz, lc);
        deviceAxpy(1.0, lc, P.source);
    }

    // pEqn.setReference -- fvMatrix.C:1011-1023: source += diag*value; diag += diag. It DOUBLES the
    // diagonal rather than setting it; a "pin the cell" reading builds a different matrix.
    if (in.pRefCell >= 0)
    {
        deviceSetReference(P.diag, P.source, in.pRefCell, in.pRefValue);
    }
}


void correctFlux(
    DeviceBuffer<scalar>&        phiInt,
    DeviceBuffer<scalar>&        phiBnd,
    const PressureStages&        st,
    const PressureMatrix&        P,
    const DeviceMesh&            dm,
    const DeviceBoundary&        dbP,
    const DeviceBuffer<scalar>&  pSolved)
{
    // phi = phiHbyA - pEqn.flux(). The step that makes phi discretely conservative.
    DeviceBuffer<scalar> fInt, fBnd;
    deviceMatrixFluxInternal(P.view(dm), pSolved, fInt);
    deviceMatrixFluxBoundary(dbP, P.iC, P.bC, pSolved, fBnd);

    deviceCopy(phiInt, st.phiHbyAInt);
    deviceAxpy(-1.0, fInt, phiInt);
    deviceCopy(phiBnd, st.phiHbyABnd);
    deviceAxpy(-1.0, fBnd, phiBnd);
}


void relaxField(DeviceBuffer<scalar>& p, const DeviceBuffer<scalar>& pPrev, scalar alpha)
{
    // p = pPrev + alpha*(p - pPrev) = alpha*p + (1-alpha)*pPrev
    if (alpha >= 1.0 || alpha <= 0.0) return;
    deviceScale(p, alpha);
    deviceAxpy(1.0 - alpha, pPrev, p);
}


void correctVelocity(
    DeviceBuffer<scalar>&        Ux,
    DeviceBuffer<scalar>&        Uy,
    DeviceBuffer<scalar>&        Uz,
    const PressureStages&        st,
    const DeviceBuffer<scalar>&  gradPx,
    const DeviceBuffer<scalar>&  gradPy,
    const DeviceBuffer<scalar>&  gradPz)
{
    // U = HbyA - rAU*grad(p), with the RELAXED p -- pEqn.H relaxes before this line, so the velocity
    // correction and the flux correction deliberately see different pressures.
    deviceCorrector(st.HbyA[0], st.rAU, gradPx, Ux);
    deviceCorrector(st.HbyA[1], st.rAU, gradPy, Uy);
    deviceCorrector(st.HbyA[2], st.rAU, gradPz, Uz);
}

} // namespace gpu
} // namespace brae
