// cf GPU offload (G7): SIMPLE coupling-glue kernels. matrixH and matrixFlux reuse the ownerStart/losort
// gather; rAU / corrector are elementwise; the HbyA flux is a per-face vector interpolation dotted with Sf.
#include "device_simple.cuh"
#include <cmath>
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

__global__ void matrixHKernel(int nC, const label* __restrict__ ownerStart, const label* __restrict__ losort,
                              const label* __restrict__ losortStart, const scalar* __restrict__ upper,
                              const scalar* __restrict__ lower, const label* __restrict__ own,
                              const label* __restrict__ nei, const scalar* __restrict__ psi,
                              const scalar* __restrict__ source, const scalar* __restrict__ V,
                              const label* __restrict__ bndCellStart, const label* __restrict__ bndPerm,
                              const scalar* __restrict__ bdDiag, const scalar* __restrict__ bdSrc, scalar* __restrict__ H) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar h = source[c];
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f) h -= upper[f] * psi[nei[f]];   // -upper*psi[nei] (owned faces)
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k) { const int f = losort[k]; h -= lower[f] * psi[own[f]]; }  // -lower*psi[own]
    scalar bd = 0, bs = 0;
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k) { const int kk = bndPerm[k]; bd += bdDiag[kk]; bs += bdSrc[kk]; }
    h += bd * psi[c] + bs;
    H[c] = h / V[c];
}
__global__ void reciprocalVKernel(int nC, const scalar* __restrict__ diagC, const scalar* __restrict__ V, scalar* __restrict__ rAU) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) rAU[c] = V[c] / diagC[c];
}
__global__ void vectorFluxKernel(int nIf, const label* __restrict__ own, const label* __restrict__ nei,
                                 const scalar* __restrict__ w, const scalar* __restrict__ Sfx,
                                 const scalar* __restrict__ Sfy, const scalar* __restrict__ Sfz,
                                 const scalar* __restrict__ Ux, const scalar* __restrict__ Uy,
                                 const scalar* __restrict__ Uz, scalar* __restrict__ phi) {
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    const scalar wf = w[f]; const int o = own[f], n = nei[f];
    const scalar ux = wf * Ux[o] + (1.0 - wf) * Ux[n];
    const scalar uy = wf * Uy[o] + (1.0 - wf) * Uy[n];
    const scalar uz = wf * Uz[o] + (1.0 - wf) * Uz[n];
    phi[f] = ux * Sfx[f] + uy * Sfy[f] + uz * Sfz[f];
}
__global__ void matrixFluxKernel(int nIf, const label* __restrict__ own, const label* __restrict__ nei,
                                 const scalar* __restrict__ upper, const scalar* __restrict__ lower,
                                 const scalar* __restrict__ p, scalar* __restrict__ flux) {
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < nIf) flux[f] = upper[f] * p[nei[f]] - lower[f] * p[own[f]];
}
__global__ void correctorKernel(int nC, const scalar* __restrict__ HbyA, const scalar* __restrict__ rAU,
                                const scalar* __restrict__ gradP, scalar* __restrict__ U) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < nC) U[c] = HbyA[c] - rAU[c] * gradP[c];
}
__global__ void bndFluxKernel(int nB, const label* __restrict__ bndGFace, const scalar* __restrict__ Sfx,
                              const scalar* __restrict__ Sfy, const scalar* __restrict__ Sfz,
                              const scalar* __restrict__ uxb, const scalar* __restrict__ uyb,
                              const scalar* __restrict__ uzb, scalar* __restrict__ phiB) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nB) return;
    const int f = bndGFace[i];
    phiB[i] = uxb[i] * Sfx[f] + uyb[i] * Sfy[f] + uzb[i] * Sfz[f];
}
__global__ void relaxKernel(int nC, const label* __restrict__ ownerStart, const label* __restrict__ losort,
                            const label* __restrict__ losortStart, const scalar* __restrict__ upper,
                            const scalar* __restrict__ lower, const label* __restrict__ bndCellStart,
                            const label* __restrict__ bndPerm, const scalar* __restrict__ iCbnd,
                            const scalar* __restrict__ rawDiag, scalar alpha, scalar* __restrict__ relaxedDiag, scalar* __restrict__ delta,
                            const scalar* __restrict__ cycSumOff, const scalar* __restrict__ iCmaxMag) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar sumOff = 0.0;
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f) sumOff += fabs(upper[f]);             // |upper| over owned faces
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k) sumOff += fabs(lower[losort[k]]);   // |lower| over neighbour faces
    if (cycSumOff) sumOff += cycSumOff[c];                                                        // |cyclic off-diagonal|
    scalar magSum = 0.0, minSum = 0.0;
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k) { const int p = bndPerm[k];
        magSum += iCmaxMag ? iCmaxMag[p] : fabs(iCbnd[p]);   // OF relax: cmptMax(cmptMag(iC)) over comps (= |iC0| when equal)
        minSum += iCbnd[p]; }
    const scalar d0 = rawDiag[c];
    const scalar dr = fmax(fabs(d0 + magSum), sumOff) / alpha - minSum;
    relaxedDiag[c] = dr; delta[c] = dr - d0;
}
} // namespace

void deviceMatrixH(const DeviceLduView& A, const DeviceMesh& dm, const DeviceBuffer<scalar>& psiK,
                   const DeviceBuffer<scalar>& sourceK, const DeviceBuffer<scalar>& bdDiagK,
                   const DeviceBuffer<scalar>& bdSrcK, DeviceBuffer<scalar>& Hk) {
    const int nC = dm.nCells; Hk.resize(nC);
    matrixHKernel<<<nBlocks(nC), TPB>>>(nC, A.ownerStart, A.losort, A.losortStart, A.upper, A.lower, A.owner, A.nei,
                                        psiK.data(), sourceK.data(), dm.V.data(), dm.bndCellStart.data(), dm.bndPerm.data(),
                                        bdDiagK.data(), bdSrcK.data(), Hk.data());
    cudaCheck(cudaGetLastError(), "matrixH");
}
// linearUpwind deferred correction for div(phi,U_i): the matrix stays pure upwind; the explicit correction
// is the convective transport of grad(U_i)_upwind . (Cf - C_upwind). Per cell: Sum(+/-) phi_f * (grad_upwind . d).
// (boundary faces use pure upwind -> no correction, as in OpenFOAM.) Caller does source -= corrSource.
__global__ void linearUpwindCorrKernel(int nC, const label* __restrict__ ownerStart, const label* __restrict__ losort,
        const label* __restrict__ losortStart, const label* __restrict__ own, const label* __restrict__ nei,
        const scalar* __restrict__ phi, const scalar* __restrict__ gx, const scalar* __restrict__ gy, const scalar* __restrict__ gz,
        const scalar* __restrict__ dOwnX, const scalar* __restrict__ dOwnY, const scalar* __restrict__ dOwnZ,
        const scalar* __restrict__ dNeiX, const scalar* __restrict__ dNeiY, const scalar* __restrict__ dNeiZ,
        scalar* __restrict__ corrSource) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= nC) return;
    scalar s = 0;
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f) {                       // c is owner (+)
        const scalar pf = phi[f]; const int up = (pf >= 0) ? own[f] : nei[f];
        const scalar dx = (pf >= 0) ? dOwnX[f] : dNeiX[f], dy = (pf >= 0) ? dOwnY[f] : dNeiY[f], dz = (pf >= 0) ? dOwnZ[f] : dNeiZ[f];
        s += pf * (gx[up]*dx + gy[up]*dy + gz[up]*dz);
    }
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k) {                     // c is neighbour (-)
        const int f = losort[k]; const scalar pf = phi[f]; const int up = (pf >= 0) ? own[f] : nei[f];
        const scalar dx = (pf >= 0) ? dOwnX[f] : dNeiX[f], dy = (pf >= 0) ? dOwnY[f] : dNeiY[f], dz = (pf >= 0) ? dOwnZ[f] : dNeiZ[f];
        s -= pf * (gx[up]*dx + gy[up]*dy + gz[up]*dz);
    }
    corrSource[c] = s;
}
void deviceLinearUpwindCorr(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& gx,
                            const DeviceBuffer<scalar>& gy, const DeviceBuffer<scalar>& gz, DeviceBuffer<scalar>& corrSource) {
    corrSource.resize(dm.nCells);
    linearUpwindCorrKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(),
        dm.owner.data(), dm.nei.data(), phiInt.data(), gx.data(), gy.data(), gz.data(),
        dm.dOwnX.data(), dm.dOwnY.data(), dm.dOwnZ.data(), dm.dNeiX.data(), dm.dNeiY.data(), dm.dNeiZ.data(), corrSource.data());
    cudaCheck(cudaGetLastError(), "linearUpwindCorr");
}

// linearUpwindV (OF finiteVolume/interpolation linearUpwindV.C correction()): the linearUpwind vector correction, but
// LIMITED so it cannot overshoot the owner<->neighbour difference in its own direction. It couples the 3 velocity
// components at each face (hence a face kernel, not the per-component linearUpwindCorrKernel):
//   sfCorr_i = (Cf-C_up).grad(U_i)_up ; maxCorr = (phi>0)?(1-w)(U[nei]-U[own]):w(U[own]-U[nei]) ;
//   sfCorrs=|sfCorr|^2, maxCorrs=sfCorr.maxCorr ; if maxCorrs<0 -> 0 ; else if sfCorrs>maxCorrs -> *= maxCorrs/sfCorrs.
__global__ void linearUpwindVFaceKernel(int nIf, const label* __restrict__ own, const label* __restrict__ nei,
        const scalar* __restrict__ phi, const scalar* __restrict__ w,
        const scalar* __restrict__ dOwnX, const scalar* __restrict__ dOwnY, const scalar* __restrict__ dOwnZ,
        const scalar* __restrict__ dNeiX, const scalar* __restrict__ dNeiY, const scalar* __restrict__ dNeiZ,
        const scalar* __restrict__ g0x, const scalar* __restrict__ g0y, const scalar* __restrict__ g0z,
        const scalar* __restrict__ g1x, const scalar* __restrict__ g1y, const scalar* __restrict__ g1z,
        const scalar* __restrict__ g2x, const scalar* __restrict__ g2y, const scalar* __restrict__ g2z,
        const scalar* __restrict__ U0, const scalar* __restrict__ U1, const scalar* __restrict__ U2,
        scalar* __restrict__ fcX, scalar* __restrict__ fcY, scalar* __restrict__ fcZ) {
    const int f = blockIdx.x * blockDim.x + threadIdx.x; if (f >= nIf) return;
    const scalar pf = phi[f]; const bool pos = (pf >= 0.0);
    const int o = own[f], n = nei[f]; const int up = pos ? o : n;
    const scalar dx = pos ? dOwnX[f] : dNeiX[f], dy = pos ? dOwnY[f] : dNeiY[f], dz = pos ? dOwnZ[f] : dNeiZ[f];
    scalar cx = g0x[up]*dx + g0y[up]*dy + g0z[up]*dz;                                 // (Cf-C_up).grad(U_i)_up
    scalar cy = g1x[up]*dx + g1y[up]*dy + g1z[up]*dz;
    scalar cz = g2x[up]*dx + g2y[up]*dy + g2z[up]*dz;
    const scalar s = pos ? (1.0 - w[f]) : w[f];                                       // OF maxCorr sign per flux
    const scalar mx = pos ? s*(U0[n]-U0[o]) : s*(U0[o]-U0[n]);
    const scalar my = pos ? s*(U1[n]-U1[o]) : s*(U1[o]-U1[n]);
    const scalar mz = pos ? s*(U2[n]-U2[o]) : s*(U2[o]-U2[n]);
    const scalar sfCorrs  = cx*cx + cy*cy + cz*cz;
    const scalar maxCorrs = cx*mx + cy*my + cz*mz;
    if (sfCorrs > 0.0) {
        if (maxCorrs < 0.0) { cx = 0.0; cy = 0.0; cz = 0.0; }
        else if (sfCorrs > maxCorrs) { const scalar r = maxCorrs / (sfCorrs + 1.0e-300); cx *= r; cy *= r; cz *= r; }
    }
    fcX[f] = cx; fcY[f] = cy; fcZ[f] = cz;
}
// div(phi * faceCorr) gather (per cell), same owner(+)/losort(-) sum as linearUpwindCorrKernel but on a precomputed face field.
__global__ void divFaceCorrKernel(int nC, const label* __restrict__ ownerStart, const label* __restrict__ losort,
        const label* __restrict__ losortStart, const scalar* __restrict__ phi, const scalar* __restrict__ fc,
        scalar* __restrict__ corrSource) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= nC) return;
    scalar s = 0;
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f) s += phi[f] * fc[f];
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k) { const int f = losort[k]; s -= phi[f] * fc[f]; }
    corrSource[c] = s;
}
void deviceLinearUpwindVCorr(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt,
        const DeviceBuffer<scalar>* gUx, const DeviceBuffer<scalar>* gUy, const DeviceBuffer<scalar>* gUz,
        const DeviceBuffer<scalar>& U0, const DeviceBuffer<scalar>& U1, const DeviceBuffer<scalar>& U2,
        DeviceBuffer<scalar>& corrX, DeviceBuffer<scalar>& corrY, DeviceBuffer<scalar>& corrZ) {
    const int nIf = dm.nInternalFaces;
    DeviceBuffer<scalar> fcX(nIf), fcY(nIf), fcZ(nIf);
    linearUpwindVFaceKernel<<<nBlocks(nIf), TPB>>>(nIf, dm.owner.data(), dm.nei.data(), phiInt.data(), dm.w.data(),
        dm.dOwnX.data(), dm.dOwnY.data(), dm.dOwnZ.data(), dm.dNeiX.data(), dm.dNeiY.data(), dm.dNeiZ.data(),
        gUx[0].data(), gUy[0].data(), gUz[0].data(), gUx[1].data(), gUy[1].data(), gUz[1].data(),
        gUx[2].data(), gUy[2].data(), gUz[2].data(), U0.data(), U1.data(), U2.data(),
        fcX.data(), fcY.data(), fcZ.data());
    corrX.resize(dm.nCells); corrY.resize(dm.nCells); corrZ.resize(dm.nCells);
    divFaceCorrKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(), phiInt.data(), fcX.data(), corrX.data());
    divFaceCorrKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(), phiInt.data(), fcY.data(), corrY.data());
    divFaceCorrKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(), phiInt.data(), fcZ.data(), corrZ.data());
    cudaCheck(cudaGetLastError(), "linearUpwindVCorr");
}

// LUST linear-part deferred correction: div(phi*(linear_face - upwind_face)) = div(phi*(w-pos0)*(field[P]-field[N])).
// Same divergence gather as linearUpwindCorrKernel. OF LUST = 0.75*linear + 0.25*linearUpwind, so the caller adds
// 0.75*this + 0.25*linearUpwindCorr. w = owner linear weight (dm.w), pos0 = (phi>=0) = the upwind owner weight.
__global__ void linearCorrKernel(int nC, const label* __restrict__ ownerStart, const label* __restrict__ losort,
        const label* __restrict__ losortStart, const label* __restrict__ own, const label* __restrict__ nei,
        const scalar* __restrict__ phi, const scalar* __restrict__ w, const scalar* __restrict__ field,
        scalar* __restrict__ corrSource) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x; if (c >= nC) return;
    scalar s = 0;
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f) {                       // c is owner (+)
        const scalar pf = phi[f]; const scalar pos0 = (pf >= 0.0) ? 1.0 : 0.0;
        s += pf * (w[f] - pos0) * (field[own[f]] - field[nei[f]]);
    }
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k) {                     // c is neighbour (-)
        const int f = losort[k]; const scalar pf = phi[f]; const scalar pos0 = (pf >= 0.0) ? 1.0 : 0.0;
        s -= pf * (w[f] - pos0) * (field[own[f]] - field[nei[f]]);
    }
    corrSource[c] = s;
}
void deviceLinearCorr(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& field,
                      DeviceBuffer<scalar>& corrSource) {
    corrSource.resize(dm.nCells);
    linearCorrKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(),
        dm.owner.data(), dm.nei.data(), phiInt.data(), dm.w.data(), field.data(), corrSource.data());
    cudaCheck(cudaGetLastError(), "linearCorr");
}

namespace {
__global__ void setRefKernel(label ref, scalar refValue, scalar* __restrict__ diag, scalar* __restrict__ b) {
    if (blockIdx.x == 0 && threadIdx.x == 0) { b[ref] += diag[ref] * refValue; diag[ref] += diag[ref]; }
}
// adjustPhi: per-face classify (massIn / fixedMassOut / adjustableMassOut) via atomics into sums[0..2].
__global__ void adjustReduceKernel(int n, const label* __restrict__ adj, const scalar* __restrict__ phiB, scalar* __restrict__ sums) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    const scalar f = phiB[i];
    if (f < 0.0) atomicAdd(&sums[0], -f);                 // massIn
    else if (adj[i]) atomicAdd(&sums[2], f);              // adjustableMassOut
    else atomicAdd(&sums[1], f);                          // fixedMassOut
}
__global__ void adjustScaleKernel(int n, const label* __restrict__ adj, scalar massCorr, scalar* __restrict__ phiB) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x; if (i >= n) return;
    if (adj[i] && phiB[i] > 0.0) phiB[i] *= massCorr;     // scale only the adjustable OUTFLOW
}
} // namespace
scalar deviceAdjustPhi(const DeviceBuffer<label>& adjustable, DeviceBuffer<scalar>& phiB) {
    const int n = static_cast<int>(phiB.size());
    if (n == 0) return 1.0;
    DeviceBuffer<scalar> sums(3); cudaCheck(cudaMemset(sums.data(), 0, 3 * sizeof(scalar)), "adjustPhi memset");
    adjustReduceKernel<<<nBlocks(n), TPB>>>(n, adjustable.data(), phiB.data(), sums.data());
    scalar h[3]; cudaCheck(cudaMemcpy(h, sums.data(), 3 * sizeof(scalar), cudaMemcpyDeviceToHost), "adjustPhi D2H");
    const scalar massIn = h[0], fixedOut = h[1], adjOut = h[2];
    scalar massCorr = 1.0;
    if (std::fabs(adjOut) > 1e-300) massCorr = (massIn - fixedOut) / adjOut;       // OF: (massIn-fixedMassOut)/adjustableMassOut
    adjustScaleKernel<<<nBlocks(n), TPB>>>(n, adjustable.data(), massCorr, phiB.data());
    cudaCheck(cudaGetLastError(), "adjustPhi");
    return massCorr;
}
void deviceSetReference(DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& b, label refCell, scalar refValue) {
    if (refCell < 0) return;
    setRefKernel<<<1, 1>>>(refCell, refValue, diag.data(), b.data());
    cudaCheck(cudaGetLastError(), "setReference");
}

void deviceReciprocalV(const DeviceMesh& dm, const DeviceBuffer<scalar>& diagC, DeviceBuffer<scalar>& rAU) {
    const int nC = dm.nCells; rAU.resize(nC);
    reciprocalVKernel<<<nBlocks(nC), TPB>>>(nC, diagC.data(), dm.V.data(), rAU.data());
    cudaCheck(cudaGetLastError(), "reciprocalV");
}
// SIMPLEC rAtU = 1/(1/rAU - H1) = V/rowSum, where rowSum = A*1 = diagA + sum(off-diag) (= 1/rAU - H1).
// OF (pEqn.H:12): rAtU = 1.0/(1.0/rAU - UEqn.H1()) -- NO floor/guard. Faithful: divide by rowSum directly.
__global__ void simplecRAtUKernel(int n, const scalar* __restrict__ V, const scalar* __restrict__ rowSum,
                                  const scalar* __restrict__ diagA, scalar* __restrict__ rAtU) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    (void)diagA;
    if (c < n) rAtU[c] = V[c] / rowSum[c];
}
void deviceSimplecRAtU(const DeviceMesh& dm, const DeviceBuffer<scalar>& rowSum, const DeviceBuffer<scalar>& diagA,
                       DeviceBuffer<scalar>& rAtU) {
    const int nC = dm.nCells; rAtU.resize(nC);
    simplecRAtUKernel<<<nBlocks(nC), TPB>>>(nC, dm.V.data(), rowSum.data(), diagA.data(), rAtU.data());
    cudaCheck(cudaGetLastError(), "simplecRAtU");
}
void deviceVectorFlux(const DeviceMesh& dm, const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy,
                      const DeviceBuffer<scalar>& Uz, DeviceBuffer<scalar>& phiInt) {
    const int nIf = dm.nInternalFaces; phiInt.resize(nIf);
    vectorFluxKernel<<<nBlocks(nIf), TPB>>>(nIf, dm.owner.data(), dm.nei.data(), dm.w.data(), dm.Sfx.data(), dm.Sfy.data(),
                                            dm.Sfz.data(), Ux.data(), Uy.data(), Uz.data(), phiInt.data());
    cudaCheck(cudaGetLastError(), "vectorFlux");
}
void deviceMatrixFluxInternal(const DeviceLduView& A, const DeviceBuffer<scalar>& p, DeviceBuffer<scalar>& fluxInt) {
    fluxInt.resize(A.nInternalFaces);
    matrixFluxKernel<<<nBlocks(A.nInternalFaces), TPB>>>(A.nInternalFaces, A.owner, A.nei, A.upper, A.lower, p.data(), fluxInt.data());
    cudaCheck(cudaGetLastError(), "matrixFlux");
}
void deviceCorrector(const DeviceBuffer<scalar>& HbyA, const DeviceBuffer<scalar>& rAU,
                     const DeviceBuffer<scalar>& gradP, DeviceBuffer<scalar>& U) {
    const int nC = static_cast<int>(HbyA.size()); U.resize(nC);
    correctorKernel<<<nBlocks(nC), TPB>>>(nC, HbyA.data(), rAU.data(), gradP.data(), U.data());
    cudaCheck(cudaGetLastError(), "corrector");
}
void deviceBoundaryFlux(const DeviceMesh& dm, const DeviceBuffer<scalar>& uxb, const DeviceBuffer<scalar>& uyb,
                        const DeviceBuffer<scalar>& uzb, DeviceBuffer<scalar>& phiB) {
    phiB.resize(dm.nBndFaces);
    bndFluxKernel<<<nBlocks(dm.nBndFaces), TPB>>>(dm.nBndFaces, dm.bndGFace.data(), dm.Sfx.data(), dm.Sfy.data(),
                                                  dm.Sfz.data(), uxb.data(), uyb.data(), uzb.data(), phiB.data());
    cudaCheck(cudaGetLastError(), "bndFlux");
}
void deviceRelaxDiag(const DeviceLduView& A, const DeviceMesh& dm, const DeviceBuffer<scalar>& iCbnd, scalar alpha,
                     DeviceBuffer<scalar>& relaxedDiag, DeviceBuffer<scalar>& delta, const scalar* cycSumOff,
                     const scalar* iCmaxMag) {
    relaxedDiag.resize(A.nCells); delta.resize(A.nCells);
    relaxKernel<<<nBlocks(A.nCells), TPB>>>(A.nCells, A.ownerStart, A.losort, A.losortStart, A.upper, A.lower,
                                            dm.bndCellStart.data(), dm.bndPerm.data(), iCbnd.data(), A.diag, alpha,
                                            relaxedDiag.data(), delta.data(), cycSumOff, iCmaxMag);
    cudaCheck(cudaGetLastError(), "relaxDiag");
}
namespace { __global__ void cmptMaxMag3Kernel(int n, const scalar* __restrict__ a, const scalar* __restrict__ b,
                                              const scalar* __restrict__ c, scalar* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = fmax(fabs(a[i]), fmax(fabs(b[i]), fabs(c[i])));
} }
void deviceCmptMaxMag3(const DeviceBuffer<scalar>& a, const DeviceBuffer<scalar>& b, const DeviceBuffer<scalar>& c,
                       DeviceBuffer<scalar>& out) {
    const int n = static_cast<int>(a.size()); out.resize(n);
    cmptMaxMag3Kernel<<<nBlocks(n), TPB>>>(n, a.data(), b.data(), c.data(), out.data());
    cudaCheck(cudaGetLastError(), "cmptMaxMag3");
}

} // namespace brae
