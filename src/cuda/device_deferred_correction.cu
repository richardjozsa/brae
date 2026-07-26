// cf GPU offload -- DEFERRED CONVECTION corrections for div(phi,U): the explicit source added to the (pure-upwind)
// momentum matrix to realise the higher-order convection schemes. linearUpwind (grad-reconstruction), linearUpwindV
// (its vector-limited variant), and the LUST linear part (0.75*linear + 0.25*linearUpwind). Split from device_simple.cu
// (the pressure-velocity coupling stays there). Shared decls: device_simple.cuh.
#include "device_simple.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }
}

// linearUpwind deferred correction for div(phi,U_i): the matrix stays pure upwind; the explicit correction
// is the convective transport of grad(U_i)_upwind . (Cf - C_upwind). Per cell: Sum(+/-) phi_f * (grad_upwind . d).
// (boundary faces use pure upwind -> no correction, as in OpenFOAM.) Caller does source -= corrSource.
__global__
void linearUpwindCorrKernel(
    int nC,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    const scalar* __restrict__ dOwnX,
    const scalar* __restrict__ dOwnY,
    const scalar* __restrict__ dOwnZ,
    const scalar* __restrict__ dNeiX,
    const scalar* __restrict__ dNeiY,
    const scalar* __restrict__ dNeiZ,
    scalar* __restrict__ corrSource)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar s = 0;
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f)                       // c is owner (+)
    {
        const scalar pf = phi[f];
        const int up = (pf >= 0) ? own[f] : nei[f];
        const scalar dx = (pf >= 0) ? dOwnX[f] : dNeiX[f], dy = (pf >= 0) ? dOwnY[f] : dNeiY[f], dz = (pf >= 0) ? dOwnZ[f] : dNeiZ[f];
        s += pf * (gx[up]*dx + gy[up]*dy + gz[up]*dz);
    }
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)                     // c is neighbour (-)
    {
        const int f = losort[k];
        const scalar pf = phi[f];
        const int up = (pf >= 0) ? own[f] : nei[f];
        const scalar dx = (pf >= 0) ? dOwnX[f] : dNeiX[f], dy = (pf >= 0) ? dOwnY[f] : dNeiY[f], dz = (pf >= 0) ? dOwnZ[f] : dNeiZ[f];
        s -= pf * (gx[up]*dx + gy[up]*dy + gz[up]*dz);
    }
    corrSource[c] = s;
}
void deviceLinearUpwindCorr(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    DeviceBuffer<scalar>& corrSource)
{
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
__global__
void linearUpwindVFaceKernel(
    int nIf,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ w,
    const scalar* __restrict__ dOwnX,
    const scalar* __restrict__ dOwnY,
    const scalar* __restrict__ dOwnZ,
    const scalar* __restrict__ dNeiX,
    const scalar* __restrict__ dNeiY,
    const scalar* __restrict__ dNeiZ,
    const scalar* __restrict__ g0x,
    const scalar* __restrict__ g0y,
    const scalar* __restrict__ g0z,
    const scalar* __restrict__ g1x,
    const scalar* __restrict__ g1y,
    const scalar* __restrict__ g1z,
    const scalar* __restrict__ g2x,
    const scalar* __restrict__ g2y,
    const scalar* __restrict__ g2z,
    const scalar* __restrict__ U0,
    const scalar* __restrict__ U1,
    const scalar* __restrict__ U2,
    scalar* __restrict__ fcX,
    scalar* __restrict__ fcY,
    scalar* __restrict__ fcZ)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    const scalar pf = phi[f];
    const bool pos = (pf >= 0.0);
    const int o = own[f], n = nei[f];
    const int up = pos ? o : n;
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
    if (sfCorrs > 0.0)
    {
        if (maxCorrs < 0.0)
        {
            cx = 0.0;
            cy = 0.0;
            cz = 0.0;
        }
        else if (sfCorrs > maxCorrs)
        {
            const scalar r = maxCorrs / (sfCorrs + 1.0e-300);
            cx *= r;
            cy *= r;
            cz *= r;
        }
    }
    fcX[f] = cx;
    fcY[f] = cy;
    fcZ[f] = cz;
}
// div(phi * faceCorr) gather (per cell), same owner(+)/losort(-) sum as linearUpwindCorrKernel but on a precomputed face field.
__global__
void divFaceCorrKernel(
    int nC,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ fc,
    scalar* __restrict__ corrSource)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar s = 0;
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f)
        s += phi[f] * fc[f];
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)
    {
        const int f = losort[k];
        s -= phi[f] * fc[f];
    }
    corrSource[c] = s;
}
void deviceLinearUpwindVCorr(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>* gUx,
    const DeviceBuffer<scalar>* gUy,
    const DeviceBuffer<scalar>* gUz,
    const DeviceBuffer<scalar>& U0,
    const DeviceBuffer<scalar>& U1,
    const DeviceBuffer<scalar>& U2,
    DeviceBuffer<scalar>& corrX,
    DeviceBuffer<scalar>& corrY,
    DeviceBuffer<scalar>& corrZ)
{
    const int nIf = dm.nInternalFaces;
    DeviceBuffer<scalar> fcX(nIf), fcY(nIf), fcZ(nIf);
    linearUpwindVFaceKernel<<<nBlocks(nIf), TPB>>>(nIf, dm.owner.data(), dm.nei.data(), phiInt.data(), dm.w.data(),
        dm.dOwnX.data(), dm.dOwnY.data(), dm.dOwnZ.data(), dm.dNeiX.data(), dm.dNeiY.data(), dm.dNeiZ.data(),
        gUx[0].data(), gUy[0].data(), gUz[0].data(), gUx[1].data(), gUy[1].data(), gUz[1].data(),
        gUx[2].data(), gUy[2].data(), gUz[2].data(), U0.data(), U1.data(), U2.data(),
        fcX.data(), fcY.data(), fcZ.data());
    corrX.resize(dm.nCells);
    corrY.resize(dm.nCells);
    corrZ.resize(dm.nCells);
    divFaceCorrKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(), phiInt.data(), fcX.data(), corrX.data());
    divFaceCorrKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(), phiInt.data(), fcY.data(), corrY.data());
    divFaceCorrKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(), phiInt.data(), fcZ.data(), corrZ.data());
    cudaCheck(cudaGetLastError(), "linearUpwindVCorr");
}

// LUST linear-part deferred correction: div(phi*(linear_face - upwind_face)) = div(phi*(w-pos0)*(field[P]-field[N])).
// Same divergence gather as linearUpwindCorrKernel. OF LUST = 0.75*linear + 0.25*linearUpwind, so the caller adds
// 0.75*this + 0.25*linearUpwindCorr. w = owner linear weight (dm.w), pos0 = (phi>=0) = the upwind owner weight.
__global__
void linearCorrKernel(
    int nC,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ w,
    const scalar* __restrict__ field,
    scalar* __restrict__ corrSource)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar s = 0;
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f)                       // c is owner (+)
    {
        const scalar pf = phi[f];
        const scalar pos0 = (pf >= 0.0) ? 1.0 : 0.0;
        s += pf * (w[f] - pos0) * (field[own[f]] - field[nei[f]]);
    }
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)                     // c is neighbour (-)
    {
        const int f = losort[k];
        const scalar pf = phi[f];
        const scalar pos0 = (pf >= 0.0) ? 1.0 : 0.0;
        s -= pf * (w[f] - pos0) * (field[own[f]] - field[nei[f]]);
    }
    corrSource[c] = s;
}
void deviceLinearCorr(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& field,
    DeviceBuffer<scalar>& corrSource)
{
    corrSource.resize(dm.nCells);
    linearCorrKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(),
        dm.owner.data(), dm.nei.data(), phiInt.data(), dm.w.data(), field.data(), corrSource.data());
    cudaCheck(cudaGetLastError(), "linearCorr");
}

} // namespace brae
