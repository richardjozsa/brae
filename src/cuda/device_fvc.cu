// cf GPU offload (G3): fvc explicit operators on device. interpolate is per-internal-face; div and
// gaussGrad are per-cell gathers (internal owner/neighbour faces via ownerStart/losort, boundary faces via
// bndCellStart), race-free, deterministic, matching the CPU fvc to machine precision.
#include "device_mesh.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }


__global__
void interpKernel(
    int nIf,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ w,
    const scalar* __restrict__ vol,
    scalar* __restrict__ sf)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f < nIf) sf[f] = w[f] * vol[own[f]] + (1.0 - w[f]) * vol[nei[f]];
}


__global__
void divKernel(
    int nC,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const scalar* __restrict__ phiInt,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const scalar* __restrict__ bval,
    const scalar* __restrict__ V,
    scalar* __restrict__ d)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar s = 0.0;
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f)
        s += phiInt[f];              // +owner internal
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)
        s -= phiInt[losort[k]];    // -neighbour internal
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)
        s += bval[bndPerm[k]];   // +boundary
    d[c] = s / V[c];
}


__global__
void gradKernel(
    int nC,
    const label* __restrict__ own,
    const label* __restrict__ nei,
    const scalar* __restrict__ w,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const scalar* __restrict__ vol,
    const label* __restrict__ ownerStart,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const label* __restrict__ bndGFace,
    const scalar* __restrict__ bval,
    const scalar* __restrict__ V,
    scalar* __restrict__ gx,
    scalar* __restrict__ gy,
    scalar* __restrict__ gz)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    scalar sx = 0.0, sy = 0.0, sz = 0.0;
    for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f)   // +owner internal
    {
        const scalar pf = w[f] * vol[own[f]] + (1.0 - w[f]) * vol[nei[f]];
        sx += Sfx[f] * pf;
        sy += Sfy[f] * pf;
        sz += Sfz[f] * pf;
    }
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k)   // -neighbour internal
    {
        const int f = losort[k];
        const scalar pf = w[f] * vol[own[f]] + (1.0 - w[f]) * vol[nei[f]];
        sx -= Sfx[f] * pf;
        sy -= Sfy[f] * pf;
        sz -= Sfz[f] * pf;
    }
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k)   // +boundary
    {
        const int kk = bndPerm[k];
        const int f = bndGFace[kk];
        const scalar pv = bval[kk];
        sx += Sfx[f] * pv;
        sy += Sfy[f] * pv;
        sz += Sfz[f] * pv;
    }
    gx[c] = sx / V[c];
    gy[c] = sy / V[c];
    gz[c] = sz / V[c];
}
} // namespace


void deviceInterpolate(const DeviceMesh& dm, const DeviceBuffer<scalar>& vol, DeviceBuffer<scalar>& sfInt)
{
    sfInt.resize(dm.nInternalFaces);
    interpKernel<<<nBlocks(dm.nInternalFaces), TPB>>>(dm.nInternalFaces, dm.owner.data(), dm.nei.data(), dm.w.data(), vol.data(), sfInt.data());
    cudaCheck(cudaGetLastError(), "interp");
}


void deviceDiv(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt, const DeviceBuffer<scalar>& bval, DeviceBuffer<scalar>& d)
{
    d.resize(dm.nCells);
    divKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(),
                                           phiInt.data(), dm.bndCellStart.data(), dm.bndPerm.data(), bval.data(), dm.V.data(), d.data());
    cudaCheck(cudaGetLastError(), "div");
}


void deviceGaussGrad(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& vol,
    const DeviceBuffer<scalar>& bval,
    DeviceBuffer<scalar>& gx,
    DeviceBuffer<scalar>& gy,
    DeviceBuffer<scalar>& gz)
{
    gx.resize(dm.nCells);
    gy.resize(dm.nCells);
    gz.resize(dm.nCells);
    gradKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, dm.owner.data(), dm.nei.data(), dm.w.data(),
                                            dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(), vol.data(),
                                            dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(),
                                            dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndGFace.data(), bval.data(),
                                            dm.V.data(), gx.data(), gy.data(), gz.data());
    cudaCheck(cudaGetLastError(), "gaussGrad");
}


namespace {
// OF cellLimitedGrad<minmod>::limitFaceCmpt: r = maxDelta/extrapolate (or minDelta/extrapolate); minmod limiter
// min(r,1). |extrapolate|~0 -> face imposes no constraint (returns 1). SMALL = Foam::SMALL (1e-15, double).
__device__ __forceinline__
scalar limFace(scalar maxD, scalar minD, scalar ex)
{
    if (ex >  1e-15) return fmin(maxD / ex, 1.0);
    if (ex < -1e-15) return fmin(minD / ex, 1.0);
    return 1.0;
}


__global__
void cellLimitGradKernel(
    int nC,
    scalar k,
    const scalar* __restrict__ U,
    const scalar* __restrict__ Ubnd,
    const label* __restrict__ ownerStart,
    const label* __restrict__ nei,
    const label* __restrict__ losort,
    const label* __restrict__ losortStart,
    const label* __restrict__ owner,
    const label* __restrict__ bndCellStart,
    const label* __restrict__ bndPerm,
    const scalar* __restrict__ dOwnX,
    const scalar* __restrict__ dOwnY,
    const scalar* __restrict__ dOwnZ,
    const scalar* __restrict__ dNeiX,
    const scalar* __restrict__ dNeiY,
    const scalar* __restrict__ dNeiZ,
    const scalar* __restrict__ dBndX,
    const scalar* __restrict__ dBndY,
    const scalar* __restrict__ dBndZ,
    scalar* __restrict__ gx,
    scalar* __restrict__ gy,
    scalar* __restrict__ gz)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;

    const scalar uc = U[c];
    scalar maxD = 0.0, minD = 0.0;   // incl. the cell itself (U[c]-U[c]=0)
    for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
    {
        const scalar d = U[nei[f]] - uc;
        maxD = fmax(maxD,d);
        minD = fmin(minD,d);
    }
    for (int j = losortStart[c]; j < losortStart[c+1]; ++j)
    {
        const scalar d = U[owner[losort[j]]] - uc;
        maxD = fmax(maxD,d);
        minD = fmin(minD,d);
    }
    for (int j = bndCellStart[c]; j < bndCellStart[c+1]; ++j)
    {
        const scalar d = Ubnd[bndPerm[j]] - uc;
        maxD = fmax(maxD,d);
        minD = fmin(minD,d);
    }
    if (k < 1.0)   // OF k<1 widening
    {
        const scalar wdn = (1.0/k - 1.0)*(maxD - minD);
        maxD += wdn;
        minD -= wdn;
    }
    const scalar gcx = gx[c], gcy = gy[c], gcz = gz[c];
    scalar lim = 1.0;
    for (int f = ownerStart[c]; f < ownerStart[c+1]; ++f)
        lim = fmin(lim, limFace(maxD, minD, dOwnX[f]*gcx + dOwnY[f]*gcy + dOwnZ[f]*gcz));
    for (int j = losortStart[c]; j < losortStart[c+1]; ++j)
    {
        const int f = losort[j];
        lim = fmin(lim, limFace(maxD, minD, dNeiX[f]*gcx + dNeiY[f]*gcy + dNeiZ[f]*gcz));
    }
    for (int j = bndCellStart[c]; j < bndCellStart[c+1]; ++j)
    {
        const int bk = bndPerm[j];
        lim = fmin(lim, limFace(maxD, minD, dBndX[bk]*gcx + dBndY[bk]*gcy + dBndZ[bk]*gcz));
    }
    gx[c] = gcx*lim;
    gy[c] = gcy*lim;
    gz[c] = gcz*lim;
}
} // namespace


// cellLimited Gauss linear <k> (OF cellLimitedGrad<vector,minmod>), applied to ONE component's gradient: scale grad
// so the reconstructed face value C[c] + (Cf-C).grad stays within the cell's face-neighbour min/max. Per-component
// (caller loops the 3 U components). k=1 = full limiting (motorBike); k<1 widens the bounds (less limiting).
void deviceCellLimitGrad(
    const DeviceMesh& dm,
    const DeviceBuffer<scalar>& U,
    const DeviceBuffer<scalar>& Ubnd,
    DeviceBuffer<scalar>& gx,
    DeviceBuffer<scalar>& gy,
    DeviceBuffer<scalar>& gz,
    scalar k)
{
    cellLimitGradKernel<<<nBlocks(dm.nCells), TPB>>>(dm.nCells, k, U.data(), Ubnd.data(),
        dm.ownerStart.data(), dm.nei.data(), dm.losort.data(), dm.losortStart.data(), dm.owner.data(),
        dm.bndCellStart.data(), dm.bndPerm.data(),
        dm.dOwnX.data(), dm.dOwnY.data(), dm.dOwnZ.data(), dm.dNeiX.data(), dm.dNeiY.data(), dm.dNeiZ.data(),
        dm.dBndX.data(), dm.dBndY.data(), dm.dBndZ.data(), gx.data(), gy.data(), gz.data());
    cudaCheck(cudaGetLastError(), "cellLimitGrad");
}

} // namespace brae
