#include "device_force.cuh"
#include "device_divdevreff.cuh"
#include "device_kepsilon.cuh"
#include <cuda_runtime.h>

namespace brae {
namespace {

constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

__global__ void forceKernel(
    int n,
    const label* __restrict__ bnd,
    const scalar* __restrict__ cfx,
    const scalar* __restrict__ cfy,
    const scalar* __restrict__ cfz,
    const label* __restrict__ bndGFace,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const scalar* __restrict__ pB,
    const scalar* __restrict__ nutBnd,
    const scalar* __restrict__ gradB,
    int nB,
    scalar nu,
    scalar rhoRef,
    scalar pRef,
    vector CofR,
    scalar* __restrict__ out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int bi = bnd[i];
    const int gf = bndGFace[bi];
    const vector Sf{Sfx[gf], Sfy[gf], Sfz[gf]};
    const vector fP = (rhoRef * (pB[bi] - pRef)) * Sf;

    // This is devTwoSymm(gradUBoundary), the same tensor and storage order as the host wallForces path.
    const scalar gxx = gradB[0*nB + bi], gxy = gradB[1*nB + bi], gxz = gradB[2*nB + bi];
    const scalar gyx = gradB[3*nB + bi], gyy = gradB[4*nB + bi], gyz = gradB[5*nB + bi];
    const scalar gzx = gradB[6*nB + bi], gzy = gradB[7*nB + bi], gzz = gradB[8*nB + bi];
    const scalar c = (scalar(2.0)/scalar(3.0)) * (gxx + gyy + gzz);
    const scalar sxx = gxx + gxx - c, sxy = gxy + gyx, sxz = gxz + gzx;
    const scalar syx = gyx + gxy, syy = gyy + gyy - c, syz = gyz + gzy;
    const scalar szx = gzx + gxz, szy = gzy + gyz, szz = gzz + gzz - c;
    const scalar scale = -rhoRef * (nu + nutBnd[bi]);
    const tensor devReff{scale*sxx, scale*sxy, scale*sxz,
                         scale*syx, scale*syy, scale*syz,
                         scale*szx, scale*szy, scale*szz};
    const vector fV = dot(Sf, devReff);
    const vector Md{cfx[i] - CofR.x, cfy[i] - CofR.y, cfz[i] - CofR.z};
    const vector mP = cross(Md, fP), mV = cross(Md, fV);

    atomicAdd(&out[0], fP.x);  atomicAdd(&out[1], fP.y);  atomicAdd(&out[2], fP.z);
    atomicAdd(&out[3], fV.x);  atomicAdd(&out[4], fV.y);  atomicAdd(&out[5], fV.z);
    atomicAdd(&out[6], mP.x);  atomicAdd(&out[7], mP.y);  atomicAdd(&out[8], mP.z);
    atomicAdd(&out[9], mV.x);  atomicAdd(&out[10], mV.y); atomicAdd(&out[11], mV.z);
}

} // namespace

DeviceForceResult deviceWallForceReduce(
    const DeviceMesh& dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBoundary& dbP,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& p,
    const DeviceBuffer<scalar>& nutBnd,
    const DeviceForceSelection& selection,
    scalar nu,
    scalar rhoRef,
    scalar pRef,
    const vector& CofR,
    DeviceCyclic* cyc,
    DeviceAMI* ami)
{
    DeviceForceResult result;
    if (selection.n == 0) return result;
    if (nutBnd.size() != static_cast<std::size_t>(dm.nBndFaces))
        throw std::runtime_error("brae forceCoeffs: boundary nut size does not match boundary geometry");

    // The SIMPLE predictor's gradient is not available here: wall sampling happens after the pressure
    // correction and the device solver retains only the solved U/p buffers, not a post-correction gradU.
    // Recompute the full-cell gradient so the boundary correction is evaluated from the exact sampled U.
    DeviceBuffer<scalar> gradU, gradB, pB;
    deviceGradU(dm, dbU, Ux, Uy, Uz, gradU, ami, cyc);
    deviceBoundaryGradU(dm, dbU, Ux, Uy, Uz, gradU, gradB);
    deviceBCValue(dbP, p, pB);
    DeviceBuffer<scalar> out(std::vector<scalar>(12, scalar(0)));
    forceKernel<<<nBlocks(selection.n), TPB>>>(
        selection.n, selection.boundaryIndex.data(), selection.cfx.data(), selection.cfy.data(), selection.cfz.data(),
        dm.bndGFace.data(), dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(), pB.data(), nutBnd.data(), gradB.data(),
        dm.nBndFaces, nu, rhoRef, pRef, CofR, out.data());
    cudaCheck(cudaGetLastError(), "forceCoeffs reduction");
    // This is one blocking D2H of twelve reduced scalars, not a face-field transfer. The host needs these
    // values immediately to append the history row; retaining them on the device would only move the sync
    // into the writer and would not remove it.
    const std::vector<scalar> h = out.host();
    result.pressure = {h[0], h[1], h[2]};
    result.viscous  = {h[3], h[4], h[5]};
    result.momentP  = {h[6], h[7], h[8]};
    result.momentV  = {h[9], h[10], h[11]};
    return result;
}

} // namespace brae
