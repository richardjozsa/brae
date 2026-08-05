#pragma once
// Shared distributed-interface helpers used by BOTH parallel_device_simple.cuh (the momentum/pressure step)
// and parallel_device_turbulence.cuh (the k/eps scalar transport). Extracted here so neither includes the
// other -- breaking the simple<->turbulence include cycle. Just the two processor-interface primitives the
// scalar transport needs: the momentum/scalar convection+diffusion interface coeff, and the relax off-diagonal
// sum. The other interface helpers (laplacian, linearUpwind, non-orth, H, flux) stay in parallel_device_simple.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_halo.cuh"
#include <vector>

namespace brae {
namespace detail {

// L1: the processor-interface coeffs of the momentum matrix M = div(phi,U) - laplacian(nuEff,U). Mirrors host
// momentumDistributed, per cut face f (upwind weight w = phi>=0 ? 1 : 0, coeff = nuEffF*magSf*procDelta):
//   diag[faceCell[f]] += w*phi[f] + coeff[f]              (outflow convection + diffusion, atomic)
//   ifCoeff[f]         = -(1 - w)*phi[f] + coeff[f]        (inflow convection + diffusion off-diagonal)
__global__
void momentumInterfaceKernel(
    const label*  __restrict__ faceCells,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ coeff,
    scalar*       __restrict__ diag,
    scalar*       __restrict__ ifCoeff,
    int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    const scalar ph = phi[f];
    const scalar w = (ph >= 0.0) ? 1.0 : 0.0;
    ifCoeff[f] = -(1.0 - w) * ph + coeff[f];
    atomicAdd(&diag[faceCells[f]], w * ph + coeff[f]);
}

// Per-cell sum of |processor off-diagonal|, for fvMatrix::relax's diagonal-dominance term. Host
// parallelRelaxMatrix adds |interfaceCoeffs| into sumOff; on device this feeds deviceRelaxDiag's cycSumOff
// hook (the same role the cyclic interface's off-diagonal sum plays).
__global__
void offDiagSumKernel(
    const label*  __restrict__ faceCells,
    const scalar* __restrict__ ifCoeff,
    scalar*       __restrict__ sumOff,
    int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    atomicAdd(&sumOff[faceCells[f]], fabs(ifCoeff[f]));
}

} // namespace detail

// Assemble the momentum matrix's processor coupling on `halo`'s interfaces: fold the convection+diffusion
// contribution into `diag` and produce `ifCoeff[i]` (per interface, for deviceParallelAmul). `phiF[i]` is the
// processor-face flux and `coeffGeo[i] = nuEffF*magSf*procDelta` of interface i (same order as the halo).
inline void deviceMomentumInterface(
    const DeviceHalo& halo,
    const std::vector<DeviceBuffer<label>>& faceCells,
    const std::vector<DeviceBuffer<scalar>>& phiF,
    const std::vector<DeviceBuffer<scalar>>& coeffGeo,
    DeviceBuffer<scalar>& diag,
    std::vector<DeviceBuffer<scalar>>& ifCoeff,
    cudaStream_t stream = cudaStreamPerThread)
{
    constexpr int TPB = 128;
    const int nI = halo.nInterfaces();
    ifCoeff.resize(nI);
    for (int i = 0; i < nI; ++i)
    {
        const int n = static_cast<int>(halo.size(i));
        if (n <= 0) continue;
        ifCoeff[i].resize(n);
        detail::momentumInterfaceKernel<<<(n + TPB - 1) / TPB, TPB, 0, stream>>>(
            faceCells[i].data(),
            phiF[i].data(),
            coeffGeo[i].data(),
            diag.data(),
            ifCoeff[i].data(),
            n);
    }
}

// sumOff[c] += sum over this rank's interface faces owned by c of |ifCoeff|. Pass the result to
// deviceRelaxDiag's cycSumOff so the processor interface counts toward diagonal dominance, matching host
// parallelRelaxMatrix. `sumOff` must be zeroed by the caller (size nCells).
inline void deviceInterfaceOffDiagSum(
    const DeviceHalo& halo,
    const std::vector<DeviceBuffer<label>>& faceCells,
    const std::vector<DeviceBuffer<scalar>>& ifCoeff,
    DeviceBuffer<scalar>& sumOff,
    cudaStream_t stream = cudaStreamPerThread)
{
    constexpr int TPB = 128;
    for (int i = 0; i < halo.nInterfaces(); ++i)
    {
        const int n = static_cast<int>(halo.size(i));
        if (n <= 0) continue;
        detail::offDiagSumKernel<<<(n + TPB - 1) / TPB, TPB, 0, stream>>>(
            faceCells[i].data(),
            ifCoeff[i].data(),
            sumOff.data(),
            n);
    }
}

} // namespace brae
