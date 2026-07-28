// Implicit transient ddt kernels -- see device_ddt.cuh for the OF-2412 correspondence + source refs.
#include "device_ddt.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

// diag[i] += a*V[i]   with a = coefft*rDeltaT*rho.
__global__ void ddtDiagKernel(const scalar* __restrict__ V, int n, scalar a, scalar* __restrict__ diag)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    diag[i] += a * V[i];
}

// backward/Euler : source[i] += V[i]*(b0*old[i] - b00*old2[i])   (b0=rDeltaT*rho*coefft0, b00=rDeltaT*rho*coefft00)
// CrankNicolson  : source[i] += V[i]*(b0*old[i] + dc*ddt0[i])    (dc=rho*ocCoeff; the ddt0 term REPLACES the old2 term)
// old2/ddt0 may be null (Euler / bootstrap): the corresponding term is then structurally absent.
__global__ void ddtSourceKernel(
    const scalar* __restrict__ V, int n, scalar b0, scalar b00, scalar dc,
    const scalar* __restrict__ old, const scalar* __restrict__ old2, const scalar* __restrict__ ddt0,
    scalar* __restrict__ source)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    scalar s = b0 * old[i];
    if (ddt0)      s += dc * ddt0[i];       // CrankNicolson stored-old-ddt term
    else if (old2) s -= b00 * old2[i];      // backward second-old-level term
    source[i] += V[i] * s;
}

// CrankNicolson ddt0 recurrence: ddt0[i] = a*(old[i] - old2[i]) - oc*ddt0[i]   with a = coefft*rDeltaT0. In place.
__global__ void ddt0UpdateKernel(
    int n, scalar a, scalar oc, const scalar* __restrict__ old, const scalar* __restrict__ old2, scalar* __restrict__ ddt0)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    ddt0[i] = a * (old[i] - old2[i]) - oc * ddt0[i];
}
}  // namespace

void deviceFvmDdtDiag(
    const DeviceBuffer<scalar>& V, const DdtCoeffs& c, scalar rho, DeviceBuffer<scalar>& diag)
{
    if (!c.active) return;                                   // steadyState / bootstrap -> no-op
    const int n = static_cast<int>(V.size());
    ddtDiagKernel<<<nBlocks(n), TPB>>>(V.data(), n, c.coefft * c.rDeltaT * rho, diag.data());
    cudaCheck(cudaGetLastError(), "fvmDdtDiag");
}

void deviceFvmDdtSource(
    const DeviceBuffer<scalar>& V, const DdtCoeffs& c, scalar rho,
    const DeviceBuffer<scalar>& psiOld, const DeviceBuffer<scalar>& psiOld2, DeviceBuffer<scalar>& source,
    const DeviceBuffer<scalar>* ddt0)
{
    if (!c.active) return;
    const int n = static_cast<int>(V.size());
    const scalar b0  = c.rDeltaT * rho * c.coefft0;
    const scalar b00 = c.rDeltaT * rho * c.coefft00;
    const scalar dc  = rho * c.ocCoeff;                                    // CrankNicolson ddt0 weight
    const scalar* old2p = psiOld2.size() ? psiOld2.data() : nullptr;
    const scalar* ddt0p = (c.cn && ddt0 && ddt0->size()) ? ddt0->data() : nullptr;   // CN uses ddt0, NOT old2
    ddtSourceKernel<<<nBlocks(n), TPB>>>(V.data(), n, b0, b00, dc, psiOld.data(), old2p, ddt0p, source.data());
    cudaCheck(cudaGetLastError(), "fvmDdtSource");
}

void deviceFvmDdtUpdateDdt0(
    const DdtCoeffs& c, const DeviceBuffer<scalar>& psiOld, const DeviceBuffer<scalar>& psiOld2, DeviceBuffer<scalar>& ddt0)
{
    if (!c.cn || !psiOld2.size() || !ddt0.size()) return;                  // steady/Euler/backward/first-step -> ddt0 untouched (stays 0)
    const int n = static_cast<int>(psiOld.size());
    ddt0UpdateKernel<<<nBlocks(n), TPB>>>(n, c.coefft0dd * c.rDeltaT0, c.ocCoeff, psiOld.data(), psiOld2.data(), ddt0.data());
    cudaCheck(cudaGetLastError(), "fvmDdtUpdateDdt0");
}

void deviceFvmDdt(
    const DeviceBuffer<scalar>& V, const DdtCoeffs& c, scalar rho,
    const DeviceBuffer<scalar>& psiOld, const DeviceBuffer<scalar>& psiOld2,
    DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& source, const DeviceBuffer<scalar>* ddt0)
{
    deviceFvmDdtDiag(V, c, rho, diag);
    deviceFvmDdtSource(V, c, rho, psiOld, psiOld2, source, ddt0);
}

}  // namespace brae
