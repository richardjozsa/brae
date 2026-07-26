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

// source[i] += V[i]*(b0*old[i] - b00*old2[i])   with b0 = rDeltaT*rho*coefft0, b00 = rDeltaT*rho*coefft00.
// old2 may be null (Euler / bootstrap): the coefft00 term is then structurally absent.
__global__ void ddtSourceKernel(
    const scalar* __restrict__ V, int n, scalar b0, scalar b00,
    const scalar* __restrict__ old, const scalar* __restrict__ old2, scalar* __restrict__ source)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    scalar s = b0 * old[i];
    if (old2) s -= b00 * old2[i];
    source[i] += V[i] * s;
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
    const DeviceBuffer<scalar>& psiOld, const DeviceBuffer<scalar>& psiOld2, DeviceBuffer<scalar>& source)
{
    if (!c.active) return;
    const int n = static_cast<int>(V.size());
    const scalar b0  = c.rDeltaT * rho * c.coefft0;
    const scalar b00 = c.rDeltaT * rho * c.coefft00;
    const scalar* old2 = psiOld2.size() ? psiOld2.data() : nullptr;
    ddtSourceKernel<<<nBlocks(n), TPB>>>(V.data(), n, b0, b00, psiOld.data(), old2, source.data());
    cudaCheck(cudaGetLastError(), "fvmDdtSource");
}

void deviceFvmDdt(
    const DeviceBuffer<scalar>& V, const DdtCoeffs& c, scalar rho,
    const DeviceBuffer<scalar>& psiOld, const DeviceBuffer<scalar>& psiOld2,
    DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& source)
{
    deviceFvmDdtDiag(V, c, rho, diag);
    deviceFvmDdtSource(V, c, rho, psiOld, psiOld2, source);
}

}  // namespace brae
