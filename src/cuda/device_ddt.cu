// Implicit transient ddt kernel -- see device_ddt.cuh for the OF-2412 correspondence + source refs.
#include "device_ddt.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

// diag[i] += a*V[i];  source[i] += V[i]*(b0*old[i] - b00*old2[i])   with a=coefft*rDeltaT*rho, b0=rDeltaT*rho*coefft0,
// b00=rDeltaT*rho*coefft00. old2 may be null (Euler): the coefft00 term is then structurally absent (b00 folds out).
__global__ void fvmDdtKernel(
    const scalar* __restrict__ V, int n,
    scalar a, scalar b0, scalar b00,
    const scalar* __restrict__ old,
    const scalar* __restrict__ old2,
    scalar* __restrict__ diag,
    scalar* __restrict__ source)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const scalar v = V[i];
    diag[i] += a * v;
    scalar s = b0 * old[i];
    if (old2) s -= b00 * old2[i];
    source[i] += v * s;
}
}  // namespace

void deviceFvmDdt(
    const DeviceBuffer<scalar>& V,
    const DdtCoeffs&            c,
    scalar                      rho,
    const DeviceBuffer<scalar>& psiOld,
    const DeviceBuffer<scalar>& psiOld2,
    DeviceBuffer<scalar>&       diag,
    DeviceBuffer<scalar>&       source)
{
    if (!c.active) return;                                   // steadyState / bootstrap -> ddt is a no-op
    const int n = static_cast<int>(V.size());
    const scalar a   = c.coefft   * c.rDeltaT * rho;
    const scalar b0  = c.rDeltaT * rho * c.coefft0;
    const scalar b00 = c.rDeltaT * rho * c.coefft00;
    const scalar* old2 = psiOld2.size() ? psiOld2.data() : nullptr;
    fvmDdtKernel<<<nBlocks(n), TPB>>>(V.data(), n, a, b0, b00, psiOld.data(), old2, diag.data(), source.data());
    cudaCheck(cudaGetLastError(), "fvmDdt");
}

}  // namespace brae
