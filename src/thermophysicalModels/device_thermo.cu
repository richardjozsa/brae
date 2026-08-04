// device_thermo.cu -- kernels behind deviceThermoUpdate.

#include "device_thermo.cuh"
#include "equation_of_state.cuh"
#include "thermo_model.cuh"
#include "transport_model.cuh"
#include "device_buffer.cuh"

namespace brae {

namespace {

constexpr int TPB = 256;

inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

}   // namespace

// One pass over the cells: invert he to get T, then evaluate every T-dependent property from it.
// Fused rather than split per property because each one is a handful of flops on a value already in a
// register -- splitting would re-read T from global memory four times for no benefit.
__global__
void thermoUpdateK(
    int n,
    ThermoCoeffs c,
    const scalar* __restrict__ p,
    const scalar* __restrict__ he,
    scalar* __restrict__ T,
    scalar* __restrict__ rho,
    scalar* __restrict__ psi,
    scalar* __restrict__ mu,
    scalar* __restrict__ alpha)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const scalar Ti = hConstHeToT(he[i], c);
    T[i] = Ti;

    // Bound p before it reaches the EOS. An unbounded iterate can go negative mid-solve, and
    // rho = p/(R T) would then hand a negative density to the next momentum predictor.
    const scalar pi = fmax(p[i], c.pMin);

    const scalar psii = perfectGasPsi(Ti, c);
    psi[i] = psii;
    rho[i] = fmin(fmax(psii * pi, c.rhoMin), c.rhoMax);

    const scalar mui = transportMu(Ti, c);
    mu[i] = mui;
    alpha[i] = transportAlpha(mui, c);
}

// he from T, used once at startup: cases ship a T field, the energy equation wants he.
__global__
void heFromTK(
    int n,
    ThermoCoeffs c,
    const scalar* __restrict__ T,
    scalar* __restrict__ he)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    he[i] = hConstTToHe(T[i], c);
}

void deviceThermoUpdate(
    DeviceThermo& th,
    const DeviceBuffer<scalar>& p,
    const ThermoCoeffs& c)
{
    if (th.n == 0) return;
    thermoUpdateK<<<nBlocks(th.n), TPB>>>(
        th.n,
        c,
        p.data(),
        th.he.data(),
        th.T.data(),
        th.rho.data(),
        th.psi.data(),
        th.mu.data(),
        th.alpha.data());
    cudaCheck(cudaGetLastError(), "thermoUpdate");
}

void deviceThermoHeFromT(
    DeviceThermo& th,
    const ThermoCoeffs& c)
{
    if (th.n == 0) return;
    heFromTK<<<nBlocks(th.n), TPB>>>(
        th.n,
        c,
        th.T.data(),
        th.he.data());
    cudaCheck(cudaGetLastError(), "heFromT");
}

} // namespace brae
