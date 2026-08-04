// device_thermo.cu -- kernels behind deviceThermoUpdate.

#include "device_thermo.cuh"
#include "equation_of_state.cuh"
#include "thermo_model.cuh"
#include "transport_model.cuh"
#include "device_buffer.cuh"
#include "device_boundary.cuh"

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

// rho = rhoPrev + a*(rho - rhoPrev), then rhoPrev = rho. Bounds are re-applied because relaxing toward a
// previous value cannot violate them, but relaxing AWAY from a clamped value can.
__global__
void rhoRelaxK(
    int n,
    scalar a,
    scalar rhoMin,
    scalar rhoMax,
    scalar* __restrict__ rho,
    scalar* __restrict__ rhoPrev)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const scalar blended = rhoPrev[i] + a * (rho[i] - rhoPrev[i]);
    const scalar bounded = fmin(fmax(blended, rhoMin), rhoMax);
    rho[i] = bounded;
    rhoPrev[i] = bounded;
}

__global__
void rhoSeedPrevK(
    int n,
    const scalar* __restrict__ rho,
    scalar* __restrict__ rhoPrev)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    rhoPrev[i] = rho[i];
}

void deviceRhoRelax(
    DeviceThermo& th,
    const ThermoCoeffs& c)
{
    if (th.n == 0) return;
    rhoRelaxK<<<nBlocks(th.n), TPB>>>(
        th.n,
        c.relaxRho,
        c.rhoMin,
        c.rhoMax,
        th.rho.data(),
        th.rhoPrev.data());
    cudaCheck(cudaGetLastError(), "rhoRelax");
}

void deviceRhoSeedPrev(DeviceThermo& th)
{
    if (th.n == 0) return;
    rhoSeedPrevK<<<nBlocks(th.n), TPB>>>(
        th.n,
        th.rho.data(),
        th.rhoPrev.data());
    cudaCheck(cudaGetLastError(), "rhoSeedPrev");
}

// alphat = rho*nut/Prt. Kept in its own kernel rather than folded into thermoUpdateK because nut is
// owned by the turbulence model and is only valid after correctTurbulence(), which runs at the END of the
// outer iteration -- whereas thermoUpdateK runs in the middle of it.
__global__
void alphatK(
    int n,
    scalar Prt,
    const scalar* __restrict__ rho,
    const scalar* __restrict__ nut,
    scalar* __restrict__ alphat)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    alphat[i] = rho[i] * nut[i] / Prt;
}

void deviceAlphat(
    DeviceThermo& th,
    const DeviceBuffer<scalar>& nut,
    const ThermoCoeffs& c)
{
    if (th.n == 0) return;
    if (static_cast<int>(nut.size()) != th.n) return;   // laminar: no nut field, alphat stays zero
    alphatK<<<nBlocks(th.n), TPB>>>(
        th.n,
        c.Prt,
        th.rho.data(),
        nut.data(),
        th.alphat.data());
    cudaCheck(cudaGetLastError(), "alphat");
}

// rho_b = p_b/(R*T_b), face by face. Same EOS as the cell update, so a boundary and its cell agree
// whenever their p and T do.
__global__
void rhoBoundaryK(
    int n,
    ThermoCoeffs c,
    const scalar* __restrict__ pBnd,
    const scalar* __restrict__ heBnd,
    scalar* __restrict__ rhoBnd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const scalar Tb = hConstHeToT(heBnd[i], c);
    const scalar pb = fmax(pBnd[i], c.pMin);
    rhoBnd[i] = fmin(fmax(pb / (c.R * Tb), c.rhoMin), c.rhoMax);
}

void deviceThermoRhoBoundary(
    const DeviceBoundary& dbP,
    const DeviceBuffer<scalar>& p,
    const DeviceBoundary& dbHe,
    const DeviceBuffer<scalar>& he,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& rhoBnd)
{
    if (dbP.n == 0) return;
    DeviceBuffer<scalar> pBnd;
    DeviceBuffer<scalar> heBnd;
    deviceBCValue(dbP, p, pBnd);
    deviceBCValue(dbHe, he, heBnd);
    rhoBnd.resize(dbP.n);
    rhoBoundaryK<<<nBlocks(dbP.n), TPB>>>(
        dbP.n,
        c,
        pBnd.data(),
        heBnd.data(),
        rhoBnd.data());
    cudaCheck(cudaGetLastError(), "rhoBoundary");
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
