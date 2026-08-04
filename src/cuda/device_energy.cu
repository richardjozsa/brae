// device_energy.cu -- alphaEff assembly and the he solve.

#include "device_energy.cuh"
#include "device_scalar_transport.cuh"
#include "thermo_model.cuh"
#include <stdexcept>

namespace brae {

// Boundary he from boundary T. Linear for hConst, so bcType is untouched and only the value moves.
__global__
void heBndFromTK(
    int n,
    ThermoCoeffs c,
    const scalar* __restrict__ refT,
    scalar* __restrict__ refHe)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    refHe[i] = hConstTToHe(refT[i], c);
}

void deviceEnergyBoundaryFromT(
    const DeviceBoundary& dbT,
    const ThermoCoeffs& c,
    DeviceBoundary& dbHe)
{
    if (dbT.n == 0) return;
    if (dbHe.n != dbT.n)
    {
        throw std::runtime_error(
            "brae: deviceEnergyBoundaryFromT needs the he boundary to share the T boundary's structure.");
    }
    dbHe.refValue.resize(dbT.n);
    heBndFromTK<<<nBlocks(dbT.n), TPB>>>(
        dbT.n,
        c,
        dbT.refValue.data(),
        dbHe.refValue.data());
    cudaCheck(cudaGetLastError(), "heBndFromT");
}

// alphaEff = alpha + alphat. Separate kernel rather than folded into thermoUpdateK because alphat is
// owned by the turbulence model, which runs after the thermo update, not before it.
__global__
void alphaEffK(
    int n,
    const scalar* __restrict__ alpha,
    const scalar* __restrict__ alphat,
    scalar* __restrict__ alphaEff)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    alphaEff[i] = alpha[i] + alphat[i];
}

void deviceAlphaEff(
    const DeviceThermo& th,
    DeviceBuffer<scalar>& alphaEff)
{
    if (th.n == 0) return;
    alphaEff.resize(th.n);
    alphaEffK<<<nBlocks(th.n), TPB>>>(
        th.n,
        th.alpha.data(),
        th.alphat.data(),
        alphaEff.data());
    cudaCheck(cudaGetLastError(), "alphaEff");
}

void deviceSolveEnergy(
    const DeviceMesh& dm,
    const DeviceBoundary& dbHe,
    DeviceThermo& th,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    const DeviceBuffer<scalar>& divU,
    bool limited,
    bool linearUpwind,
    bool nonOrth,
    scalar twoByk,
    scalar relax,
    scalar tol,
    scalar relTol,
    int checkEvery,
    bool useGS,
    DeviceAMI* ami,
    DeviceCyclic* cyc)
{
    if (th.n == 0) return;

    DeviceBuffer<scalar> alphaEff;
    deviceAlphaEff(th, alphaEff);

    // bounded = false: he is not a positive-definite turbulence quantity, so the -Sp(div(phi),he)
    // correction the k/epsilon models use does not belong here. The reaction is empty in phase 1 --
    // no viscous dissipation, no pressure work, nothing to add to diag or source.
    deviceSolveScalarTransport(
        dm,
        dbHe,
        th.he,
        "he",
        alphaEff,
        phiInt,
        phiBnd,
        divU,
        false,
        limited,
        linearUpwind,
        nonOrth,
        twoByk,
        relax,
        tol,
        relTol,
        checkEvery,
        useGS,
        [](DeviceBuffer<scalar>&, DeviceBuffer<scalar>&) {},
        nullptr,
        nullptr,
        ami,
        cyc);
}

} // namespace brae
