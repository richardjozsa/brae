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

// K = 0.5|U|^2 per cell.
__global__
void kineticEnergyK(
    int n,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar* __restrict__ K)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    K[c] = 0.5 * (Ux[c]*Ux[c] + Uy[c]*Uy[c] + Uz[c]*Uz[c]);
}

// Upwind face value of K times the mass flux: ffc_f = phi_f * K_upwind(f).
__global__
void kineticFaceFluxK(
    int nF,
    const label* __restrict__ owner,
    const label* __restrict__ neighbour,
    const scalar* __restrict__ phiInt,
    const scalar* __restrict__ K,
    scalar* __restrict__ ffc)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nF) return;
    const scalar ph = phiInt[f];
    ffc[f] = ph * (ph >= scalar(0) ? K[owner[f]] : K[neighbour[f]]);
}

// Boundary half: += phi_b*K_b, and the "bounded" correction -K_c*(sum_f phi_f) that OF's
// boundedConvectionScheme subtracts. Both scattered with atomics onto the adjacent cell.
__global__
void kineticBoundaryK(
    int nB,
    const label* __restrict__ faceCell,
    const scalar* __restrict__ phiBnd,
    const scalar* __restrict__ Kb,
    scalar* __restrict__ src,
    scalar* __restrict__ sumPhi)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nB) return;
    const int c = faceCell[i];
    atomicAdd(&src[c], phiBnd[i] * Kb[i]);
    atomicAdd(&sumPhi[c], phiBnd[i]);
}

__global__
void kineticBoundedK(
    int n,
    const scalar* __restrict__ K,
    const scalar* __restrict__ sumPhi,
    scalar* __restrict__ src)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    src[c] -= K[c] * sumPhi[c];   // OF: boundedConvectionScheme subtracts fvc::surfaceIntegrate(phi)*vf
}

void deviceEnergyKineticSource(
    const DeviceMesh& dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    DeviceBuffer<scalar>& src)
{
    const int nC = dm.nCells;
    const int nF = dm.nInternalFaces;
    DeviceBuffer<scalar> K;
    K.resize(nC);
    kineticEnergyK<<<nBlocks(nC), TPB>>>(nC, Ux.data(), Uy.data(), Uz.data(), K.data());
    cudaCheck(cudaGetLastError(), "kineticEnergy");

    DeviceBuffer<scalar> ffc;
    ffc.resize(nF);
    kineticFaceFluxK<<<nBlocks(nF), TPB>>>(nF, dm.owner.data(), dm.nei.data(), phiInt.data(), K.data(), ffc.data());
    cudaCheck(cudaGetLastError(), "kineticFaceFlux");
    deviceFaceDivSource(dm, ffc, src);   // internal faces -> sum_f, i.e. V*div(phi,K)

    // K at boundary faces, from the boundary velocity (noSlip walls give K_b = 0, which is the point:
    // the flow gives up its kinetic energy there and OF puts that into the enthalpy equation).
    const int nB = dbU.comp[0].n;
    if (nB > 0)
    {
        DeviceBuffer<scalar> ubx, uby, ubz;
        deviceBCValue(dbU.comp[0], Ux, ubx);
        deviceBCValue(dbU.comp[1], Uy, uby);
        deviceBCValue(dbU.comp[2], Uz, ubz);
        DeviceBuffer<scalar> Kb;
        Kb.resize(nB);
        kineticEnergyK<<<nBlocks(nB), TPB>>>(nB, ubx.data(), uby.data(), ubz.data(), Kb.data());
        cudaCheck(cudaGetLastError(), "kineticEnergyBnd");

        DeviceBuffer<scalar> sumPhi;
        sumPhi.resize(nC);
        cudaCheck(cudaMemsetAsync(sumPhi.data(), 0, nC*sizeof(scalar), cudaStreamPerThread), "sumPhi zero");
        kineticBoundaryK<<<nBlocks(nB), TPB>>>(nB, dbU.comp[0].faceCell.data(), phiBnd.data(), Kb.data(),
                                               src.data(), sumPhi.data());
        cudaCheck(cudaGetLastError(), "kineticBoundary");
        // internal faces contribute to sum_f phi_f too -- reuse the same gather on phi itself
        DeviceBuffer<scalar> divPhiInt;
        deviceFaceDivSource(dm, phiInt, divPhiInt);
        deviceAxpy(1.0, divPhiInt, sumPhi);
        kineticBoundedK<<<nBlocks(nC), TPB>>>(nC, K.data(), sumPhi.data(), src.data());
        cudaCheck(cudaGetLastError(), "kineticBounded");
    }
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
    DeviceCyclic* cyc,
    const DeviceBuffer<scalar>* kineticSrc)
{
    if (th.n == 0) return;

    DeviceBuffer<scalar> alphaEff;
    deviceAlphaEff(th, alphaEff);

    // bounded = false: he is not a positive-definite turbulence quantity, so the -Sp(div(phi),he)
    // correction the k/epsilon models use does not belong here.
    //
    // The only source is OF's kinetic-energy transport: EEqn carries + fvc::div(phi, K) on the LHS with
    // K = 0.5|U|^2, so it enters the RHS with a minus. Negligible in a slow laminar duct, which is why
    // phase 1 left it out; at 50 m/s against a noSlip wall it is not, and it shifts T by percent.
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
        [&](DeviceBuffer<scalar>&, DeviceBuffer<scalar>& source)
        {
            if (kineticSrc && kineticSrc->size()) deviceAxpy(-1.0, *kineticSrc, source);
        },
        nullptr,
        nullptr,
        ami,
        cyc);
}

} // namespace brae
