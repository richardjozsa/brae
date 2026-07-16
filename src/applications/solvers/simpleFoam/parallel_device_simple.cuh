#pragma once
// brae distributed device SIMPLE, the multi-GPU laminar SIMPLE step composed from the distributed device
// primitives (deviceParallelAmul, deviceParallelJacobiPCG, DeviceHalo). Built increment by increment; kept
// SEPARATE from the single-GPU DeviceSimpleSolver so it cannot regress the OpenFOAM-validated solver.
//
// Processor faces are a coupled interface (the DeviceCyclic/DeviceAMI pattern): the matrix path adds an
// off-diagonal interface coeff (this file), and the explicit operators inject the coupled face value into the
// boundary array (DeviceHalo::scatterBoundaryValues). A processor face is NEVER a real boundary face.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_ldu.cuh"
#include "device_simple.cuh"
#include "device_halo.cuh"
#include <cuda_runtime.h>
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

// L4: the processor contribution to H(). lduMatrix::H(psi) = -offdiag.psi, and the PARALLEL A.psi carries
// -ifCoeff*psiNbr at each interface face, so H picks up +ifCoeff*psiNbr there. deviceMatrixH already divided
// by V, hence the /V. atomicAdd: a cell may own several faces on one interface.
__global__
void matrixHInterfaceKernel(
    const label*  __restrict__ faceCells,
    const scalar* __restrict__ ifCoeff,
    const scalar* __restrict__ psiNbr,
    const scalar* __restrict__ V,
    scalar*       __restrict__ Hk,
    int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    const label c = faceCells[f];
    atomicAdd(&Hk[c], ifCoeff[f] * psiNbr[f] / V[c]);
}

// L7: the pressure flux across a processor face, pEqn.flux() -- mirrors host parallelMatrixFlux:
//   flux[f] = -ifCoeff[f] * (p_neighbour[f] - p_local[faceCells[f]])
// It is conservative because the laplacian interface coeff has the same magnitude on both sides, so the two
// sides differ only by the swap of (p_nbr, p_local) -> equal and opposite. fluxOut points at the interface's
// slice of the flattened boundary-flux array.
__global__
void matrixFluxInterfaceKernel(
    const label*  __restrict__ faceCells,
    const scalar* __restrict__ ifCoeff,
    const scalar* __restrict__ pNbr,
    const scalar* __restrict__ p,
    scalar*       __restrict__ fluxOut,
    int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    fluxOut[f] = -ifCoeff[f] * (pNbr[f] - p[faceCells[f]]);
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

// Write the processor-face pressure flux into `fluxB` (the flattened boundary-flux array) at each interface's
// procStart offset. Exchanges p itself. The result cancels across a cut face (both sides equal and opposite),
// which is what keeps the corrected phi globally conservative.
inline void deviceParallelMatrixFluxInterface(
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<label>>& faceCells,
    const std::vector<DeviceBuffer<scalar>>& ifCoeff,
    const std::vector<label>& procStart,
    const DeviceBuffer<scalar>& p,
    DeviceBuffer<scalar>& fluxB,
    cudaStream_t stream = cudaStreamPerThread)
{
    constexpr int TPB = 128;
    halo.exchange(p.data(), stream);
    for (int i = 0; i < halo.nInterfaces(); ++i)
    {
        const int n = static_cast<int>(halo.size(i));
        if (n <= 0) continue;
        detail::matrixFluxInterfaceKernel<<<(n + TPB - 1) / TPB, TPB, 0, stream>>>(
            faceCells[i].data(),
            ifCoeff[i].data(),
            halo.recvData(i),
            p.data(),
            fluxB.data() + procStart[i],
            n);
    }
    halo.waitExchange(stream);   // protect the recv buffer from the next exchange (see device_halo.cuh hazard)
}

// Distributed H() per component: the local deviceMatrixH plus the processor-interface term, i.e. the device
// counterpart of host parallelMatrixH (H = (diag*psi - A_parallel*psi + source)/V). Exchanges psiK itself, so
// the caller does not need to pre-exchange. `faceCells[i]`/`ifCoeff[i]` are interface i's addressing/coeffs.
inline void deviceParallelMatrixH(
    const DeviceLduView& A,
    const DeviceMesh& dm,
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<label>>& faceCells,
    const std::vector<DeviceBuffer<scalar>>& ifCoeff,
    const DeviceBuffer<scalar>& psiK,
    const DeviceBuffer<scalar>& sourceK,
    const DeviceBuffer<scalar>& bdDiagK,
    const DeviceBuffer<scalar>& bdSrcK,
    DeviceBuffer<scalar>& Hk,
    cudaStream_t stream = cudaStreamPerThread)
{
    constexpr int TPB = 128;
    halo.postExchange(psiK.data(), stream);                                   // psiNbr for the interface term
    deviceMatrixH(A, dm, psiK, sourceK, bdDiagK, bdSrcK, Hk);                 // local H (overlaps the transfer)
    halo.waitExchange(stream);
    for (int i = 0; i < halo.nInterfaces(); ++i)
    {
        const int n = static_cast<int>(halo.size(i));
        if (n <= 0) continue;
        detail::matrixHInterfaceKernel<<<(n + TPB - 1) / TPB, TPB, 0, stream>>>(
            faceCells[i].data(),
            ifCoeff[i].data(),
            halo.recvData(i),
            dm.V.data(),
            Hk.data(),
            n);
    }
}

} // namespace brae
