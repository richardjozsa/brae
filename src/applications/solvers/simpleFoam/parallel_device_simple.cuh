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
#include "device_blas.cuh"
#include "device_boundary.cuh"
#include "device_pcg.cuh"
#include "device_divdevreff.cuh"
#include "device_halo.cuh"
#include "parallel_simple.cuh"     // Partition, distributeFromCells
#include "reconstruct.cuh"
#include "local_assembly.cuh"   // computeProcUpwindD
#include <cuda_runtime.h>
#include <fstream>
#include <string>
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

// The processor-interface coeffs of a laplacian(gamma, .) matrix -- the pressure equation's coupling. Mirrors
// host assembleLocalLaplacianF, per cut face f (coeff = gammaF*magSf*procDelta):
//   diag[faceCells[f]] -= coeff[f]      (atomic: a cell may own several interface faces)
//   ifCoeff[f]          = -coeff[f]
__global__
void laplacianInterfaceKernel(
    const label*  __restrict__ faceCells,
    const scalar* __restrict__ coeff,
    scalar*       __restrict__ diag,
    scalar*       __restrict__ ifCoeff,
    int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    ifCoeff[f] = -coeff[f];
    atomicAdd(&diag[faceCells[f]], -coeff[f]);
}

// linearUpwind deferred correction at a PROCESSOR face -- the distributed analogue of
// linearUpwindCorrKernel (device_simple.cu) and of deviceCyclicAddLinUpwindCorr (no rotation here: a
// processor cut is a plain interior face, the two sides share an orientation).
//
// The matrix stays plain UPWIND; linearUpwind is this explicit source. Per cut face, the local cell is the
// face's owner and phi is the OUTWARD flux, so it is the owner branch of the internal kernel:
//   corr[localCell] += phi * (grad(U_comp)[upwind] . d_upwind)
// with the upwind side chosen by sign(phi):
//   phi >= 0 -> LOCAL is upwind:  grad at the local cell,  d = dOwn = Cf - C[local]
//   phi <  0 -> REMOTE is upwind: grad at the remote cell (gxN/gyN/gzN, from the halo), d = dNei = Cf - C[remote]
// dOwn/dNei are static geometry, precomputed once by host computeProcUpwindD (the remote centre needs an
// exchange). The caller does relaxSrc -= corr, exactly as for the internal correction.
__global__
void linUpwindInterfaceKernel(
    const label*  __restrict__ faceCells,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    const scalar* __restrict__ gxN,
    const scalar* __restrict__ gyN,
    const scalar* __restrict__ gzN,
    const scalar* __restrict__ dOwn,
    const scalar* __restrict__ dNei,
    scalar*       __restrict__ corr,
    int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    const scalar pf = phi[f];
    const label  c  = faceCells[f];
    scalar g[3], d[3];
    if (pf >= 0.0)   // local upwind: local gradient, local offset
    {
        g[0] = gx[c];  g[1] = gy[c];  g[2] = gz[c];
        d[0] = dOwn[3*f]; d[1] = dOwn[3*f+1]; d[2] = dOwn[3*f+2];
    }
    else             // remote upwind: the halo gradient at the remote cell, remote offset
    {
        g[0] = gxN[f]; g[1] = gyN[f]; g[2] = gzN[f];
        d[0] = dNei[3*f]; d[1] = dNei[3*f+1]; d[2] = dNei[3*f+2];
    }
    atomicAdd(&corr[c], pf * (g[0]*d[0] + g[1]*d[1] + g[2]*d[2]));   // a cell may own several cut faces
}

// LUST linear part at a PROCESSOR face -- the distributed analogue of linearCorrKernel (device_simple.cu).
// The face correction is (linear - upwind) weighted by phi; with linear = w*own + (1-w)*nbr and
// upwind = pos0*own + (1-pos0)*nbr (pos0 = phi>=0), that difference collapses to (w - pos0)*(own - nbr):
//   corr[localCell] += phi * (w - pos0) * (psi[local] - psi[remote])
// The local cell is the cut face's owner and phi is the OUTWARD flux, so this is the internal kernel's owner
// branch. `w` is procW -- the LOCAL side's interpolation weight, matching linearCorrKernel's w[f] = w of own.
// NB the single-GPU path OMITS the cyclic linear part ("LUST cases are non-cyclic"); that shortcut is NOT
// available here, since a decomposed mesh has processor faces everywhere.
__global__
void linearCorrInterfaceKernel(
    const label*  __restrict__ faceCells,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ w,
    const scalar* __restrict__ psi,
    const scalar* __restrict__ psiNbr,
    scalar*       __restrict__ corr,
    int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    const scalar pf   = phi[f];
    const scalar pos0 = (pf >= 0.0) ? 1.0 : 0.0;
    const label  c    = faceCells[f];
    atomicAdd(&corr[c], pf * (w[f] - pos0) * (psi[c] - psiNbr[f]));   // a cell may own several cut faces
}

// Non-orthogonal ("corrected") laplacian correction at a PROCESSOR face -- the distributed analogue of
// lapCorrFaceKernel + lapCorrGatherKernel (device_fvm.cu) and of deviceCyclicAddLapCorr (no rotation here).
// Mirrors the internal formula exactly, with the local cell as owner and the remote as neighbour:
//     grad_face = w*grad[local] + (1-w)*grad[remote]          (w = procW, the LOCAL/owner weight)
//     ffc       = gamma_f * magSf * (corrVec . grad_face)
//     corr[local] -= ffc                                      (the gather kernel's OWNER branch: s -= ffc)
// gamma_f is the face value of the diffusivity (nuEff for momentum, rAU for the pEqn) -- already the
// halo-interpolated boundary value. corrVec/magSf are static geometry from host computeProcNonOrth.
// On an ORTHOGONAL mesh corrVec == 0, so this contributes exactly nothing -- which is why it is invisible
// there, and why it must be validated on the SHEARED duct (see test_gpu_parallel_duct).
__global__
void lapCorrInterfaceKernel(
    const label*  __restrict__ faceCells,
    const scalar* __restrict__ gammaF,
    const scalar* __restrict__ magSfF,
    const scalar* __restrict__ corrVec,
    const scalar* __restrict__ w,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    const scalar* __restrict__ gxN,
    const scalar* __restrict__ gyN,
    const scalar* __restrict__ gzN,
    scalar*       __restrict__ corr,
    int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    const label  c  = faceCells[f];
    const scalar wf = w[f], wn = 1.0 - wf;
    const scalar gfx = wf * gx[c] + wn * gxN[f];
    const scalar gfy = wf * gy[c] + wn * gyN[f];
    const scalar gfz = wf * gz[c] + wn * gzN[f];
    const scalar ffc = gammaF[f] * magSfF[f]
                     * (corrVec[3*f] * gfx + corrVec[3*f+1] * gfy + corrVec[3*f+2] * gfz);
    atomicAdd(&corr[c], -ffc);   // owner branch; a cell may own several cut faces
}

// PRESSURE non-orthogonal correction at a PROCESSOR face -- the distributed analogue of deviceCyclicLapCorrP.
// Does BOTH jobs, exactly as the internal path does with deviceLaplacianCorrFlux + deviceFaceDivSource:
//   ffc         = rAU_f * magSf * (corrVec . grad(p)_face)      (grad_face = w*grad[local] + (1-w)*grad[remote])
//   pb[local]  -= ffc      (the faceDivSource OWNER branch: b += -V*div(ffc))
//   ffcOut[f]   = ffc      (the caller then does phi -= ffc, or continuity breaks on non-orthogonal cut faces)
// Both are required: the source alone leaves the reconstructed flux non-conservative.
__global__
void lapCorrPInterfaceKernel(
    const label*  __restrict__ faceCells,
    const scalar* __restrict__ gammaF,
    const scalar* __restrict__ magSfF,
    const scalar* __restrict__ corrVec,
    const scalar* __restrict__ w,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    const scalar* __restrict__ gxN,
    const scalar* __restrict__ gyN,
    const scalar* __restrict__ gzN,
    scalar*       __restrict__ pb,
    scalar*       __restrict__ ffcOut,
    int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    const label  c  = faceCells[f];
    const scalar wf = w[f], wn = 1.0 - wf;
    const scalar gfx = wf * gx[c] + wn * gxN[f];
    const scalar gfy = wf * gy[c] + wn * gyN[f];
    const scalar gfz = wf * gz[c] + wn * gzN[f];
    const scalar ffc = gammaF[f] * magSfF[f]
                     * (corrVec[3*f] * gfx + corrVec[3*f+1] * gfy + corrVec[3*f+2] * gfz);
    ffcOut[f] = ffc;
    atomicAdd(&pb[c], -ffc);
}

// dst[offset + f] -= src[f]  (subtract an interface's flux correction out of the flattened boundary-flux array)
__global__
void subtractSliceKernel(scalar* __restrict__ dst, int offset, const scalar* __restrict__ src, int n)
{
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= n) return;
    dst[offset + f] -= src[f];
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

// Add the processor faces' linearUpwind deferred correction into `corr` (which already holds the internal
// deviceLinearUpwindCorr result). `gx/gy/gz` are grad(U_comp) as CELL fields -- already processor-consistent
// (the halo injected the coupled face value before the gaussGrad). The three exchanges below fetch that same
// gradient at the REMOTE cell, needed only where the remote side is upwind.
// `phiF[i]` is the processor-face flux, `dOwnD/dNeiD[i]` the precomputed offsets (3 per face).
inline void deviceLinUpwindInterface(
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<label>>& faceCells,
    const std::vector<DeviceBuffer<scalar>>& phiF,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    const std::vector<DeviceBuffer<scalar>>& dOwnD,
    const std::vector<DeviceBuffer<scalar>>& dNeiD,
    DeviceBuffer<scalar>& corr,
    cudaStream_t stream = cudaStreamPerThread)
{
    constexpr int TPB = 128;
    // The recv buffer is shared and reused, so each component's remote gradient must be COPIED out before the
    // next exchange overwrites it (see the hazard note in device_halo.cuh).
    const int nI = halo.nInterfaces();
    std::vector<DeviceBuffer<scalar>> gN[3];
    const DeviceBuffer<scalar>* gsrc[3] = { &gx, &gy, &gz };
    for (int k = 0; k < 3; ++k)
    {
        halo.exchange(gsrc[k]->data(), stream);
        gN[k].resize(nI);
        for (int i = 0; i < nI; ++i)
        {
            const int n = static_cast<int>(halo.size(i));
            if (n <= 0) continue;
            gN[k][i].resize(n);
            cudaCheck(
                cudaMemcpyAsync(gN[k][i].data(), halo.recvData(i), n * sizeof(scalar),
                                cudaMemcpyDeviceToDevice, stream),
                "linUpwind halo grad copy");
        }
        halo.waitExchange(stream);
    }
    for (int i = 0; i < nI; ++i)
    {
        const int n = static_cast<int>(halo.size(i));
        if (n <= 0) continue;
        detail::linUpwindInterfaceKernel<<<(n + TPB - 1) / TPB, TPB, 0, stream>>>(
            faceCells[i].data(),
            phiF[i].data(),
            gx.data(),
            gy.data(),
            gz.data(),
            gN[0][i].data(),
            gN[1][i].data(),
            gN[2][i].data(),
            dOwnD[i].data(),
            dNeiD[i].data(),
            corr.data(),
            n);
    }
}

// Add the processor faces' LUST linear-part correction into `corr` (which already holds the internal
// deviceLinearCorr result). Only the remote psi is needed, so this is a single exchange per component.
inline void deviceLinearCorrInterface(
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<label>>& faceCells,
    const std::vector<DeviceBuffer<scalar>>& phiF,
    const std::vector<DeviceBuffer<scalar>>& weights,
    const DeviceBuffer<scalar>& psi,
    DeviceBuffer<scalar>& corr,
    cudaStream_t stream = cudaStreamPerThread)
{
    constexpr int TPB = 128;
    halo.exchange(psi.data(), stream);
    for (int i = 0; i < halo.nInterfaces(); ++i)
    {
        const int n = static_cast<int>(halo.size(i));
        if (n <= 0) continue;
        detail::linearCorrInterfaceKernel<<<(n + TPB - 1) / TPB, TPB, 0, stream>>>(
            faceCells[i].data(),
            phiF[i].data(),
            weights[i].data(),
            psi.data(),
            halo.recvData(i),
            corr.data(),
            n);
    }
    halo.waitExchange(stream);   // the kernels above READ the shared recv buffer -- see device_halo.cuh
}

// Add the processor faces' non-orthogonal laplacian correction into `corr` (which already holds the internal
// deviceLaplacianCorr result). Needs grad(psi) at the REMOTE cell, so it exchanges the 3 gradient components --
// the same pattern as deviceLinUpwindInterface, copying each out of the shared recv buffer before the next
// exchange overwrites it (see the hazard note in device_halo.cuh).
inline void deviceLapCorrInterface(
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<label>>& faceCells,
    const std::vector<DeviceBuffer<scalar>>& gammaF,
    const std::vector<DeviceBuffer<scalar>>& magSfF,
    const std::vector<DeviceBuffer<scalar>>& corrVec,
    const std::vector<DeviceBuffer<scalar>>& weights,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    DeviceBuffer<scalar>& corr,
    cudaStream_t stream = cudaStreamPerThread)
{
    constexpr int TPB = 128;
    const int nI = halo.nInterfaces();
    std::vector<DeviceBuffer<scalar>> gN[3];
    const DeviceBuffer<scalar>* gsrc[3] = { &gx, &gy, &gz };
    for (int k = 0; k < 3; ++k)
    {
        halo.exchange(gsrc[k]->data(), stream);
        gN[k].resize(nI);
        for (int i = 0; i < nI; ++i)
        {
            const int n = static_cast<int>(halo.size(i));
            if (n <= 0) continue;
            gN[k][i].resize(n);
            cudaCheck(
                cudaMemcpyAsync(gN[k][i].data(), halo.recvData(i), n * sizeof(scalar),
                                cudaMemcpyDeviceToDevice, stream),
                "lapCorr halo grad copy");
        }
        halo.waitExchange(stream);
    }
    for (int i = 0; i < nI; ++i)
    {
        const int n = static_cast<int>(halo.size(i));
        if (n <= 0) continue;
        detail::lapCorrInterfaceKernel<<<(n + TPB - 1) / TPB, TPB, 0, stream>>>(
            faceCells[i].data(),
            gammaF[i].data(),
            magSfF[i].data(),
            corrVec[i].data(),
            weights[i].data(),
            gx.data(), gy.data(), gz.data(),
            gN[0][i].data(), gN[1][i].data(), gN[2][i].data(),
            corr.data(),
            n);
    }
}

// The processor faces' PRESSURE non-orth correction: folds -ffc into `pb` and returns ffc per interface so the
// caller can keep the reconstructed flux conservative. `gammaF[i]` is rAU at interface i's faces.
inline void deviceLapCorrPInterface(
    DeviceHalo& halo,
    const std::vector<DeviceBuffer<label>>& faceCells,
    const std::vector<DeviceBuffer<scalar>>& gammaF,
    const std::vector<DeviceBuffer<scalar>>& magSfF,
    const std::vector<DeviceBuffer<scalar>>& corrVec,
    const std::vector<DeviceBuffer<scalar>>& weights,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    DeviceBuffer<scalar>& pb,
    std::vector<DeviceBuffer<scalar>>& ffcOut,
    cudaStream_t stream = cudaStreamPerThread)
{
    constexpr int TPB = 128;
    const int nI = halo.nInterfaces();
    std::vector<DeviceBuffer<scalar>> gN[3];
    const DeviceBuffer<scalar>* gsrc[3] = { &gx, &gy, &gz };
    for (int k = 0; k < 3; ++k)
    {
        halo.exchange(gsrc[k]->data(), stream);
        gN[k].resize(nI);
        for (int i = 0; i < nI; ++i)
        {
            const int n = static_cast<int>(halo.size(i));
            if (n <= 0) continue;
            gN[k][i].resize(n);
            cudaCheck(
                cudaMemcpyAsync(gN[k][i].data(), halo.recvData(i), n * sizeof(scalar),
                                cudaMemcpyDeviceToDevice, stream),
                "lapCorrP halo grad copy");
        }
        halo.waitExchange(stream);
    }
    ffcOut.resize(nI);
    for (int i = 0; i < nI; ++i)
    {
        const int n = static_cast<int>(halo.size(i));
        if (n <= 0) continue;
        ffcOut[i].resize(n);
        detail::lapCorrPInterfaceKernel<<<(n + TPB - 1) / TPB, TPB, 0, stream>>>(
            faceCells[i].data(),
            gammaF[i].data(),
            magSfF[i].data(),
            corrVec[i].data(),
            weights[i].data(),
            gx.data(), gy.data(), gz.data(),
            gN[0][i].data(), gN[1][i].data(), gN[2][i].data(),
            pb.data(),
            ffcOut[i].data(),
            n);
    }
}

// Assemble a laplacian(gamma, .) matrix's processor coupling (the pressure equation): fold -coeff into `diag`
// and produce `ifCoeff[i]` per interface. `coeffGeo[i] = gammaF*magSf*procDelta` of interface i.
inline void deviceLaplacianInterface(
    const DeviceHalo& halo,
    const std::vector<DeviceBuffer<label>>& faceCells,
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
        detail::laplacianInterfaceKernel<<<(n + TPB - 1) / TPB, TPB, 0, stream>>>(
            faceCells[i].data(),
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

// Max mesh non-orthogonality in DEGREES: the angle between a face's Sf and its owner->neighbour centre vector,
// over internal AND processor faces (the remote centre comes from the same exchange computeProcUpwindD uses).
// OF's "corrected"/"limited" laplacian adds an explicit correction that scales with this angle; the distributed
// device path does NOT implement that correction at processor faces, so on a non-orthogonal mesh it would
// silently solve the "orthogonal" scheme instead -- converged, plausible and WRONG. On an orthogonal mesh the
// correction is identically zero, so the scheme is safe to accept there. This measures which case we are in.
inline scalar maxNonOrthogonality(const Partition& P)
{
    const PrimitiveMesh& m = P.Lm.mesh;
    const FvGeometry& g = P.lg;
    const scalar rad2deg = 180.0 / 3.14159265358979323846;
    auto angle = [&](const vector& S, const vector& d) -> scalar
    {
        const scalar den = mag(S) * mag(d);
        if (den <= 0) return 0.0;
        const scalar c = std::fmin(1.0, std::fmax(-1.0, dot(S, d) / den));
        return std::acos(c) * rad2deg;
    };
    scalar mx = 0;
    for (label f = 0; f < m.nInternalFaces(); ++f)
        mx = std::fmax(mx, angle(g.Sf()[f], g.C()[m.neighbour()[f]] - g.C()[m.owner()[f]]));

    std::vector<std::vector<scalar>> dO, dN;   // dOwn = Cf-C_local, dNei = Cf-C_remote
    computeProcUpwindD(P.Lm, P.lg, P.lp, dO, dN);
    std::size_t j = 0;
    for (std::size_t pi = 0; pi < P.lp.size(); ++pi)
    {
        if (P.lp[pi].type != "processor") continue;
        for (label i = 0; i < P.lp[pi].size; ++i)
        {
            const vector d{ dO[j][3*i]   - dN[j][3*i],        // = C_remote - C_local
                            dO[j][3*i+1] - dN[j][3*i+1],
                            dO[j][3*i+2] - dN[j][3*i+2] };
            mx = std::fmax(mx, angle(g.Sf()[P.lp[pi].start + i], d));
        }
        ++j;
    }
    return Pstream::allReduce(mx, ReduceOp::Max);
}

// ----------------------------------------------------------------------------------------------------------
// ParallelDeviceSimple: the closed distributed laminar SIMPLE loop. One rank == one partition == one GPU.
//
// U/p/phi stay on the device between iterations; step() runs one SIMPLE iteration entirely on the GPU:
//   assemble M = div(phi,U) - laplacian(nuEff,U) (+ processor interface)  ->  relax  ->  momentum predictor
//   rAU / H() / HbyA  ->  phiHbyA  ->  pEqn laplacian(rAU,p) == div(phiHbyA)  ->  conservative phi + corrector
// Each stage is the one validated against host parallelSimpleStepLaminar in test_gpu_parallel_predictor.
//
// Laminar only (nuEff = nu); turbulence is Phase 4b. Processor faces are COUPLED: zero matrix coeffs (bcType 8)
// with the halo-interpolated face value injected for the explicit operators.
// ----------------------------------------------------------------------------------------------------------
class ParallelDeviceSimple
{
public:
    ParallelDeviceSimple(
        const Partition& part,
        const GeometricField<vector>& U0,
        const GeometricField<scalar>& p0,
        scalar nu,
        scalar relaxU,
        scalar relaxP,
        scalar tolU,
        scalar tolP,
        int maxIter,
        bool bounded = false,
        bool linearUpwind = false,
        bool lust = false,
        bool nonOrth = false)
        : P_(part),
          nu_(nu),
          relaxU_(relaxU),
          relaxP_(relaxP),
          tolU_(tolU),
          tolP_(tolP),
          maxIter_(maxIter),
          bounded_(bounded),
          linearUpwind_(linearUpwind),
          lust_(lust),
          nonOrth_(nonOrth),
          lnC_(part.Lm.mesh.nCells()),
          nIf_(part.Lm.mesh.nInternalFaces()),
          dm_(buildDeviceMesh(part.Lm.mesh, part.lg, part.lp)),
          dbU_(buildDeviceVectorBoundary(U0, part.lp, part.lg)),
          dbP_(buildDeviceBoundary(p0, part.lp, part.lg)),
          halo_(part.rank, intNbrs(part), part.Lm.procFaceCells)
    {
        const std::vector<FvPatch>& lp = P_.lp;
        // initial device state
        std::vector<scalar> ux(lnC_), uy(lnC_), uz(lnC_);
        for (label c = 0; c < lnC_; ++c)
        {
            ux[c] = U0.internal[c].x;
            uy[c] = U0.internal[c].y;
            uz[c] = U0.internal[c].z;
        }
        Uk_[0].copyFrom(ux);
        Uk_[1].copyFrom(uy);
        Uk_[2].copyFrom(uz);
        dp_.copyFrom(p0.internal);
        ones_.copyFrom(std::vector<scalar>(lnC_, 1.0));
        zeroSrc_.copyFrom(std::vector<scalar>(lnC_, 0.0));

        // validComponents (fvMesh::validComponents): an empty-patch direction is not solved, so it must not
        // pollute the residualControl measure. Mirrors host parallelSimpleStepLaminar.
        for (std::size_t pi = 0; pi < lp.size(); ++pi)
            if (lp[pi].type == "empty" && lp[pi].size > 0)
            {
                scalar ax = 0, ay = 0, az = 0;
                for (label i = 0; i < lp[pi].size; ++i)
                {
                    const vector& n = P_.lg.Sf()[lp[pi].start + i];
                    ax += std::fabs(n.x);
                    ay += std::fabs(n.y);
                    az += std::fabs(n.z);
                }
                validC_[(ax >= ay && ax >= az) ? 0 : (ay >= az ? 1 : 2)] = false;
            }

        // processor-patch offsets in the flattened boundary array + per-interface device addressing
        label bidx = 0;
        for (std::size_t pi = 0; pi < lp.size(); ++pi)
        {
            if (lp[pi].type == "cyclic" || lp[pi].type == "cyclicAMI") continue;
            if (lp[pi].type == "processor") procStart_.push_back(bidx);
            bidx += lp[pi].size;
        }
        nBnd_ = dbU_.n;   // == bidx; the flattened boundary array excludes cyclic patches
        zeroBnd_.copyFrom(std::vector<scalar>(nBnd_, 0.0));
        nuEffBnd_.copyFrom(std::vector<scalar>(nBnd_, nu_));
        nuCell_.copyFrom(std::vector<scalar>(lnC_, nu_));
        faceCellsD_.resize(P_.Lm.procFaceCells.size());
        weightsD_.resize(P_.procW.size());
        for (std::size_t i = 0; i < P_.Lm.procFaceCells.size(); ++i) faceCellsD_[i].copyFrom(P_.Lm.procFaceCells[i]);
        for (std::size_t i = 0; i < P_.procW.size(); ++i) weightsD_[i].copyFrom(P_.procW[i]);

        // linearUpwind needs Cf-C on BOTH sides of a cut; the remote centre needs an exchange, but the
        // geometry is static, so do it once here rather than per iteration.
        if (linearUpwind_ || lust_)
        {
            std::vector<std::vector<scalar>> dO, dN;
            computeProcUpwindD(P_.Lm, P_.lg, P_.lp, dO, dN);
            dOwnD_.resize(dO.size());
            dNeiD_.resize(dN.size());
            for (std::size_t i = 0; i < dO.size(); ++i) dOwnD_[i].copyFrom(dO[i]);
            for (std::size_t i = 0; i < dN.size(); ++i) dNeiD_[i].copyFrom(dN[i]);
        }

        // non-orth geometry: static, so build it once. procNonOrth_ REPLACES procDelta in the interface
        // coeff (implicit); corrVecD_ drives the explicit correction. Also cache per-interface magSf and the
        // (constant, laminar) face nuEff, both needed by the correction kernel.
        if (nonOrth_)
        {
            std::vector<std::vector<scalar>> cvH;
            computeProcNonOrth(P_.Lm, P_.lg, P_.lp, procNonOrth_, cvH);
            corrVecD_.resize(cvH.size());
            for (std::size_t i = 0; i < cvH.size(); ++i) corrVecD_[i].copyFrom(cvH[i]);
        }
        {
            std::size_t pj = 0;
            for (std::size_t pi = 0; pi < lp.size(); ++pi)
            {
                if (lp[pi].type != "processor") continue;
                std::vector<scalar> ms(lp[pi].size), nf(lp[pi].size, nu_);
                for (label i = 0; i < lp[pi].size; ++i) ms[i] = P_.lg.magSf()[lp[pi].start + i];
                magSfD_.emplace_back();
                magSfD_.back().copyFrom(ms);
                nuFaceD_.emplace_back();
                nuFaceD_.back().copyFrom(nf);
                ++pj;
            }
        }

        // initial conservative flux phi = flux(U0), internal + boundary (processor faces carry the coupled flux)
        const SurfaceScalarField phi0 = fvc::flux(U0, P_.Lm.mesh, P_.lg, P_.lp);
        phiInt_.copyFrom(std::vector<scalar>(phi0.internal.begin(), phi0.internal.begin() + nIf_));
        std::vector<scalar> pb;
        for (std::size_t pi = 0; pi < lp.size(); ++pi)
        {
            if (lp[pi].type == "cyclic" || lp[pi].type == "cyclicAMI") continue;
            for (label i = 0; i < lp[pi].size; ++i) pb.push_back(phi0.boundary[pi][i]);
        }
        phiBnd_.copyFrom(pb);
        // rAU's boundary shape: zeroGradient on real patches, coupled on processor (mirrors distributeFromCells)
        rAUfld_ = distributeFromCells<scalar>(std::vector<scalar>(lnC_, 1.0), P_);
        dbRAU_  = buildDeviceBoundary(rAUfld_, part.lp, part.lg);
    }

    // STAGE DUMP: write every intermediate of iteration `iter`, gathered to the GLOBAL cell ordering,
    // so an np=1 run (no processor faces -> interface terms inactive -> THE reference discretisation) and an
    // np=N run can be diffed stage by stage. The FIRST stage that differs localises the bug, instead of only
    // seeing a blown-up field N iterations later.
    //
    // This is how the LUST interface bug was found: at iteration 2 (the first iteration where the corrections
    // are non-zero at all -- U starts uniform, so grad(U)=0 and every correction is identically 0 at iteration
    // 1) `luCorr_int` was 3.7e-02 off and `luCorr_tot` 1.2e-09 (the linearUpwind interface term restoring it
    // exactly), while `linCorr_tot` was 6.2e+01 off -- one kernel, isolated. Free when unused (early return).
    //
    // Two traps it also exposed, both worth knowing before adding any interface term:
    //   - dump at an iteration where the term is ACTUALLY NON-ZERO (iteration 1 proves nothing here);
    //   - `sumOff` legitimately DIFFERS between np=1 and np>1: at np=1 those cut faces are internal and
    //     deviceRelaxDiag already counts them from the LDU; at np>1 they arrive via the cycSumOff hook. It is
    //     the complement, not a bug -- check `mDiagR` (which must match) instead.
    void setStageDump(const std::string& path, int iter) { dumpPath_ = path; dumpAt_ = iter; }

    // One distributed SIMPLE iteration. U/p/phi are updated in place on the device. Returns the momentum and
    // pressure INITIAL residuals -- OpenFOAM's SIMPLE convergence measure (fvSolution residualControl).
    ParStepResidual step();

    std::vector<vector> U() const
    {
        const std::vector<scalar> ux = Uk_[0].host(), uy = Uk_[1].host(), uz = Uk_[2].host();
        std::vector<vector> out(lnC_);
        for (label c = 0; c < lnC_; ++c) out[c] = vector{ux[c], uy[c], uz[c]};
        return out;
    }
    std::vector<scalar> p() const { return dp_.host(); }

    // max |phi| over this rank's processor faces. The interface deferred corrections (linearUpwind, LUST) are
    // all multiplied by the cut-face flux, so a test on a mesh where this is ~0 cannot see them at all -- the
    // 3D cavity cuts on a symmetry midplane and measures ~3e-14 here. Tests assert on this to prove teeth.
    // Total Krylov iterations (3 momentum BiCGStab + 1 pressure PCG) since construction. A speedup number is
    // uninterpretable without this: if np=1 and np=2 run different iteration counts (the reduction order
    // changes convergence slightly), wall-clock per SIMPLE step measures SOLVER WORK, not parallel efficiency.
    long krylovIters() const { return kIters_; }

    scalar maxProcFlux() const
    {
        const std::vector<scalar> b = phiBnd_.host();
        scalar mx = 0;
        for (std::size_t i = 0; i < procStart_.size(); ++i)
        {
            const label n = static_cast<label>(halo_.size(static_cast<int>(i)));
            for (label f = 0; f < n; ++f)
                mx = std::fmax(mx, std::fabs(b[procStart_[i] + f]));
        }
        return mx;
    }

    // The global field, gathered from every partition (decomposePar's inverse) -- for output/validation.
    std::vector<scalar> reconstructP() const { return reconstructField(P_.Lm.cellProcAddr, dp_.host(), P_.globalNCells); }

    // gather `d` to the global ordering and append "<name> v0 v1 ..." to the dump file (master only)
    void dumpStage(const char* name, const DeviceBuffer<scalar>& d, int k = -1) const
    {
        if (dumpPath_.empty() || iter_ != dumpAt_) return;
        cudaStreamSynchronize(cudaStreamPerThread);
        std::vector<scalar> g(P_.globalNCells, 0.0);
        const std::vector<scalar> h = d.host();
        for (label c = 0; c < lnC_ && c < static_cast<label>(h.size()); ++c)
            g[P_.Lm.cellProcAddr[c]] = h[c];
        Pstream::allReduce(g.data(), static_cast<int>(g.size()), ReduceOp::Sum);
        if (!Pstream::master()) return;
        std::ofstream os(dumpPath_, std::ios::app);
        os.precision(17);
        os << name;
        if (k >= 0) os << '_' << k;
        for (scalar v : g) os << ' ' << v;
        os << '\n';
    }

private:
    static std::vector<int> intNbrs(const Partition& part)
    {
        std::vector<int> n;
        for (int q : part.Lm.procNbr) n.push_back(q);
        return n;
    }

    const Partition& P_;
    scalar nu_, relaxU_, relaxP_, tolU_, tolP_;
    int    maxIter_;
    bool   bounded_ = false;
    bool   linearUpwind_ = false;
    bool   lust_ = false;
    bool   nonOrth_ = false;
    std::string dumpPath_;
    int    dumpAt_ = -1;
    mutable int iter_ = 0;
    mutable long kIters_ = 0;
    label  lnC_, nIf_, nBnd_ = 0;
    bool   validC_[3] = { true, true, true };
    DeviceMesh           dm_;
    DeviceVectorBoundary dbU_;
    DeviceBoundary       dbP_, dbRAU_;
    GeometricField<scalar> rAUfld_;
    DeviceHalo           halo_;
    std::vector<label>   procStart_;
    std::vector<DeviceBuffer<label>>  faceCellsD_;
    std::vector<DeviceBuffer<scalar>> weightsD_;
    std::vector<DeviceBuffer<scalar>> dOwnD_, dNeiD_;   // linearUpwind: static per-cut-face offsets
    // non-orth ("corrected") laplacian: static per-cut-face geometry + the per-interface face areas
    std::vector<std::vector<scalar>> procNonOrth_;      // host: the IMPLICIT coeff (replaces procDelta)
    std::vector<DeviceBuffer<scalar>> corrVecD_, magSfD_, nuFaceD_;
    DeviceBuffer<scalar> Uk_[3], dp_, phiInt_, phiBnd_;
    DeviceBuffer<scalar> ones_, zeroSrc_, zeroBnd_, nuEffBnd_, nuCell_;
};

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
    // REQUIRED: the interface kernels above READ the shared recv buffer. Without this barrier a neighbour's
    // NEXT exchange (e.g. the following velocity component's H) can overwrite our recv buffer while these
    // kernels are still reading it -- see the hazard note in device_halo.cuh. Calling H() per component in a
    // loop hits this immediately.
    halo.waitExchange(stream);
}

// One distributed SIMPLE iteration -- the sequence validated stage-by-stage against host
// parallelSimpleStepLaminar in test_gpu_parallel_predictor.
inline ParStepResidual ParallelDeviceSimple::step()
{
    ParStepResidual res;
    ++iter_;
    const std::vector<FvPatch>& lp = P_.lp;
    const FvGeometry& lg = P_.lg;

    // ---- momentum matrix: div(phi,U) - laplacian(nuEff,U), + processor interface ----
    DeviceBuffer<scalar> nuEff_f(std::vector<scalar>(nIf_, nu_));
    DeviceBuffer<scalar> mDiag, mUp, mLo, lD, lU, lL;
    deviceDivUpwindCoeffs(dm_, phiInt_, mDiag, mUp, mLo);
    deviceLaplacianCoeffs(dm_, nuEff_f, lD, lU, lL, nonOrth_);
    deviceAxpy(-1.0, lD, mDiag);
    deviceAxpy(-1.0, lU, mUp);
    deviceAxpy(-1.0, lL, mLo);

    // bounded Gauss upwind: - fvm::Sp(fvc::div(phi), U). Diagonal gets -V*div(phi); it stabilises the
    // transient (a rest-start cell where div(phi)!=0 at low nu blows up otherwise) and vanishes at
    // convergence. Mirrors host parallelSimpleStepLaminar's `bounded` branch.
    if (bounded_)
    {
        DeviceBuffer<scalar> divPhi, sp;
        deviceDiv(dm_, phiInt_, phiBnd_, divPhi);
        deviceHadamard(sp, dm_.V, divPhi);
        deviceAxpy(-1.0, sp, mDiag);
    }

    const std::vector<scalar> phiBndH = phiBnd_.host();
    std::vector<DeviceBuffer<scalar>> phiF, coeffGeo;
    {
        std::size_t pj = 0;
        label bi = 0;
        for (std::size_t pi = 0; pi < lp.size(); ++pi)
        {
            if (lp[pi].type == "cyclic" || lp[pi].type == "cyclicAMI") continue;
            if (lp[pi].type == "processor")
            {
                DeviceBuffer<scalar> pf;
                pf.copyFrom(std::vector<scalar>(phiBndH.begin() + bi, phiBndH.begin() + bi + lp[pi].size));
                phiF.push_back(std::move(pf));
                std::vector<scalar> cg(lp[pi].size);
                for (label i = 0; i < lp[pi].size; ++i)
                    cg[i] = nu_ * lg.magSf()[lp[pi].start + i]
                          * (nonOrth_ ? procNonOrth_[pj][i] : P_.procDelta[pj][i]);
                DeviceBuffer<scalar> cgd;
                cgd.copyFrom(cg);
                coeffGeo.push_back(std::move(cgd));
                ++pj;
            }
            bi += lp[pi].size;
        }
    }
    std::vector<DeviceBuffer<scalar>> ifCoeff;
    deviceMomentumInterface(halo_, faceCellsD_, phiF, coeffGeo, mDiag, ifCoeff);
    dumpStage("mDiag", mDiag);

    // ---- relax (the processor interface counts toward diagonal dominance) ----
    DeviceBuffer<scalar> iC[3], bC[3], relaxSrc[3];
    for (int k = 0; k < 3; ++k)
    {
        DeviceBuffer<scalar> lIC, lBC;
        deviceBCDivCoeffs(dbU_.comp[k], phiBnd_, iC[k], bC[k]);
        deviceBCLaplacianCoeffsFace(dbU_.comp[k], nuEffBnd_, lIC, lBC);
        deviceAxpy(-1.0, lIC, iC[k]);
        deviceAxpy(-1.0, lBC, bC[k]);
    }
    DeviceBuffer<scalar> sumOff(std::vector<scalar>(lnC_, 0.0));
    deviceInterfaceOffDiagSum(halo_, faceCellsD_, ifCoeff, sumOff);
    DeviceBuffer<scalar> mDiagR, delta;
    deviceRelaxDiag(deviceLduView(dm_, mDiag, mUp, mLo), dm_, iC[0], relaxU_, mDiagR, delta, sumOff.data());
    dumpStage("sumOff", sumOff);
    dumpStage("mDiagR", mDiagR);
    dumpStage("delta", delta);

    // The explicit stress source, V*fvc::div(nuEff*dev2(T(grad U))), from the PRE-predictor U -- processor
    // faces coupled via the halo. Host parallelSimpleStepLaminar folds this into Ml.source BEFORE relax, so
    // it must sit in the same source relax then adds delta*U_old to, i.e. the one H() reads later.
    DeviceProcStress proc;
    proc.halo      = &halo_;
    proc.weights   = &weightsD_;
    proc.procStart = &procStart_;
    DeviceBuffer<scalar> sX, sY, sZ;
    deviceDivDevReff(dm_, dbU_, Uk_[0], Uk_[1], Uk_[2], nuCell_, nuEffBnd_, sX, sY, sZ, nullptr, nullptr, &proc);
    DeviceBuffer<scalar>* sS[3] = { &sX, &sY, &sZ };
    dumpStage("stress", sX, 0);
    dumpStage("stress", sY, 1);
    dumpStage("stress", sZ, 2);
    for (int k = 0; k < 3; ++k)
    {
        deviceHadamard(relaxSrc[k], delta, Uk_[k]);      // delta*U_old
        deviceAxpy(1.0, *sS[k], relaxSrc[k]);            // + V*divSig  -> == host Ml.source
    }

    // linearUpwind: the matrix stays upwind and this deferred correction is an explicit source. It must land
    // in relaxSrc -- the SAME source H() reads later -- not only in the predictor's rhs, or HbyA would be
    // built from an upwind-only source and the corrector would disagree with the predictor.
    // LUST also runs the linearUpwind correction (OF LUST.H = 0.75*linear + 0.25*linearUpwind), mirroring the
    // single-GPU `if (ctl_.linearUpwind || ctl_.lust)`.
    if (linearUpwind_ || lust_)
    {
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> ub;
            deviceBCValue(dbU_.comp[k], Uk_[k], ub);
            halo_.exchange(Uk_[k].data());
            halo_.scatterBoundaryValues(Uk_[k].data(), weightsD_, procStart_, ub.data());
            halo_.waitExchange();
            DeviceBuffer<scalar> ggx, ggy, ggz;
            deviceGaussGrad(dm_, Uk_[k], ub, ggx, ggy, ggz);   // processor-consistent grad(U_k)
            dumpStage("gradx", ggx, k);
            dumpStage("grady", ggy, k);

            DeviceBuffer<scalar> corr;
            deviceLinearUpwindCorr(dm_, phiInt_, ggx, ggy, ggz, corr);          // internal faces
            dumpStage("luCorr_int", corr, k);
            deviceLinUpwindInterface(halo_, faceCellsD_, phiF, ggx, ggy, ggz,
                                     dOwnD_, dNeiD_, corr);                     // + cut faces
            dumpStage("luCorr_tot", corr, k);
            if (lust_)   // 0.25*linearUpwind + 0.75*linear, cut faces included on BOTH parts
            {
                deviceScale(corr, 0.25);
                DeviceBuffer<scalar> lc;
                deviceLinearCorr(dm_, phiInt_, Uk_[k], lc);                     // internal faces
                dumpStage("linCorr_int", lc, k);
                deviceLinearCorrInterface(halo_, faceCellsD_, phiF, weightsD_, Uk_[k], lc);   // + cut faces
                dumpStage("linCorr_tot", lc, k);
                deviceAxpy(0.75, lc, corr);
            }
            dumpStage("corr_final", corr, k);
            deviceAxpy(-1.0, corr, relaxSrc[k]);                                // source -= corr
            dumpStage("relaxSrc", relaxSrc[k], k);
        }
    }

    // Non-orthogonal ("corrected") laplacian: the matrix above used nonOrthDeltaCoeffs (implicit); this is
    // the matching EXPLICIT deferred correction, internal faces + cut faces. It must land in relaxSrc -- the
    // same source H() reads -- like every other explicit momentum term. Mirrors the single-GPU
    // `if (ctl_.nonOrth) { deviceLaplacianCorr; +cyclic; relaxSrc -= corr; }`.
    if (nonOrth_)
    {
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> ub;
            deviceBCValue(dbU_.comp[k], Uk_[k], ub);
            halo_.exchange(Uk_[k].data());
            halo_.scatterBoundaryValues(Uk_[k].data(), weightsD_, procStart_, ub.data());
            halo_.waitExchange();
            DeviceBuffer<scalar> ggx, ggy, ggz;
            deviceGaussGrad(dm_, Uk_[k], ub, ggx, ggy, ggz);

            DeviceBuffer<scalar> lc;
            deviceLaplacianCorr(dm_, nuEff_f, ggx, ggy, ggz, lc);                       // internal faces
            deviceLapCorrInterface(halo_, faceCellsD_, nuFaceD_, magSfD_, corrVecD_,
                                   weightsD_, ggx, ggy, ggz, lc);                       // + cut faces
            deviceAxpy(-1.0, lc, relaxSrc[k]);   // momentum is div - laplacian: source -= lapCorr
        }
    }

    // ---- grad(p): coupled processor boundary value from the halo ----
    DeviceBuffer<scalar> pbv;
    deviceBCValue(dbP_, dp_, pbv);
    halo_.exchange(dp_.data());
    halo_.scatterBoundaryValues(dp_.data(), weightsD_, procStart_, pbv.data());
    halo_.waitExchange();
    DeviceBuffer<scalar> gx, gy, gz;
    deviceGaussGrad(dm_, dp_, pbv, gx, gy, gz);
    DeviceBuffer<scalar>* gg[3] = { &gx, &gy, &gz };

    // ---- momentum predictor ----
    for (int k = 0; k < 3; ++k)
    {
        DeviceBuffer<scalar> s;
        deviceHadamard(s, dm_.V, *gg[k]);
        deviceScale(s, -1.0);
        deviceAxpy(1.0, relaxSrc[k], s);
        DeviceBuffer<scalar> diagC, b;
        deviceFold(dm_, mDiagR, s, iC[k], bC[k], diagC, b);
        const DeviceLduView mv = deviceLduView(dm_, diagC, mUp, mLo);
        const scalar nf = deviceParallelNormFactor(mv, halo_, ifCoeff, Uk_[k], b, ones_, P_.globalNCells);
        const DeviceSolverPerf up =
            deviceParallelJacobiBiCGStab(mv, halo_, ifCoeff, b, Uk_[k], nf, tolU_, 0.0, maxIter_);
        kIters_ += up.nIterations;
        if (validC_[k] && up.initialResidual > res.Ux) res.Ux = up.initialResidual;
        dumpStage("Upred", Uk_[k], k);
    }

    // ---- rAU, HbyA ----
    DeviceBuffer<scalar> cmptAvIC;
    deviceCopy(cmptAvIC, iC[0]);
    deviceAxpy(1.0, iC[1], cmptAvIC);
    deviceAxpy(1.0, iC[2], cmptAvIC);
    deviceScale(cmptAvIC, 1.0 / 3.0);
    DeviceBuffer<scalar> diagA, dumb, rAU;
    deviceFold(dm_, mDiagR, zeroSrc_, cmptAvIC, zeroBnd_, diagA, dumb);
    deviceReciprocalV(dm_, diagA, rAU);
    dumpStage("rAU", rAU);

    DeviceBuffer<scalar> HbyA[3];
    const DeviceLduView av = deviceLduView(dm_, diagA, mUp, mLo);
    for (int k = 0; k < 3; ++k)
    {
        DeviceBuffer<scalar> bdDiag;
        deviceCopy(bdDiag, cmptAvIC);
        deviceAxpy(-1.0, iC[k], bdDiag);
        DeviceBuffer<scalar> Hk;
        deviceParallelMatrixH(av, dm_, halo_, faceCellsD_, ifCoeff, Uk_[k], relaxSrc[k], bdDiag, bC[k], Hk);
        deviceHadamard(HbyA[k], rAU, Hk);
        dumpStage("HbyA", HbyA[k], k);
    }

    // ---- phiHbyA ----
    DeviceBuffer<scalar> hb[3];
    for (int k = 0; k < 3; ++k)
    {
        deviceBCValue(dbU_.comp[k], HbyA[k], hb[k]);
        halo_.exchange(HbyA[k].data());
        halo_.scatterBoundaryValues(HbyA[k].data(), weightsD_, procStart_, hb[k].data());
        halo_.waitExchange();
    }
    DeviceBuffer<scalar> phiHbyAint, phiHbyAbnd;
    deviceVectorFlux(dm_, HbyA[0], HbyA[1], HbyA[2], phiHbyAint);
    deviceBoundaryFlux(dm_, hb[0], hb[1], hb[2], phiHbyAbnd);

    // ---- pEqn: laplacian(rAU,p) == div(phiHbyA) ----
    DeviceBuffer<scalar> rAUf_int;
    deviceInterpolate(dm_, rAU, rAUf_int);
    DeviceBuffer<scalar> rAUbnd;
    deviceBCValue(dbRAU_, rAU, rAUbnd);
    halo_.exchange(rAU.data());
    halo_.scatterBoundaryValues(rAU.data(), weightsD_, procStart_, rAUbnd.data());
    halo_.waitExchange();

    DeviceBuffer<scalar> pD, pU_, pL_;
    deviceLaplacianCoeffs(dm_, rAUf_int, pD, pU_, pL_, nonOrth_);
    const std::vector<scalar> rAUbndH = rAUbnd.host();
    std::vector<DeviceBuffer<scalar>> pCoeffGeo;
    {
        std::size_t pj = 0;
        label bi = 0;
        for (std::size_t pi = 0; pi < lp.size(); ++pi)
        {
            if (lp[pi].type == "cyclic" || lp[pi].type == "cyclicAMI") continue;
            if (lp[pi].type == "processor")
            {
                std::vector<scalar> cg(lp[pi].size);
                for (label i = 0; i < lp[pi].size; ++i)
                    cg[i] = rAUbndH[bi + i] * lg.magSf()[lp[pi].start + i]
                          * (nonOrth_ ? procNonOrth_[pj][i] : P_.procDelta[pj][i]);
                DeviceBuffer<scalar> cgd;
                cgd.copyFrom(cg);
                pCoeffGeo.push_back(std::move(cgd));
                ++pj;
            }
            bi += lp[pi].size;
        }
    }
    std::vector<DeviceBuffer<scalar>> pIfCoeff;
    deviceLaplacianInterface(halo_, faceCellsD_, pCoeffGeo, pD, pIfCoeff);

    DeviceBuffer<scalar> dphi, psrc;
    deviceDiv(dm_, phiHbyAint, phiHbyAbnd, dphi);
    deviceHadamard(psrc, dm_.V, dphi);
    DeviceBuffer<scalar> piC, pbC;
    deviceBCLaplacianCoeffsFace(dbP_, rAUbnd, piC, pbC);
    DeviceBuffer<scalar> pDiagC, pb;
    deviceFold(dm_, pD, psrc, piC, pbC, pDiagC, pb);

    // Explicit non-orth correction of laplacian(rAU,p), from the ENTRY grad(p) (gx/gy/gz), matching the
    // single-GPU pass-0 convention. It needs BOTH halves: b += -V*div(ffc) here, and phi -= ffc at the
    // corrector below -- the source alone leaves the reconstructed flux non-conservative on cut faces.
    DeviceBuffer<scalar> ffcP;
    std::vector<DeviceBuffer<scalar>> ffcPif;
    if (nonOrth_)
    {
        deviceLaplacianCorrFlux(dm_, rAUf_int, gx, gy, gz, ffcP);      // internal faces
        DeviceBuffer<scalar> sc;
        deviceFaceDivSource(dm_, ffcP, sc);
        deviceAxpy(1.0, sc, pb);
        // rAU at the cut faces = the halo-interpolated boundary value, sliced per interface
        std::vector<DeviceBuffer<scalar>> rAUfaceD(procStart_.size());
        for (std::size_t i = 0; i < procStart_.size(); ++i)
        {
            const int n = static_cast<int>(halo_.size(static_cast<int>(i)));
            if (n <= 0) continue;
            rAUfaceD[i].resize(n);
            cudaCheck(
                cudaMemcpyAsync(rAUfaceD[i].data(), rAUbnd.data() + procStart_[i], n * sizeof(scalar),
                                cudaMemcpyDeviceToDevice, cudaStreamPerThread),
                "rAU face slice");
        }
        deviceLapCorrPInterface(halo_, faceCellsD_, rAUfaceD, magSfD_, corrVecD_, weightsD_,
                                gx, gy, gz, pb, ffcPif);               // + cut faces
    }

    DeviceBuffer<scalar> pSol;
    deviceCopy(pSol, dp_);
    const DeviceLduView pv = deviceLduView(dm_, pDiagC, pU_, pL_);
    const scalar nfp = deviceParallelNormFactor(pv, halo_, pIfCoeff, pSol, pb, ones_, P_.globalNCells);
    const DeviceSolverPerf pp = deviceParallelJacobiPCG(pv, halo_, pIfCoeff, pb, pSol, nfp, tolP_, 0.0, maxIter_);
    kIters_ += pp.nIterations;
    res.p = pp.initialResidual;
    dumpStage("pSol", pSol);

    // ---- corrector: conservative phi, relax p, U = HbyA - rAU*grad(p) ----
    DeviceBuffer<scalar> pfluxInt, pfluxBnd;
    deviceMatrixFluxInternal(pv, pSol, pfluxInt);
    deviceMatrixFluxBoundary(dbP_, piC, pbC, pSol, pfluxBnd);
    deviceParallelMatrixFluxInterface(halo_, faceCellsD_, pIfCoeff, procStart_, pSol, pfluxBnd);
    deviceAxpy(-1.0, pfluxInt, phiHbyAint);          // phi = phiHbyA - pflux
    deviceAxpy(-1.0, pfluxBnd, phiHbyAbnd);
    if (nonOrth_)   // ... and the non-orth face-flux correction, so div(phi)=0 holds on non-orthogonal faces
    {
        deviceAxpy(-1.0, ffcP, phiHbyAint);
        constexpr int TPB = 128;
        for (std::size_t i = 0; i < ffcPif.size(); ++i)
        {
            const int n = static_cast<int>(halo_.size(static_cast<int>(i)));
            if (n <= 0) continue;
            detail::subtractSliceKernel<<<(n + TPB - 1) / TPB, TPB, 0, cudaStreamPerThread>>>(
                phiHbyAbnd.data(), static_cast<int>(procStart_[i]), ffcPif[i].data(), n);
        }
    }
    deviceCopy(phiInt_, phiHbyAint);                 // maintained across iterations
    deviceCopy(phiBnd_, phiHbyAbnd);

    DeviceBuffer<scalar> pRelax;                     // p = pPrev + relaxP*(pNew - pPrev)
    deviceCopy(pRelax, pSol);
    deviceAxpy(-1.0, dp_, pRelax);
    deviceScale(pRelax, relaxP_);
    deviceAxpy(1.0, dp_, pRelax);
    deviceCopy(dp_, pRelax);

    DeviceBuffer<scalar> pbv2;
    deviceBCValue(dbP_, dp_, pbv2);
    halo_.exchange(dp_.data());
    halo_.scatterBoundaryValues(dp_.data(), weightsD_, procStart_, pbv2.data());
    halo_.waitExchange();
    DeviceBuffer<scalar> gxn, gyn, gzn;
    deviceGaussGrad(dm_, dp_, pbv2, gxn, gyn, gzn);
    DeviceBuffer<scalar>* gn[3] = { &gxn, &gyn, &gzn };
    for (int k = 0; k < 3; ++k)
    {
        DeviceBuffer<scalar> Un;
        deviceCorrector(HbyA[k], rAU, *gn[k], Un);
        deviceCopy(Uk_[k], Un);
        dumpStage("Ucorr", Uk_[k], k);
    }
    return res;
}

} // namespace brae
