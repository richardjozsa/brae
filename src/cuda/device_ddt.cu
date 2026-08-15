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

namespace {
// One face. Written once and used for both the internal and the boundary sweep so the two cannot drift:
// the only difference between them is where phi_old and rAU_face come from.
__device__ __forceinline__ scalar ddtCorrFace(scalar phiOld, scalar fluxUold, scalar rAUf, scalar rDeltaT)
{
    const scalar phiCorr = phiOld - fluxUold;
    // ddtScheme<Type>::fvcDdtPhiCoeff, the ddtPhiCoeff_ < 0 branch (the default -- v2412 never reads
    // ddtPhiCoeff from fvSchemes, so that branch is the only one a case can reach):
    //     coeff = 1 - min(mag(phiCorr)/(mag(phi) + SMALL), 1)
    // SMALL is Foam::SMALL = 1e-15 in double precision.
    const scalar coeff = scalar(1) - fmin(fabs(phiCorr)/(fabs(phiOld) + scalar(1e-15)), scalar(1));
    return rAUf*coeff*rDeltaT*phiCorr;
}

__global__ void ddtCorrIntK(int nIf, const scalar* phiOld, const scalar* fluxUold, const scalar* rAUf,
                            scalar rDeltaT, scalar* out)
{
    const int f = blockIdx.x*blockDim.x + threadIdx.x;
    if (f >= nIf) return;
    out[f] += ddtCorrFace(phiOld[f], fluxUold[f], rAUf[f], rDeltaT);
}

__global__ void ddtCorrBndK(int nBf, const scalar* phiOld, const scalar* fluxUold, const scalar* rAUb,
                            const scalar* mask, scalar rDeltaT, scalar* out)
{
    const int b = blockIdx.x*blockDim.x + threadIdx.x;
    if (b >= nBf) return;
    if (mask[b] == scalar(0)) return;   // fixesValue / cyclicAMI: OF sets the coupling coefficient to 0
    out[b] += ddtCorrFace(phiOld[b], fluxUold[b], rAUb[b], rDeltaT);
}
}   // namespace

void deviceDdtCorrFlux(
    int                         nInternalFaces,
    const DeviceBuffer<scalar>& phiOldInt,
    const DeviceBuffer<scalar>& phiOldBnd,
    const DeviceBuffer<scalar>& fluxUoldInt,
    const DeviceBuffer<scalar>& fluxUoldBnd,
    const DeviceBuffer<scalar>& rAUf,
    const DeviceBuffer<scalar>& rAUbnd,
    const DeviceBuffer<scalar>& coeffMask,
    scalar                      rDeltaT,
    DeviceBuffer<scalar>&       outInt,
    DeviceBuffer<scalar>&       outBnd)
{
    const int nIf = nInternalFaces;
    if (nIf > 0 && (int)phiOldInt.size() >= nIf && (int)fluxUoldInt.size() >= nIf
        && (int)rAUf.size() >= nIf && (int)outInt.size() >= nIf)
    {
        ddtCorrIntK<<<nBlocks(nIf), TPB>>>(nIf, phiOldInt.data(), fluxUoldInt.data(), rAUf.data(),
                                           rDeltaT, outInt.data());
        cudaCheck(cudaGetLastError(), "ddtCorrInt");
    }
    const int nBf = (int)outBnd.size();
    if (nBf > 0 && (int)phiOldBnd.size() >= nBf && (int)fluxUoldBnd.size() >= nBf
        && (int)rAUbnd.size() >= nBf && (int)coeffMask.size() >= nBf)
    {
        ddtCorrBndK<<<nBlocks(nBf), TPB>>>(nBf, phiOldBnd.data(), fluxUoldBnd.data(), rAUbnd.data(),
                                           coeffMask.data(), rDeltaT, outBnd.data());
        cudaCheck(cudaGetLastError(), "ddtCorrBnd");
    }
}

namespace {
__global__ void correctUfK(int n, const label* __restrict__ idx,
                           const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy,
                           const scalar* __restrict__ Sfz, const scalar* __restrict__ magSf,
                           const scalar* __restrict__ phi,
                           scalar* __restrict__ ux, scalar* __restrict__ uy, scalar* __restrict__ uz)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int f = idx ? (int)idx[i] : i;
    const scalar ms = magSf[f];
    if (!(ms > scalar(0))) return;              // a zero-area face (an uncovered ACMI face) has no normal
    const scalar nx = Sfx[f]/ms, ny = Sfy[f]/ms, nz = Sfz[f]/ms;
    const scalar nu = nx*ux[i] + ny*uy[i] + nz*uz[i];
    const scalar c  = phi[i]/ms - nu;
    ux[i] += nx*c;  uy[i] += ny*c;  uz[i] += nz*c;
}

__global__ void dotSfK(int n, const label* __restrict__ idx,
                       const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy,
                       const scalar* __restrict__ Sfz, const scalar* __restrict__ ux,
                       const scalar* __restrict__ uy, const scalar* __restrict__ uz,
                       scalar* __restrict__ out)
{
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int f = idx ? (int)idx[i] : i;
    out[i] = Sfx[f]*ux[i] + Sfy[f]*uy[i] + Sfz[f]*uz[i];
}
}   // namespace

void deviceCorrectUf(
    int n, const label* faceIdx,
    const DeviceBuffer<scalar>& Sfx, const DeviceBuffer<scalar>& Sfy, const DeviceBuffer<scalar>& Sfz,
    const DeviceBuffer<scalar>& magSf, const DeviceBuffer<scalar>& phi,
    DeviceBuffer<scalar>& ufx, DeviceBuffer<scalar>& ufy, DeviceBuffer<scalar>& ufz)
{
    if (n <= 0 || (int)phi.size() < n || (int)ufx.size() < n) return;
    correctUfK<<<nBlocks(n), TPB>>>(n, faceIdx, Sfx.data(), Sfy.data(), Sfz.data(), magSf.data(),
                                    phi.data(), ufx.data(), ufy.data(), ufz.data());
    cudaCheck(cudaGetLastError(), "correctUf");
}

void deviceDotSf(
    int n, const label* faceIdx,
    const DeviceBuffer<scalar>& Sfx, const DeviceBuffer<scalar>& Sfy, const DeviceBuffer<scalar>& Sfz,
    const DeviceBuffer<scalar>& ufx, const DeviceBuffer<scalar>& ufy, const DeviceBuffer<scalar>& ufz,
    DeviceBuffer<scalar>& out)
{
    out.resize(n);
    if (n <= 0 || (int)ufx.size() < n) return;
    dotSfK<<<nBlocks(n), TPB>>>(n, faceIdx, Sfx.data(), Sfy.data(), Sfz.data(),
                                ufx.data(), ufy.data(), ufz.data(), out.data());
    cudaCheck(cudaGetLastError(), "dotSf");
}

}  // namespace brae
