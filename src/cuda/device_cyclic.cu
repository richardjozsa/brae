// cf GPU offload, cyclic (periodic) interface kernels. See device_cyclic.cuh. Both sides of each periodic
// pair are stored with their OWN outward Sf and own flux, so the coupling/continuity are symmetric and
// conservative automatically (the N-side flux = -the O-side flux; the N-side weight = 1 - the O-side weight).
// Sign convention matches device_fvm.cu: off-diagonal ifCoeff is added in Amul as Apsi[own]+=ifCoeff*psi[nbr];
// diag[own] -= ifCoeff (Laplacian) / += max(phi,0) (upwind convection).
#include "device_cyclic.cuh"
#include <cuda_runtime.h>

namespace brae {
namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }


__global__
void laplKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ gamma,
    const scalar* __restrict__ dc,
    const scalar* __restrict__ w,
    const scalar* __restrict__ magSf,
    scalar* __restrict__ ifCoeff,
    scalar* __restrict__ diag,
    int addToDiag)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const scalar gf = w[j] * gamma[own[j]] + (1.0 - w[j]) * gamma[nbr[j]];   // gamma interpolated to the face
    const scalar c = gf * dc[j] * magSf[j];
    ifCoeff[j] = c;
    if (addToDiag) atomicAdd(&diag[own[j]], -c);
}


__global__
void convKernel(
    int n,
    const label* __restrict__ own,
    const scalar* __restrict__ phi,
    scalar* __restrict__ ifCoeff,
    scalar* __restrict__ diag)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const scalar p = phi[j];
    ifCoeff[j] += (p < 0.0) ? p : 0.0;                  // off-diag += min(phi,0)
    atomicAdd(&diag[own[j]], (p > 0.0) ? p : 0.0);      // diag    += max(phi,0)
}


__global__
void momKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ nu,
    const scalar* __restrict__ dc,
    const scalar* __restrict__ w,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ phi,
    scalar* __restrict__ ifCoeff,
    scalar* __restrict__ diag)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const scalar nf = w[j] * nu[own[j]] + (1.0 - w[j]) * nu[nbr[j]];
    const scalar lap = nf * dc[j] * magSf[j];           // diffusion magnitude (>0)
    const scalar p = phi[j];
    ifCoeff[j] = -lap + ((p < 0.0) ? p : 0.0);          // off-diag: -laplacian + min(phi,0)
    atomicAdd(&diag[own[j]], lap + ((p > 0.0) ? p : 0.0));   // diag: +laplacian + max(phi,0)
}


__global__
void addHKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ ifCoeff,
    const scalar* __restrict__ psi,
    const scalar* __restrict__ V,
    scalar* __restrict__ H)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    atomicAdd(&H[own[j]], -ifCoeff[j] * psi[nbr[j]] / V[own[j]]);
}


__global__
void offSumKernel(
    int n,
    const label* __restrict__ own,
    const scalar* __restrict__ ifCoeff,
    scalar* __restrict__ sumOff)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    atomicAdd(&sumOff[own[j]], fabs(ifCoeff[j]));
}


__global__
void fluxKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ Hx,
    const scalar* __restrict__ Hy,
    const scalar* __restrict__ Hz,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    scalar* __restrict__ phi)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const label o = own[j], nb = nbr[j];
    const scalar wj = w[j], wn = 1.0 - wj;
    const scalar fx = wj * Hx[o] + wn * Hx[nb], fy = wj * Hy[o] + wn * Hy[nb], fz = wj * Hz[o] + wn * Hz[nb];
    phi[j] = fx * Sfx[j] + fy * Sfy[j] + fz * Sfz[j];
}


__global__
void divAddKernel(
    int n,
    const label* __restrict__ own,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ V,
    scalar* __restrict__ div)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    atomicAdd(&div[own[j]], phi[j] / V[own[j]]);   // deviceDiv returns the volume-normalized divergence (Sum phi / V)
}


__global__
void fluxCorrKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ ifCoeff,
    const scalar* __restrict__ p,
    scalar* __restrict__ phi)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    phi[j] -= ifCoeff[j] * (p[nbr[j]] - p[own[j]]);     // snGrad(p) flux: -coeff*(p_nbr - p_own)
}


__global__
void cycTensorDivKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const scalar* __restrict__ sigmaC,
    int nC,
    const scalar* __restrict__ fT,
    int rotational,
    scalar* __restrict__ dX,
    scalar* __restrict__ dY,
    scalar* __restrict__ dZ)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const int o = own[j], nb = nbr[j];
    const scalar wf = w[j], sx = Sfx[j], sy = Sfy[j], sz = Sfz[j];
    // neighbour stress tensor; for ROTATIONAL cyclic a rank-2 tensor transforms as sigma' = R*sigma*R^T (NOT the
    // vector rotation) before interpolation to the face, else the cyclic divDevReff injects a spurious stress.
    scalar sn[9];
    for (int q = 0; q < 9; ++q)
        sn[q] = sigmaC[q*nC + nb];
    if (rotational)
    {
        scalar R[9];
        for (int q = 0; q < 9; ++q)
            R[q] = fT[q*n + j];
        scalar M[9];   // M = R*sigma
        for (int i = 0; i < 3; ++i)
            for (int l = 0; l < 3; ++l)
            {
                scalar s = 0;
                for (int k = 0; k < 3; ++k)
                    s += R[3*i+k]*sn[3*k+l];
                M[3*i+l] = s;
            }
        scalar sr[9];   // sigma' = M*R^T : sr[i][jj] = sum_l M[i][l]*R[jj][l]
        for (int i = 0; i < 3; ++i)
            for (int jj = 0; jj < 3; ++jj)
            {
                scalar s = 0;
                for (int l = 0; l < 3; ++l)
                    s += M[3*i+l]*R[3*jj+l];
                sr[3*i+jj] = s;
            }
        for (int q = 0; q < 9; ++q)
            sn[q] = sr[q];
    }
    scalar d[3];
    for (int jc = 0; jc < 3; ++jc)   // sigma_face = w*sigma[own] + (1-w)*sigma'[nbr]
    {
        const scalar s0 = wf*sigmaC[(0*3+jc)*nC+o] + (1.0-wf)*sn[0*3+jc];
        const scalar s1 = wf*sigmaC[(1*3+jc)*nC+o] + (1.0-wf)*sn[1*3+jc];
        const scalar s2 = wf*sigmaC[(2*3+jc)*nC+o] + (1.0-wf)*sn[2*3+jc];
        d[jc] = sx*s0 + sy*s1 + sz*s2;
    }
    // raw sum (= V*fvc::div), like tensorDivKernel
    atomicAdd(&dX[o], d[0]);
    atomicAdd(&dY[o], d[1]);
    atomicAdd(&dZ[o], d[2]);
}


__global__
void gradAddKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ psi,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const scalar* __restrict__ V,
    scalar* __restrict__ gx,
    scalar* __restrict__ gy,
    scalar* __restrict__ gz)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const label o = own[j];
    const scalar fv = (w[j] * psi[o] + (1.0 - w[j]) * psi[nbr[j]]) / V[o];
    atomicAdd(&gx[o], Sfx[j] * fv);
    atomicAdd(&gy[o], Sfy[j] * fv);
    atomicAdd(&gz[o], Sfz[j] * fv);
}


// rotational helpers. forwardT packed (3*i+j)*n + face ; (forwardT.v)[i] = sum_k fT[(3i+k)n+j]*v_k
__device__ __forceinline__
void rotNbr(const scalar* fT, int n, int j, scalar vx, scalar vy, scalar vz,
            scalar& rx, scalar& ry, scalar& rz)
{
    rx = fT[(0*3+0)*n+j]*vx + fT[(0*3+1)*n+j]*vy + fT[(0*3+2)*n+j]*vz;
    ry = fT[(1*3+0)*n+j]*vx + fT[(1*3+1)*n+j]*vy + fT[(1*3+2)*n+j]*vz;
    rz = fT[(2*3+0)*n+j]*vx + fT[(2*3+1)*n+j]*vy + fT[(2*3+2)*n+j]*vz;
}


__global__
void scaleImplicitKernel(
    int n,
    const scalar* __restrict__ ifc,
    const scalar* __restrict__ fT,
    scalar* __restrict__ ic0,
    scalar* __restrict__ ic1,
    scalar* __restrict__ ic2)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    ic0[j] = ifc[j] * fT[(0*3+0)*n+j];   // * forwardT[0][0]
    ic1[j] = ifc[j] * fT[(1*3+1)*n+j];   // * forwardT[1][1]
    ic2[j] = ifc[j] * fT[(2*3+2)*n+j];   // * forwardT[2][2]
}


__global__
void fluxRotKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ Hx,
    const scalar* __restrict__ Hy,
    const scalar* __restrict__ Hz,
    const scalar* __restrict__ fT,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    scalar* __restrict__ phi)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const label o = own[j], nb = nbr[j];
    const scalar wj = w[j], wn = 1.0 - wj;
    scalar rx, ry, rz;
    rotNbr(fT, n, j, Hx[nb], Hy[nb], Hz[nb], rx, ry, rz);
    const scalar fx = wj*Hx[o] + wn*rx, fy = wj*Hy[o] + wn*ry, fz = wj*Hz[o] + wn*rz;
    phi[j] = fx*Sfx[j] + fy*Sfy[j] + fz*Sfz[j];
}


__global__
void addHRotKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ ifc,
    const scalar* __restrict__ fT,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ V,
    scalar* __restrict__ Hx,
    scalar* __restrict__ Hy,
    scalar* __restrict__ Hz)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const label o = own[j], nb = nbr[j];
    scalar rx, ry, rz;
    rotNbr(fT, n, j, Ux[nb], Uy[nb], Uz[nb], rx, ry, rz);
    const scalar c = ifc[j] / V[o];
    atomicAdd(&Hx[o], -c*rx);
    atomicAdd(&Hy[o], -c*ry);
    atomicAdd(&Hz[o], -c*rz);
}


__global__
void gradRotKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ w,
    const scalar* __restrict__ fT,
    int comp,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    const scalar* __restrict__ Sfx,
    const scalar* __restrict__ Sfy,
    const scalar* __restrict__ Sfz,
    const scalar* __restrict__ V,
    scalar* __restrict__ gx,
    scalar* __restrict__ gy,
    scalar* __restrict__ gz)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const label o = own[j], nb = nbr[j];
    scalar rx, ry, rz;
    rotNbr(fT, n, j, Ux[nb], Uy[nb], Uz[nb], rx, ry, rz);
    const scalar uOwn = (comp==0)?Ux[o]:(comp==1)?Uy[o]:Uz[o];
    const scalar uNbrR = (comp==0)?rx:(comp==1)?ry:rz;            // (forwardT.U[nbr])[comp]
    const scalar fv = (w[j]*uOwn + (1.0-w[j])*uNbrR) / V[o];
    atomicAdd(&gx[o], Sfx[j]*fv);
    atomicAdd(&gy[o], Sfy[j]*fv);
    atomicAdd(&gz[o], Sfz[j]*fv);
}


__global__
void deferredRotKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ ifc,
    const scalar* __restrict__ fT,
    int comp,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar* __restrict__ src)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const label o = own[j], nb = nbr[j];
    scalar rx, ry, rz;
    rotNbr(fT, n, j, Ux[nb], Uy[nb], Uz[nb], rx, ry, rz);
    const scalar uNbrComp = (comp==0)?Ux[nb]:(comp==1)?Uy[nb]:Uz[nb];
    const scalar diag     = fT[(3*comp+comp)*n+j];               // forwardT[comp][comp]
    const scalar rComp    = (comp==0)?rx:(comp==1)?ry:rz;        // (forwardT.U[nbr])[comp]
    atomicAdd(&src[o], -ifc[j] * (rComp - diag*uNbrComp));       // -ifc*(full - diag) mixing
}


__global__
void addHDiagKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ ifcC,
    const scalar* __restrict__ psi,
    const scalar* __restrict__ V,
    scalar* __restrict__ H)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    atomicAdd(&H[own[j]], -ifcC[j] * psi[nbr[j]] / V[own[j]]);   // diag cyclic off-diag (ifCoeffC[comp])
}
} // namespace


void deviceCyclicAssembleLaplacian(
    DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& gammaCell,
    DeviceBuffer<scalar>& diag,
    bool addToDiag)
{
    if (cyc.n == 0) return;
    laplKernel<<<nBlocks(cyc.n), TPB>>>(cyc.n, cyc.ownCell.data(), cyc.nbrCell.data(), gammaCell.data(),
        cyc.deltaCoeffs.data(), cyc.weights.data(), cyc.magSf.data(), cyc.ifCoeff.data(), diag.data(), addToDiag ? 1 : 0);
    cudaCheck(cudaGetLastError(), "cyclicLapl");
}


void deviceCyclicAddConvection(DeviceCyclic& cyc, DeviceBuffer<scalar>& diag)
{
    if (cyc.n == 0) return;
    convKernel<<<nBlocks(cyc.n), TPB>>>(cyc.n, cyc.ownCell.data(), cyc.phi.data(), cyc.ifCoeff.data(), diag.data());
    cudaCheck(cudaGetLastError(), "cyclicConv");
}


void deviceCyclicAssembleMomentum(DeviceCyclic& cyc, const DeviceBuffer<scalar>& nuEffCell, DeviceBuffer<scalar>& diag)
{
    if (cyc.n == 0) return;
    momKernel<<<nBlocks(cyc.n), TPB>>>(cyc.n, cyc.ownCell.data(), cyc.nbrCell.data(), nuEffCell.data(),
        cyc.deltaCoeffs.data(), cyc.weights.data(), cyc.magSf.data(), cyc.phi.data(), cyc.ifCoeff.data(), diag.data());
    cudaCheck(cudaGetLastError(), "cyclicMom");
}


void deviceCyclicAddH(
    const DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& psi,
    const DeviceBuffer<scalar>& V,
    DeviceBuffer<scalar>& H)
{
    if (cyc.n == 0) return;
    addHKernel<<<nBlocks(cyc.n), TPB>>>(cyc.n, cyc.ownCell.data(), cyc.nbrCell.data(), cyc.ifCoeff.data(),
        psi.data(), V.data(), H.data());
    cudaCheck(cudaGetLastError(), "cyclicAddH");
}


void deviceCyclicOffDiagSum(const DeviceCyclic& cyc, DeviceBuffer<scalar>& sumOff)
{
    if (cyc.n == 0) return;
    offSumKernel<<<nBlocks(cyc.n), TPB>>>(cyc.n, cyc.ownCell.data(), cyc.ifCoeff.data(), sumOff.data());
    cudaCheck(cudaGetLastError(), "cyclicOffSum");
}


void deviceCyclicFlux(
    DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& Hx,
    const DeviceBuffer<scalar>& Hy,
    const DeviceBuffer<scalar>& Hz)
{
    if (cyc.n == 0) return;
    fluxKernel<<<nBlocks(cyc.n), TPB>>>(cyc.n, cyc.ownCell.data(), cyc.nbrCell.data(), cyc.weights.data(),
        Hx.data(), Hy.data(), Hz.data(), cyc.Sfx.data(), cyc.Sfy.data(), cyc.Sfz.data(), cyc.phi.data());
    cudaCheck(cudaGetLastError(), "cyclicFlux");
}


void deviceCyclicAddDiv(const DeviceCyclic& cyc, const DeviceBuffer<scalar>& V, DeviceBuffer<scalar>& div)
{
    if (cyc.n == 0) return;
    divAddKernel<<<nBlocks(cyc.n), TPB>>>(cyc.n, cyc.ownCell.data(), cyc.phi.data(), V.data(), div.data());
    cudaCheck(cudaGetLastError(), "cyclicDiv");
}


namespace {
__global__
void cycZeroWallKernel(int n, const label* __restrict__ own, const label* __restrict__ isW, scalar* __restrict__ ifc)
{
    const int j = blockIdx.x*blockDim.x+threadIdx.x;
    if (j<n && isW[own[j]]) ifc[j]=0.0;
}
} // namespace


// epsilon setValues: a wall cell's eps is fixed (eps0); zero the cyclic interface off-diagonal for wall-cell owners.
void deviceCyclicZeroWallIfCoeff(DeviceCyclic& cyc, const DeviceBuffer<label>& isWallCell)
{
    if (cyc.n == 0) return;
    cycZeroWallKernel<<<nBlocks(cyc.n), TPB>>>(cyc.n, cyc.ownCell.data(), isWallCell.data(), cyc.ifCoeff.data());
    cudaCheck(cudaGetLastError(), "cyclicZeroWall");
}


void deviceCyclicCorrectFlux(DeviceCyclic& cyc, const DeviceBuffer<scalar>& p)
{
    if (cyc.n == 0) return;
    fluxCorrKernel<<<nBlocks(cyc.n), TPB>>>(cyc.n, cyc.ownCell.data(), cyc.nbrCell.data(), cyc.ifCoeff.data(),
        p.data(), cyc.phi.data());
    cudaCheck(cudaGetLastError(), "cyclicFluxCorr");
}


void deviceCyclicAddGrad(
    const DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& psi,
    const DeviceBuffer<scalar>& V,
    DeviceBuffer<scalar>& gx,
    DeviceBuffer<scalar>& gy,
    DeviceBuffer<scalar>& gz)
{
    if (cyc.n == 0) return;
    gradAddKernel<<<nBlocks(cyc.n), TPB>>>(cyc.n, cyc.ownCell.data(), cyc.nbrCell.data(), cyc.weights.data(),
        psi.data(), cyc.Sfx.data(), cyc.Sfy.data(), cyc.Sfz.data(), V.data(), gx.data(), gy.data(), gz.data());
    cudaCheck(cudaGetLastError(), "cyclicGrad");
}


void deviceCyclicScaleImplicit(DeviceCyclic& cyc)
{
    if (cyc.n == 0) return;
    scaleImplicitKernel<<<nBlocks(cyc.n), TPB>>>(cyc.n, cyc.ifCoeff.data(), cyc.fT.data(),
        cyc.ifCoeffC[0].data(), cyc.ifCoeffC[1].data(), cyc.ifCoeffC[2].data());
    cudaCheck(cudaGetLastError(), "cyclicScaleImplicit");
}


void deviceCyclicFluxRot(
    DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& Hx,
    const DeviceBuffer<scalar>& Hy,
    const DeviceBuffer<scalar>& Hz)
{
    if (cyc.n == 0) return;
    fluxRotKernel<<<nBlocks(cyc.n), TPB>>>(cyc.n, cyc.ownCell.data(), cyc.nbrCell.data(), cyc.weights.data(),
        Hx.data(), Hy.data(), Hz.data(), cyc.fT.data(), cyc.Sfx.data(), cyc.Sfy.data(), cyc.Sfz.data(), cyc.phi.data());
    cudaCheck(cudaGetLastError(), "cyclicFluxRot");
}


void deviceCyclicAddHRot(
    const DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& V,
    DeviceBuffer<scalar>& Hx,
    DeviceBuffer<scalar>& Hy,
    DeviceBuffer<scalar>& Hz)
{
    if (cyc.n == 0) return;
    addHRotKernel<<<nBlocks(cyc.n), TPB>>>(cyc.n, cyc.ownCell.data(), cyc.nbrCell.data(), cyc.ifCoeff.data(),
        cyc.fT.data(), Ux.data(), Uy.data(), Uz.data(), V.data(), Hx.data(), Hy.data(), Hz.data());
    cudaCheck(cudaGetLastError(), "cyclicAddHRot");
}


void deviceCyclicAddGradRot(
    const DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    int comp,
    const DeviceBuffer<scalar>& V,
    DeviceBuffer<scalar>& gx,
    DeviceBuffer<scalar>& gy,
    DeviceBuffer<scalar>& gz)
{
    if (cyc.n == 0) return;
    gradRotKernel<<<nBlocks(cyc.n), TPB>>>(cyc.n, cyc.ownCell.data(), cyc.nbrCell.data(), cyc.weights.data(),
        cyc.fT.data(), comp, Ux.data(), Uy.data(), Uz.data(), cyc.Sfx.data(), cyc.Sfy.data(), cyc.Sfz.data(),
        V.data(), gx.data(), gy.data(), gz.data());
    cudaCheck(cudaGetLastError(), "cyclicGradRot");
}


void deviceCyclicAddDeferredRot(
    const DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    int comp,
    DeviceBuffer<scalar>& src)
{
    if (cyc.n == 0) return;
    deferredRotKernel<<<nBlocks(cyc.n), TPB>>>(cyc.n, cyc.ownCell.data(), cyc.nbrCell.data(), cyc.ifCoeff.data(),
        cyc.fT.data(), comp, Ux.data(), Uy.data(), Uz.data(), src.data());
    cudaCheck(cudaGetLastError(), "cyclicDeferredRot");
}


void deviceCyclicAddHDiag(
    const DeviceCyclic& cyc,
    int comp,
    const DeviceBuffer<scalar>& psi,
    const DeviceBuffer<scalar>& V,
    DeviceBuffer<scalar>& H)
{
    if (cyc.n == 0) return;
    addHDiagKernel<<<nBlocks(cyc.n), TPB>>>(cyc.n, cyc.ownCell.data(), cyc.nbrCell.data(), cyc.ifCoeffC[comp].data(),
        psi.data(), V.data(), H.data());
    cudaCheck(cudaGetLastError(), "cyclicAddHDiag");
}


__global__
void cycLinUpwindKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ phi,
    const scalar* __restrict__ gx0,
    const scalar* __restrict__ gy0,
    const scalar* __restrict__ gz0,
    const scalar* __restrict__ gx1,
    const scalar* __restrict__ gy1,
    const scalar* __restrict__ gz1,
    const scalar* __restrict__ gx2,
    const scalar* __restrict__ gy2,
    const scalar* __restrict__ gz2,
    const scalar* __restrict__ dox,
    const scalar* __restrict__ doy,
    const scalar* __restrict__ doz,
    const scalar* __restrict__ dnx,
    const scalar* __restrict__ dny,
    const scalar* __restrict__ dnz,
    const scalar* __restrict__ fT,
    int rotational,
    int comp,
    scalar* __restrict__ corr)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const int o = own[j], nb = nbr[j];
    const scalar pf = phi[j];
    scalar c;
    if (pf >= 0)   // own is upwind: grad(U_comp)[own] . dOwn  (no rotation)
    {
        const scalar* gx = (comp==0)?gx0:(comp==1)?gx1:gx2;
        const scalar* gy = (comp==0)?gy0:(comp==1)?gy1:gy2;
        const scalar* gz = (comp==0)?gz0:(comp==1)?gz1:gz2;
        c = pf * (gx[o]*dox[j] + gy[o]*doy[j] + gz[o]*doz[j]);
    }
    else   // nbr is upwind: forwardT . (gradU[nbr] . dNbr), take comp
    {
        const scalar rl0 = gx0[nb]*dnx[j] + gy0[nb]*dny[j] + gz0[nb]*dnz[j];   // grad(U_l)[nbr] . dNbr (nbr frame)
        const scalar rl1 = gx1[nb]*dnx[j] + gy1[nb]*dny[j] + gz1[nb]*dnz[j];
        const scalar rl2 = gx2[nb]*dnx[j] + gy2[nb]*dny[j] + gz2[nb]*dnz[j];
        c = rotational ? pf * (fT[(3*comp+0)*n+j]*rl0 + fT[(3*comp+1)*n+j]*rl1 + fT[(3*comp+2)*n+j]*rl2)
                       : pf * ((comp==0)?rl0:(comp==1)?rl1:rl2);
    }
    atomicAdd(&corr[o], c);                             // own is the "+" side (mirrors the internal owner pass)
}


void deviceCyclicAddLinUpwindCorr(
    const DeviceCyclic& cyc,
    int comp,
    const DeviceBuffer<scalar>* gUx,
    const DeviceBuffer<scalar>* gUy,
    const DeviceBuffer<scalar>* gUz,
    DeviceBuffer<scalar>& corr)
{
    if (cyc.n == 0) return;
    cycLinUpwindKernel<<<nBlocks(cyc.n), TPB>>>(cyc.n, cyc.ownCell.data(), cyc.nbrCell.data(), cyc.phi.data(),
        gUx[0].data(), gUy[0].data(), gUz[0].data(), gUx[1].data(), gUy[1].data(), gUz[1].data(),
        gUx[2].data(), gUy[2].data(), gUz[2].data(),
        cyc.dOwnX.data(), cyc.dOwnY.data(), cyc.dOwnZ.data(), cyc.dNbrX.data(), cyc.dNbrY.data(), cyc.dNbrZ.data(),
        cyc.rotational ? cyc.fT.data() : nullptr, cyc.rotational ? 1 : 0, comp, corr.data());
    cudaCheck(cudaGetLastError(), "cyclicLinUpwind");
}


__global__
void cycLapCorrKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ nu,
    const scalar* __restrict__ w,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ cvx,
    const scalar* __restrict__ cvy,
    const scalar* __restrict__ cvz,
    const scalar* __restrict__ gx0,
    const scalar* __restrict__ gy0,
    const scalar* __restrict__ gz0,
    const scalar* __restrict__ gx1,
    const scalar* __restrict__ gy1,
    const scalar* __restrict__ gz1,
    const scalar* __restrict__ gx2,
    const scalar* __restrict__ gy2,
    const scalar* __restrict__ gz2,
    const scalar* __restrict__ fT,
    int rotational,
    int comp,
    scalar* __restrict__ src)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const int o = own[j], nb = nbr[j];
    const scalar wf = w[j], wn = 1.0 - wf;
    const scalar* gxc = (comp==0)?gx0:(comp==1)?gx1:gx2;
    const scalar* gyc = (comp==0)?gy0:(comp==1)?gy1:gy2;
    const scalar* gzc = (comp==0)?gz0:(comp==1)?gz1:gz2;
    scalar gnx, gny, gnz;                              // grad(U_comp)[nbr] in the OWN frame (rotated)
    if (rotational)                                    // grad((R.U)_comp) = (R . gradU[nbr] . R^T)[comp][:]
    {
        const scalar G[9] = { gx0[nb],gy0[nb],gz0[nb], gx1[nb],gy1[nb],gz1[nb], gx2[nb],gy2[nb],gz2[nb] };
        scalar R[9];
        for (int q = 0; q < 9; ++q)
            R[q] = fT[q*n + j];
        scalar gn[3];
        for (int jj = 0; jj < 3; ++jj)
        {
            scalar s = 0;
            for (int i = 0; i < 3; ++i)
                for (int k = 0; k < 3; ++k)
                    s += R[3*comp+i]*G[3*i+k]*R[3*jj+k];
            gn[jj] = s;
        }
        gnx = gn[0]; gny = gn[1]; gnz = gn[2];
    }
    else { gnx = gxc[nb]; gny = gyc[nb]; gnz = gzc[nb]; }
    const scalar gfx = wf*gxc[o] + wn*gnx, gfy = wf*gyc[o] + wn*gny, gfz = wf*gzc[o] + wn*gnz;
    const scalar gammaf = wf*nu[o] + wn*nu[nb];
    const scalar ffc = gammaf * magSf[j] * (cvx[j]*gfx + cvy[j]*gfy + cvz[j]*gfz);
    atomicAdd(&src[o], -ffc);                          // owner contribution (gather: src[c] -= ffc for owned faces)
}


void deviceCyclicAddLapCorr(
    const DeviceCyclic& cyc,
    int comp,
    const DeviceBuffer<scalar>& gammaCell,
    const DeviceBuffer<scalar>* gUx,
    const DeviceBuffer<scalar>* gUy,
    const DeviceBuffer<scalar>* gUz,
    DeviceBuffer<scalar>& corr)
{
    if (cyc.n == 0) return;
    cycLapCorrKernel<<<nBlocks(cyc.n), TPB>>>(cyc.n, cyc.ownCell.data(), cyc.nbrCell.data(), gammaCell.data(),
        cyc.weights.data(), cyc.magSf.data(), cyc.corrVecX.data(), cyc.corrVecY.data(), cyc.corrVecZ.data(),
        gUx[0].data(), gUy[0].data(), gUz[0].data(), gUx[1].data(), gUy[1].data(), gUz[1].data(),
        gUx[2].data(), gUy[2].data(), gUz[2].data(),
        cyc.rotational ? cyc.fT.data() : nullptr, cyc.rotational ? 1 : 0, comp, corr.data());
    cudaCheck(cudaGetLastError(), "cyclicLapCorr");
}


__global__
void cycLapCorrScalarKernel(
    int n,
    const label* __restrict__ own,
    const label* __restrict__ nbr,
    const scalar* __restrict__ gamma,
    const scalar* __restrict__ w,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ cvx,
    const scalar* __restrict__ cvy,
    const scalar* __restrict__ cvz,
    const scalar* __restrict__ gx,
    const scalar* __restrict__ gy,
    const scalar* __restrict__ gz,
    scalar* __restrict__ bp,
    scalar* __restrict__ ffcOut)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const int o = own[j], nb = nbr[j];
    const scalar wf = w[j], wn = 1.0 - wf;
    const scalar gfx = wf*gx[o]+wn*gx[nb], gfy = wf*gy[o]+wn*gy[nb], gfz = wf*gz[o]+wn*gz[nb];   // grad(p)_face (scalar)
    const scalar gammaf = wf*gamma[o] + wn*gamma[nb];                                            // rAtU_face
    const scalar ffc = gammaf * magSf[j] * (cvx[j]*gfx + cvy[j]*gfy + cvz[j]*gfz);
    ffcOut[j] = ffc;                                    // for the post-solve flux correction (cyc.phi -= ffc)
    atomicAdd(&bp[o], -ffc);                            // -V*div(ffc): owner contribution to the pressure source
}


void deviceCyclicLapCorrP(
    const DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& gammaCell,
    const DeviceBuffer<scalar>& gx,
    const DeviceBuffer<scalar>& gy,
    const DeviceBuffer<scalar>& gz,
    DeviceBuffer<scalar>& bp,
    DeviceBuffer<scalar>& ffcOut)
{
    if (cyc.n == 0) return;
    ffcOut.resize(cyc.n);
    cycLapCorrScalarKernel<<<nBlocks(cyc.n), TPB>>>(cyc.n, cyc.ownCell.data(), cyc.nbrCell.data(), gammaCell.data(),
        cyc.weights.data(), cyc.magSf.data(), cyc.corrVecX.data(), cyc.corrVecY.data(), cyc.corrVecZ.data(),
        gx.data(), gy.data(), gz.data(), bp.data(), ffcOut.data());
    cudaCheck(cudaGetLastError(), "cyclicLapCorrP");
}


void deviceCyclicAddTensorDiv(
    const DeviceCyclic& cyc,
    const DeviceBuffer<scalar>& sigmaC,
    int nC,
    DeviceBuffer<scalar>& srcX,
    DeviceBuffer<scalar>& srcY,
    DeviceBuffer<scalar>& srcZ)
{
    if (cyc.n == 0) return;
    const scalar* fT = cyc.rotational ? cyc.fT.data() : nullptr;
    cycTensorDivKernel<<<nBlocks(cyc.n), TPB>>>(cyc.n, cyc.ownCell.data(), cyc.nbrCell.data(), cyc.weights.data(),
        cyc.Sfx.data(), cyc.Sfy.data(), cyc.Sfz.data(), sigmaC.data(), nC, fT, cyc.rotational ? 1 : 0,
        srcX.data(), srcY.data(), srcZ.data());
    cudaCheck(cudaGetLastError(), "cyclicTensorDiv");
}

} // namespace brae
