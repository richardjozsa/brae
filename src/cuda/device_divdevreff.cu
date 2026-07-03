// cf GPU offload (#2): explicit divDevReff stress source. Tensors packed component-major (i*3+j)*n + cell.
#include "device_divdevreff.cuh"
#include "device_blas.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

// sigma = nuEff * dev2(transpose(gradU)) per cell. T(gradU)_ij = g[j][i]; dev2 subtracts (2/3)tr on diag.
__global__ void sigmaKernel(int n, const scalar* __restrict__ gradU, const scalar* __restrict__ nu, scalar* __restrict__ sigma) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= n) return;
    scalar g[9];
    for (int q = 0; q < 9; ++q) g[q] = gradU[q * n + c];
    const scalar tr = g[0] + g[4] + g[8];
    const scalar ne = nu[c];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) {
            const scalar T = g[j * 3 + i];                 // transpose
            const scalar d = (i == j) ? (2.0 / 3.0) * tr : 0.0;
            sigma[(i * 3 + j) * n + c] = ne * (T - d);
        }
}

// boundary gradient: gradB = gradC + n (x) (snGrad - n & gradC). snGrad = (U_b - U_cell)*deltaCoeffs.
__global__ void gradBKernel(int nB, const label* __restrict__ fc, const label* __restrict__ bndGFace,
                            const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy, const scalar* __restrict__ Sfz,
                            const scalar* __restrict__ gradU, int nC, const scalar* __restrict__ uxb,
                            const scalar* __restrict__ uyb, const scalar* __restrict__ uzb, const scalar* __restrict__ Ux,
                            const scalar* __restrict__ Uy, const scalar* __restrict__ Uz, const scalar* __restrict__ dc,
                            scalar* __restrict__ gradB) {
    const int bi = blockIdx.x * blockDim.x + threadIdx.x;
    if (bi >= nB) return;
    const int c = fc[bi], f = bndGFace[bi];
    scalar nx = Sfx[f], ny = Sfy[f], nz = Sfz[f];
    const scalar mag = sqrt(nx * nx + ny * ny + nz * nz); nx /= mag; ny /= mag; nz /= mag;
    scalar gc[9]; for (int q = 0; q < 9; ++q) gc[q] = gradU[q * nC + c];
    const scalar nv[3] = { nx, ny, nz };
    const scalar sn[3] = { (uxb[bi]-Ux[c])*dc[bi], (uyb[bi]-Uy[c])*dc[bi], (uzb[bi]-Uz[c])*dc[bi] };
    scalar ngc[3];                                          // (n & gradC)_j = sum_i n_i gc[i][j]
    for (int j = 0; j < 3; ++j) ngc[j] = nv[0]*gc[0*3+j] + nv[1]*gc[1*3+j] + nv[2]*gc[2*3+j];
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) gradB[(i*3+j)*nB + bi] = gc[i*3+j] + nv[i] * (sn[j] - ngc[j]);
}

// tensor divergence gathered per cell: divSig_j = (1/V)[ sum_faces (Sf & sigma_face)_j ].
__global__ void tensorDivKernel(int nC, const label* __restrict__ ownerStart, const label* __restrict__ losort,
                                const label* __restrict__ losortStart, const label* __restrict__ own, const label* __restrict__ nei,
                                const scalar* __restrict__ w, const scalar* __restrict__ Sfx, const scalar* __restrict__ Sfy,
                                const scalar* __restrict__ Sfz, const label* __restrict__ bndCellStart, const label* __restrict__ bndPerm,
                                const label* __restrict__ bndGFace, const label* __restrict__ bndIsEmpty, const scalar* __restrict__ sigmaC,
                                const scalar* __restrict__ sigmaB, int nB, const scalar* __restrict__ V,
                                scalar* __restrict__ dX, scalar* __restrict__ dY, scalar* __restrict__ dZ) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= nC) return;
    scalar d[3] = { 0, 0, 0 };
    // internal faces owned by c (+): sigma_face = w*own + (1-w)*nei
    for (int fi = ownerStart[c]; fi < ownerStart[c + 1]; ++fi) {
        const int o = own[fi], n2 = nei[fi]; const scalar wf = w[fi]; const scalar sx = Sfx[fi], sy = Sfy[fi], sz = Sfz[fi];
        for (int j = 0; j < 3; ++j) {
            const scalar s0 = wf*sigmaC[(0*3+j)*nC+o] + (1.0-wf)*sigmaC[(0*3+j)*nC+n2];
            const scalar s1 = wf*sigmaC[(1*3+j)*nC+o] + (1.0-wf)*sigmaC[(1*3+j)*nC+n2];
            const scalar s2 = wf*sigmaC[(2*3+j)*nC+o] + (1.0-wf)*sigmaC[(2*3+j)*nC+n2];
            d[j] += sx*s0 + sy*s1 + sz*s2;
        }
    }
    // internal faces neighbouring c (-)
    for (int k = losortStart[c]; k < losortStart[c + 1]; ++k) {
        const int fi = losort[k]; const int o = own[fi], n2 = nei[fi]; const scalar wf = w[fi]; const scalar sx = Sfx[fi], sy = Sfy[fi], sz = Sfz[fi];
        for (int j = 0; j < 3; ++j) {
            const scalar s0 = wf*sigmaC[(0*3+j)*nC+o] + (1.0-wf)*sigmaC[(0*3+j)*nC+n2];
            const scalar s1 = wf*sigmaC[(1*3+j)*nC+o] + (1.0-wf)*sigmaC[(1*3+j)*nC+n2];
            const scalar s2 = wf*sigmaC[(2*3+j)*nC+o] + (1.0-wf)*sigmaC[(2*3+j)*nC+n2];
            d[j] -= sx*s0 + sy*s1 + sz*s2;
        }
    }
    // boundary faces (+): sigma_b at the boundary face (empty patches excluded, as in CPU fvc::div)
    for (int k = bndCellStart[c]; k < bndCellStart[c + 1]; ++k) {
        const int bi = bndPerm[k]; if (bndIsEmpty[bi]) continue;
        const int f = bndGFace[bi]; const scalar sx = Sfx[f], sy = Sfy[f], sz = Sfz[f];
        for (int j = 0; j < 3; ++j)
            d[j] += sx*sigmaB[(0*3+j)*nB+bi] + sy*sigmaB[(1*3+j)*nB+bi] + sz*sigmaB[(2*3+j)*nB+bi];
    }
    dX[c] = d[0]; dY[c] = d[1]; dZ[c] = d[2];               // = V*fvc::div (the /V and *V cancel -> raw sum)
}
} // namespace

void deviceDivDevReff(const DeviceMesh& dm, const DeviceVectorBoundary& dbU,
                      const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                      const DeviceBuffer<scalar>& nuCell, const DeviceBuffer<scalar>& nuBnd,
                      DeviceBuffer<scalar>& srcX, DeviceBuffer<scalar>& srcY, DeviceBuffer<scalar>& srcZ,
                      const DeviceCyclic* cyc, const DeviceAMI* ami) {
    const int nC = dm.nCells, nB = dm.nBndFaces;
    const DeviceBuffer<scalar>* Uc[3] = { &Ux, &Uy, &Uz };

    // gradU (packed 9*nC): row i = gaussGrad(U_i).
    DeviceBuffer<scalar> gradU(static_cast<std::size_t>(9) * nC);
    DeviceBuffer<scalar> uxb, uyb, uzb;  // boundary values (reused for snGrad)
    DeviceBuffer<scalar>* ub[3] = { &uxb, &uyb, &uzb };
    // rotational AMI: the gradU neighbour is the ROTATED AMI-interpolated U (forwardT·interp), precompute it.
    DeviceBuffer<scalar> amiUNx, amiUNy, amiUNz;
    if (ami && ami->rotational) deviceAmiInterpolateVec(*ami, Ux, Uy, Uz, amiUNx, amiUNy, amiUNz);
    DeviceBuffer<scalar>* amiUN[3] = { &amiUNx, &amiUNy, &amiUNz };
    // gaussGrad(U_i) = (dUi/dx, dUi/dy, dUi/dz) = COLUMN i of G (OF convention G_ij = dU_j/dx_i, via outer(Sf,U)).
    for (int i = 0; i < 3; ++i) {
        DeviceBuffer<scalar> bval; deviceBCValue(dbU.comp[i], *Uc[i], bval);
        DeviceBuffer<scalar> gx, gy, gz; deviceGaussGrad(dm, *Uc[i], bval, gx, gy, gz);
        if (cyc) {                                                     // + cyclic faces in gradU (periodic consistency)
            if (cyc->rotational) deviceCyclicAddGradRot(*cyc, Ux, Uy, Uz, i, dm.V, gx, gy, gz);   // neighbour rotated
            else deviceCyclicAddGrad(*cyc, *Uc[i], dm.V, gx, gy, gz);
        }
        if (ami) { if (ami->rotational) deviceAmiAddGradRot(*ami, *Uc[i], *amiUN[i], dm.V, gx, gy, gz);   // neighbour rotated
                   else deviceAmiAddGrad(*ami, *Uc[i], dm.V, gx, gy, gz); }
        cudaCheck(cudaMemcpy(gradU.data() + (0*3+i)*nC, gx.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice), "ddr g");
        cudaCheck(cudaMemcpy(gradU.data() + (1*3+i)*nC, gy.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice), "ddr g");
        cudaCheck(cudaMemcpy(gradU.data() + (2*3+i)*nC, gz.data(), nC*sizeof(scalar), cudaMemcpyDeviceToDevice), "ddr g");
        *ub[i] = std::move(bval);
    }
    // sigma cell
    DeviceBuffer<scalar> sigmaC(static_cast<std::size_t>(9) * nC);
    sigmaKernel<<<nBlocks(nC), TPB>>>(nC, gradU.data(), nuCell.data(), sigmaC.data());
    cudaCheck(cudaGetLastError(), "ddr sigmaC");
    // boundary gradient + sigma boundary
    DeviceBuffer<scalar> gradB(static_cast<std::size_t>(9) * nB);
    gradBKernel<<<nBlocks(nB), TPB>>>(nB, dm.bndCell.data(), dm.bndGFace.data(), dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(),
                                      gradU.data(), nC, uxb.data(), uyb.data(), uzb.data(), Ux.data(), Uy.data(), Uz.data(),
                                      /* bnd deltaCoeffs */ dbU.comp[0].deltaCoeffs.data(), gradB.data());
    cudaCheck(cudaGetLastError(), "ddr gradB");
    DeviceBuffer<scalar> sigmaB(static_cast<std::size_t>(9) * nB);
    sigmaKernel<<<nBlocks(nB), TPB>>>(nB, gradB.data(), nuBnd.data(), sigmaB.data());
    cudaCheck(cudaGetLastError(), "ddr sigmaB");
    // tensor divergence (= V*fvc::div)
    srcX.resize(nC); srcY.resize(nC); srcZ.resize(nC);
    tensorDivKernel<<<nBlocks(nC), TPB>>>(nC, dm.ownerStart.data(), dm.losort.data(), dm.losortStart.data(), dm.owner.data(),
                                          dm.nei.data(), dm.w.data(), dm.Sfx.data(), dm.Sfy.data(), dm.Sfz.data(),
                                          dm.bndCellStart.data(), dm.bndPerm.data(), dm.bndGFace.data(), dm.bndIsEmpty.data(),
                                          sigmaC.data(), sigmaB.data(), nB, dm.V.data(), srcX.data(), srcY.data(), srcZ.data());
    cudaCheck(cudaGetLastError(), "ddr tensorDiv");
    if (cyc) deviceCyclicAddTensorDiv(*cyc, sigmaC, nC, srcX, srcY, srcZ);   // + cyclic-face stress flux (V*fvc::div)
    if (ami) deviceAmiAddTensorDiv(*ami, sigmaC, nC, srcX, srcY, srcZ);      // + AMI-face stress flux
}

} // namespace brae
