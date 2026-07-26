// cf GPU offload -- boundary contributions to the DISCRETISED matrix/fluxes: the valueInternal/valueBoundary
// coeffs of each BC category (fvPatchField), for the value, laplacian, div and matrix-flux operators. One thread
// per boundary face; the BC category (bcType) selects the formula. Split from device_boundary.cu (the per-iteration
// flow-BC value updates are in device_boundary_flow.cu). Shared internal decls: device_boundary.cuh.
#include "device_boundary.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }


__global__
void bcValueKernel(
    int n,
    const label* __restrict__ type,
    const scalar* __restrict__ ref,
    const scalar* __restrict__ vf,
    const label* __restrict__ fc,
    const scalar* __restrict__ internal,
    scalar* __restrict__ value)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (type[i] == 8)      return;                                                       // coupled (processor): the
                                                                                         // face value is the halo-
                                                                                         // interpolated one, injected
                                                                                         // by DeviceHalo::scatterBoundaryValues -- never derived from the local cell.
    else if (type[i] == 0) value[i] = internal[fc[i]];                                   // extrapolated -> internal cell
    else if (type[i] == 5) value[i] = (1.0 - vf[i]) * internal[fc[i]] + vf[i] * ref[i];  // mixed (Robin)
    else                   value[i] = ref[i];                                            // fixedValue / calculated
}


// mixed-aware laplacian gradient weight: w = vf (fixedValue vf=1 -> gradIC=-dc; zeroGradient vf=0 -> 0). vf[i] is
// read ONLY for type==5 (the ternary short-circuits), so a non-mixed boundary with no valueFraction is safe.
__global__
void bcLaplacianKernel(
    int n,
    const label* __restrict__ type,
    const scalar* __restrict__ ref,
    const scalar* __restrict__ vf,
    const scalar* __restrict__ dc,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ gammaCell,
    const label* __restrict__ fc,
    scalar* __restrict__ iC,
    scalar* __restrict__ bC)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const scalar pG = gammaCell[fc[i]] * magSf[i];                        // pGamma = gammaf_b * |Sf|
    const scalar w = (type[i] == 1) ? 1.0 : ((type[i] == 5) ? vf[i] : 0.0);
    iC[i] = pG * (-w * dc[i]);
    bC[i] = -pG * (w * ref[i] * dc[i]);   // gradIC=-vf*dc, gradBC=vf*ref*dc
}


// same as bcLaplacianKernel but gamma is given per BOUNDARY FACE (e.g. nuEff = nu + nutkWallFunction at walls).
__global__
void bcLaplacianFaceKernel(
    int n,
    const label* __restrict__ type,
    const scalar* __restrict__ ref,
    const scalar* __restrict__ vf,
    const scalar* __restrict__ dc,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ gammaFace,
    scalar* __restrict__ iC,
    scalar* __restrict__ bC)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const scalar pG = gammaFace[i] * magSf[i];
    const scalar w = (type[i] == 1) ? 1.0 : ((type[i] == 5) ? vf[i] : 0.0);
    iC[i] = pG * (-w * dc[i]);
    bC[i] = -pG * (w * ref[i] * dc[i]);
}


__global__
void bcDivKernel(
    int n,
    const label* __restrict__ type,
    const scalar* __restrict__ ref,
    const scalar* __restrict__ vf,
    const scalar* __restrict__ phiB,
    scalar* __restrict__ iC,
    scalar* __restrict__ bC)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    // A COUPLED (processor) face contributes NOTHING as a boundary: its coupling is the interface off-diagonal
    // (deviceMomentumInterface), so both coeffs are zero. Without this the default branch below would give
    // vIC = 1 -> iC = phi, DOUBLE-COUNTING the interface diagonal. Mirrors host ProcessorFvPatchField, whose
    // valueInternalCoeffs/valueBoundaryCoeffs are overridden to zero for exactly this reason.
    if (type[i] == 8)
    {
        iC[i] = 0.0;
        bC[i] = 0.0;
        return;
    }
    // valueInternalCoeffs: fixedValue 0, zeroGradient/calculated 1, mixed 1-vf. valueBoundaryCoeffs: fixedValue ref,
    // mixed vf*ref, else 0.
    const scalar vIC = (type[i] == 1) ? 0.0 : ((type[i] == 5) ? (1.0 - vf[i]) : 1.0);
    const scalar vBC = (type[i] == 1) ? ref[i] : ((type[i] == 5) ? vf[i] * ref[i] : 0.0);
    iC[i] = phiB[i] * vIC;
    bC[i] = -phiB[i] * vBC;
}


__global__
void bcMatrixFluxKernel(
    int n,
    const label* __restrict__ fc,
    const scalar* __restrict__ iC,
    const scalar* __restrict__ bC,
    const scalar* __restrict__ p,
    scalar* __restrict__ fluxB)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) fluxB[i] = iC[i] * p[fc[i]] - bC[i];
}
} // namespace


void deviceBCValue(const DeviceBoundary& db, const DeviceBuffer<scalar>& internal, DeviceBuffer<scalar>& value)
{
    value.resize(db.n);
    bcValueKernel<<<nBlocks(db.n), TPB>>>(db.n, db.bcType.data(), db.refValue.data(), db.valueFraction.data(),
                                          db.faceCell.data(), internal.data(), value.data());
    cudaCheck(cudaGetLastError(), "bcValue");
}


void deviceBCLaplacianCoeffsFace(
    const DeviceBoundary& db,
    const DeviceBuffer<scalar>& gammaFace,
    DeviceBuffer<scalar>& iC,
    DeviceBuffer<scalar>& bC)
{
    iC.resize(db.n);
    bC.resize(db.n);
    bcLaplacianFaceKernel<<<nBlocks(db.n), TPB>>>(db.n, db.bcType.data(), db.refValue.data(), db.valueFraction.data(),
                                                  db.deltaCoeffs.data(), db.magSf.data(), gammaFace.data(), iC.data(), bC.data());
    cudaCheck(cudaGetLastError(), "bcLaplacianFace");
}


void deviceBCLaplacianCoeffs(
    const DeviceBoundary& db,
    const DeviceBuffer<scalar>& gammaCell,
    DeviceBuffer<scalar>& iC,
    DeviceBuffer<scalar>& bC)
{
    iC.resize(db.n);
    bC.resize(db.n);
    bcLaplacianKernel<<<nBlocks(db.n), TPB>>>(db.n, db.bcType.data(), db.refValue.data(), db.valueFraction.data(),
                                              db.deltaCoeffs.data(), db.magSf.data(), gammaCell.data(), db.faceCell.data(),
                                              iC.data(), bC.data());
    cudaCheck(cudaGetLastError(), "bcLaplacian");
}


void deviceBCDivCoeffs(
    const DeviceBoundary& db,
    const DeviceBuffer<scalar>& phiB,
    DeviceBuffer<scalar>& iC,
    DeviceBuffer<scalar>& bC)
{
    iC.resize(db.n);
    bC.resize(db.n);
    bcDivKernel<<<nBlocks(db.n), TPB>>>(db.n, db.bcType.data(), db.refValue.data(), db.valueFraction.data(),
                                        phiB.data(), iC.data(), bC.data());
    cudaCheck(cudaGetLastError(), "bcDiv");
}


void deviceMatrixFluxBoundary(
    const DeviceBoundary& db,
    const DeviceBuffer<scalar>& iC,
    const DeviceBuffer<scalar>& bC,
    const DeviceBuffer<scalar>& p,
    DeviceBuffer<scalar>& fluxB)
{
    fluxB.resize(db.n);
    bcMatrixFluxKernel<<<nBlocks(db.n), TPB>>>(db.n, db.faceCell.data(), iC.data(), bC.data(), p.data(), fluxB.data());
    cudaCheck(cudaGetLastError(), "bcMatrixFlux");
}

} // namespace brae
