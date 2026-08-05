// cf GPU offload -- per-iteration flow BC value updates (updateCoeffs): the BC state that changes with the flux/
// velocity each SIMPLE iteration -- inletOutlet/outletInlet, mixed freestream, pressureInletOutletVelocity,
// slip/symmetry, totalPressure, and the HbyA constraints at slip/mixed faces. Split from device_boundary.cu
// (the BC matrix/flux contributions are in device_boundary_assembly.cu). Shared decls: device_boundary.cuh.
#include "device_boundary.cuh"
#include <cuda_runtime.h>

namespace brae {

namespace {
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }


// inletOutlet/outletInlet updateCoeffs: valueFraction is BINARY from the flux sign. inletOutlet -> inflow
// (phi<0) = fixedValue; outletInlet (freestreamPressure) -> the OPPOSITE: outflow (phi>=0) = fixedValue.
__global__
void ioUpdateKernel(
    int n,
    const label* __restrict__ ioMask,
    const label* __restrict__ oioMask,
    const scalar* __restrict__ phiB,
    label* __restrict__ bcType)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (ioMask[i])       bcType[i] = (phiB[i] <  0.0) ? 1 : 0;            // inflow = fixedValue(inletValue)
    else if (oioMask[i]) bcType[i] = (phiB[i] >= 0.0) ? 1 : 0;            // outflow = fixedValue(outletValue)
}


// mixed freestream updateCoeffs: vf = 0.5 -/+ 0.5*(U.n)/|U|. The normal flux U.n = phi_b/|Sf| is exact; |U| is the
// LOCAL adjacent-cell speed (OF uses the patch |U|; the cell value ~= it and avoids the vf circularity). Using the
// freestream |Uinf| instead is fine at a true far field but mis-scales an outlet whose speed != |Uinf|. Continuous
// in the flow angle (not a binary switch); at grazing faces (phi~0) vf->0.5 (the Robin midpoint OF uses).
__global__
void mixedUpdateKernel(
    int n,
    const label* __restrict__ maskU,
    const label* __restrict__ maskP,
    const label* __restrict__ fc,
    const scalar* __restrict__ phiB,
    const scalar* __restrict__ magSf,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar* __restrict__ vfU0,
    scalar* __restrict__ vfU1,
    scalar* __restrict__ vfU2,
    scalar* __restrict__ vfP)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const bool mu = maskU[i], mp = maskP[i];
    if (!mu && !mp) return;
    const int c = fc[i];
    const scalar mU = sqrt(Ux[c] * Ux[c] + Uy[c] * Uy[c] + Uz[c] * Uz[c]);   // local |U| (adjacent cell)
    scalar ct = (phiB[i] / magSf[i]) / fmax(mU, 1e-30);                      // (U.n)/|U|
    ct = fmin(fmax(ct, -1.0), 1.0);
    if (mu)   // velocity sign
    {
        const scalar vfu = 0.5 - 0.5 * ct;
        vfU0[i] = vfu;
        vfU1[i] = vfu;
        vfU2[i] = vfu;
    }
    if (mp) vfP[i] = 0.5 + 0.5 * ct;                                                              // pressure sign
}


// pressureInletOutletVelocity updateCoeffs (directionMixed): per piov face, outflow -> zeroGradient (bcType 0),
// inflow -> fixedValue (bcType 1) with refValue = n*(n.U_cell) (the normal projection; tangential refValue 0).
__global__
void piovUpdateKernel(
    int n,
    const label* __restrict__ piov,
    const label* __restrict__ fc,
    const scalar* __restrict__ phiB,
    const scalar* __restrict__ nx,
    const scalar* __restrict__ ny,
    const scalar* __restrict__ nz,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    label* __restrict__ ty0,
    label* __restrict__ ty1,
    label* __restrict__ ty2,
    scalar* __restrict__ r0,
    scalar* __restrict__ r1,
    scalar* __restrict__ r2)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !piov[i]) return;

    if (phiB[i] >= 0.0)   // outflow -> zeroGradient
    {
        ty0[i] = ty1[i] = ty2[i] = 0;
    }
    else   // inflow -> fixedValue normal projection
    {
        const int c = fc[i];
        const scalar Un = nx[i] * Ux[c] + ny[i] * Uy[c] + nz[i] * Uz[c];         // n . U_cell
        ty0[i] = ty1[i] = ty2[i] = 1;
        r0[i] = nx[i] * Un;
        r1[i] = ny[i] * Un;
        r2[i] = nz[i] * Un;
    }
}


// slip/symmetry updateCoeffs (OF basicSymmetry, general normal): per symMask face, per component k set the mixed
// valueFraction vf_k = |n_k| and ref_k = U_c[k] - sign(n_k)*(n.U_c). The cat-5 kernels then give valueIC_k = 1-|n_k|,
// gradIC_k = -dc*|n_k|, value = U_c - n(n.U_c), OF's symmetry coeffs. sign(0)=0 -> tangential (n_k=0) is zeroGradient.
__global__
void symUpdateKernel(
    int n,
    const label* __restrict__ sym,
    const label* __restrict__ fc,
    const scalar* __restrict__ nx,
    const scalar* __restrict__ ny,
    const scalar* __restrict__ nz,
    const scalar* __restrict__ Ux,
    const scalar* __restrict__ Uy,
    const scalar* __restrict__ Uz,
    scalar* __restrict__ vf0,
    scalar* __restrict__ vf1,
    scalar* __restrict__ vf2,
    scalar* __restrict__ r0,
    scalar* __restrict__ r1,
    scalar* __restrict__ r2)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !sym[i]) return;

    const int c = fc[i];
    const scalar nU = nx[i] * Ux[c] + ny[i] * Uy[c] + nz[i] * Uz[c];               // n . U_cell
    const scalar sgx = (nx[i] > 0) - (nx[i] < 0), sgy = (ny[i] > 0) - (ny[i] < 0), sgz = (nz[i] > 0) - (nz[i] < 0);
    vf0[i] = fabs(nx[i]);
    vf1[i] = fabs(ny[i]);
    vf2[i] = fabs(nz[i]);
    r0[i] = Ux[c] - sgx * nU;
    r1[i] = Uy[c] - sgy * nU;
    r2[i] = Uz[c] - sgz * nU;
}


// constrainHbyA at slip/symmetry faces: the wall flux must be 0 (no penetration), so HbyA_b = HbyA_c - n(n.HbyA_c)
// (tangential projection) -> phiHbyA_b = HbyA_b.Sf = |Sf|(HbyA_b.n) = 0. The cat-5 deviceBCValue blends `ref` (built
// from U, not HbyA), which is NOT the HbyA projection on an angled wall, so override it here. (Axis-aligned already
// gets 0 from ref_normal=0, so this is a no-op there.)
__global__
void symHbyAKernel(
    int n,
    const label* __restrict__ sym,
    const label* __restrict__ fc,
    const scalar* __restrict__ nx,
    const scalar* __restrict__ ny,
    const scalar* __restrict__ nz,
    const scalar* __restrict__ Hx,
    const scalar* __restrict__ Hy,
    const scalar* __restrict__ Hz,
    scalar* __restrict__ hxb,
    scalar* __restrict__ hyb,
    scalar* __restrict__ hzb)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !sym[i]) return;

    const int c = fc[i];
    const scalar nH = nx[i] * Hx[c] + ny[i] * Hy[c] + nz[i] * Hz[c];
    hxb[i] = Hx[c] - nx[i] * nH;
    hyb[i] = Hy[c] - ny[i] * nH;
    hzb[i] = Hz[c] - nz[i] * nH;
}


// at mixed faces, overwrite the HbyA boundary value with the U boundary value (constrainHbyA at fixesValue patches).
__global__
void selectMixedKernel(
    int n,
    const label* __restrict__ mask,
    const scalar* __restrict__ ux,
    const scalar* __restrict__ uy,
    const scalar* __restrict__ uz,
    scalar* __restrict__ hx,
    scalar* __restrict__ hy,
    scalar* __restrict__ hz)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && mask[i]) { hx[i] = ux[i]; hy[i] = uy[i]; hz[i] = uz[i]; }
}


// totalPressure (OF totalPressureFvPatchScalarField::updateCoeffs). OF branches on the DIMENSIONS of p:
//
//   kinematic p (incompressible):  p = p0 - 0.5*neg(phi)*magSqr(U)
//   absolute p, psi "none":        p = p0 - 0.5*RHO*neg(phi)*magSqr(U)      <- rhoBnd supplies the rho
//
// The rho factor is not optional on a compressible case: it is the difference between a dynamic head in
// m2/s2 and one in Pa. Passing rhoBnd = null gives the incompressible form, bit-identical to before.
// (OF's third branch, the high-speed isentropic form with a named psi and gamma, is NOT implemented --
// readThermoCoeffs-style refusal happens at load rather than silently running the low-speed form.)
__global__
void tpUpdateKernel(
    int n,
    const label* __restrict__ tpMask,
    const scalar* __restrict__ p0,
    const scalar* __restrict__ phiB,
    const scalar* __restrict__ Uxb,
    const scalar* __restrict__ Uyb,
    const scalar* __restrict__ Uzb,
    const scalar* __restrict__ rhoBnd,   // compressible: rho at the face; null -> kinematic (incompressible)
    scalar* __restrict__ refValue)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || !tpMask[i]) return;

    const scalar u2 = Uxb[i]*Uxb[i] + Uyb[i]*Uyb[i] + Uzb[i]*Uzb[i];
    const scalar rw = rhoBnd ? rhoBnd[i] : scalar(1);
    refValue[i] = p0[i] - 0.5 * rw * (phiB[i] < 0.0 ? 1.0 : 0.0) * u2;   // neg(phi)=inflow -> static = total - dynamic head
}
} // namespace


// flowRateInletVelocity, massFlowRate form (OF flowRateInletVelocityFvPatchVectorField::updateValues,
// extrapolateProfile false):  avgU = -flowRate/gSum(rho*magSf);  U_b = avgU*n.
//
// avgU is a single patch-wide scalar, computed by the caller as -mdot/dot(rhoBnd, maskedMagSf) -- the
// mask makes that dot product exactly OF's gSum over this patch. Faces outside the patch have mask 0
// and are left untouched.
__global__
void frUpdateKernel(
    int n,
    const scalar* __restrict__ mask,
    scalar avgU,
    const scalar* __restrict__ nx,
    const scalar* __restrict__ ny,
    const scalar* __restrict__ nz,
    scalar* __restrict__ refX,
    scalar* __restrict__ refY,
    scalar* __restrict__ refZ)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n || mask[i] <= scalar(0)) return;
    refX[i] = avgU * nx[i];
    refY[i] = avgU * ny[i];
    refZ[i] = avgU * nz[i];
}

void deviceUpdateFlowRateInlet(
    DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& maskMagSf,
    scalar avgU,
    const DeviceBuffer<scalar>& nx,
    const DeviceBuffer<scalar>& ny,
    const DeviceBuffer<scalar>& nz)
{
    const int n = dbU.comp[0].n;
    if (n == 0) return;
    frUpdateKernel<<<nBlocks(n), TPB>>>(n, maskMagSf.data(), avgU, nx.data(), ny.data(), nz.data(),
                                        dbU.comp[0].refValue.data(),
                                        dbU.comp[1].refValue.data(),
                                        dbU.comp[2].refValue.data());
    cudaCheck(cudaGetLastError(), "frUpdate");
}

void deviceUpdateInletOutlet(DeviceBoundary& db, const DeviceBuffer<scalar>& phiBnd)
{
    if (db.n == 0) return;
    ioUpdateKernel<<<nBlocks(db.n), TPB>>>(db.n, db.ioMask.data(), db.oioMask.data(), phiBnd.data(), db.bcType.data());
    cudaCheck(cudaGetLastError(), "ioUpdate");
}


void deviceUpdateMixedFreestream(
    DeviceVectorBoundary& dbU,
    DeviceBoundary& dbP,
    const DeviceBuffer<scalar>& phiBnd,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz)
{
    const int n = dbP.n;
    if (n == 0) return;
    mixedUpdateKernel<<<nBlocks(n), TPB>>>(n, dbU.comp[0].mixedMask.data(), dbP.mixedMask.data(), dbP.faceCell.data(),
                                           phiBnd.data(), dbP.magSf.data(), Ux.data(), Uy.data(), Uz.data(),
                                           dbU.comp[0].valueFraction.data(), dbU.comp[1].valueFraction.data(),
                                           dbU.comp[2].valueFraction.data(), dbP.valueFraction.data());
    cudaCheck(cudaGetLastError(), "mixedUpdate");
}


void deviceUpdatePressureInletOutletVelocity(
    DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& phiBnd,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz)
{
    const int n = dbU.n;
    if (n == 0) return;
    piovUpdateKernel<<<nBlocks(n), TPB>>>(n, dbU.comp[0].piovMask.data(), dbU.comp[0].faceCell.data(), phiBnd.data(),
                                          dbU.nx.data(), dbU.ny.data(), dbU.nz.data(), Ux.data(), Uy.data(), Uz.data(),
                                          dbU.comp[0].bcType.data(), dbU.comp[1].bcType.data(), dbU.comp[2].bcType.data(),
                                          dbU.comp[0].refValue.data(), dbU.comp[1].refValue.data(), dbU.comp[2].refValue.data());
    cudaCheck(cudaGetLastError(), "piovUpdate");
}


void deviceUpdateSymmetry(
    DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz)
{
    const int n = dbU.n;
    if (n == 0) return;
    symUpdateKernel<<<nBlocks(n), TPB>>>(n, dbU.comp[0].symMask.data(), dbU.comp[0].faceCell.data(),
                                         dbU.nx.data(), dbU.ny.data(), dbU.nz.data(), Ux.data(), Uy.data(), Uz.data(),
                                         dbU.comp[0].valueFraction.data(), dbU.comp[1].valueFraction.data(), dbU.comp[2].valueFraction.data(),
                                         dbU.comp[0].refValue.data(), dbU.comp[1].refValue.data(), dbU.comp[2].refValue.data());
    cudaCheck(cudaGetLastError(), "symUpdate");
}


void deviceConstrainSymmetryHbyA(
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Hx,
    const DeviceBuffer<scalar>& Hy,
    const DeviceBuffer<scalar>& Hz,
    DeviceBuffer<scalar>& hbx,
    DeviceBuffer<scalar>& hby,
    DeviceBuffer<scalar>& hbz)
{
    const int n = dbU.n;
    if (n == 0) return;
    symHbyAKernel<<<nBlocks(n), TPB>>>(n, dbU.comp[0].symMask.data(), dbU.comp[0].faceCell.data(),
                                       dbU.nx.data(), dbU.ny.data(), dbU.nz.data(), Hx.data(), Hy.data(), Hz.data(),
                                       hbx.data(), hby.data(), hbz.data());
    cudaCheck(cudaGetLastError(), "constrainSymmetryHbyA");
}


void deviceConstrainMixedHbyA(
    const DeviceVectorBoundary& dbU,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& hbx,
    DeviceBuffer<scalar>& hby,
    DeviceBuffer<scalar>& hbz)
{
    const int n = dbU.n;
    if (n == 0) return;
    DeviceBuffer<scalar> ubx, uby, ubz;   // U_b = mixed boundary value of U (uses dbU vf); deviceBCValue in device_boundary_assembly.cu
    deviceBCValue(dbU.comp[0], Ux, ubx);
    deviceBCValue(dbU.comp[1], Uy, uby);
    deviceBCValue(dbU.comp[2], Uz, ubz);
    selectMixedKernel<<<nBlocks(n), TPB>>>(n, dbU.comp[0].mixedMask.data(), ubx.data(), uby.data(), ubz.data(),
                                           hbx.data(), hby.data(), hbz.data());
    cudaCheck(cudaGetLastError(), "constrainMixedHbyA");
}


void deviceUpdateTotalPressure(
    DeviceBoundary& db,
    const DeviceBuffer<scalar>& phiB,
    const DeviceBuffer<scalar>& Uxb,
    const DeviceBuffer<scalar>& Uyb,
    const DeviceBuffer<scalar>& Uzb,
    const DeviceBuffer<scalar>* rhoBnd)
{
    if (db.n == 0) return;
    tpUpdateKernel<<<nBlocks(db.n), TPB>>>(db.n, db.tpMask.data(), db.p0.data(), phiB.data(), Uxb.data(),
                                           Uyb.data(), Uzb.data(),
                                           (rhoBnd && rhoBnd->size() == static_cast<std::size_t>(db.n)) ? rhoBnd->data() : nullptr,
                                           db.refValue.data());
    cudaCheck(cudaGetLastError(), "tpUpdate");
}

} // namespace brae
