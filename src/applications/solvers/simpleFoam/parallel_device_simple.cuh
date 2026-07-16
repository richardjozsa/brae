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
        int maxIter)
        : P_(part),
          nu_(nu),
          relaxU_(relaxU),
          relaxP_(relaxP),
          tolU_(tolU),
          tolP_(tolP),
          maxIter_(maxIter),
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

    // One distributed SIMPLE iteration. U/p/phi are updated in place on the device.
    void step();

    std::vector<vector> U() const
    {
        const std::vector<scalar> ux = Uk_[0].host(), uy = Uk_[1].host(), uz = Uk_[2].host();
        std::vector<vector> out(lnC_);
        for (label c = 0; c < lnC_; ++c) out[c] = vector{ux[c], uy[c], uz[c]};
        return out;
    }
    std::vector<scalar> p() const { return dp_.host(); }

    // The global field, gathered from every partition (decomposePar's inverse) -- for output/validation.
    std::vector<scalar> reconstructP() const { return reconstructField(P_.Lm.cellProcAddr, dp_.host(), P_.globalNCells); }

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
    label  lnC_, nIf_, nBnd_ = 0;
    DeviceMesh           dm_;
    DeviceVectorBoundary dbU_;
    DeviceBoundary       dbP_, dbRAU_;
    GeometricField<scalar> rAUfld_;
    DeviceHalo           halo_;
    std::vector<label>   procStart_;
    std::vector<DeviceBuffer<label>>  faceCellsD_;
    std::vector<DeviceBuffer<scalar>> weightsD_;
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
inline void ParallelDeviceSimple::step()
{
    const std::vector<FvPatch>& lp = P_.lp;
    const FvGeometry& lg = P_.lg;

    // ---- momentum matrix: div(phi,U) - laplacian(nuEff,U), + processor interface ----
    DeviceBuffer<scalar> nuEff_f(std::vector<scalar>(nIf_, nu_));
    DeviceBuffer<scalar> mDiag, mUp, mLo, lD, lU, lL;
    deviceDivUpwindCoeffs(dm_, phiInt_, mDiag, mUp, mLo);
    deviceLaplacianCoeffs(dm_, nuEff_f, lD, lU, lL, false);
    deviceAxpy(-1.0, lD, mDiag);
    deviceAxpy(-1.0, lU, mUp);
    deviceAxpy(-1.0, lL, mLo);

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
                    cg[i] = nu_ * lg.magSf()[lp[pi].start + i] * P_.procDelta[pj][i];
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
    for (int k = 0; k < 3; ++k)
    {
        deviceHadamard(relaxSrc[k], delta, Uk_[k]);      // delta*U_old
        deviceAxpy(1.0, *sS[k], relaxSrc[k]);            // + V*divSig  -> == host Ml.source
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
        deviceParallelJacobiBiCGStab(mv, halo_, ifCoeff, b, Uk_[k], nf, tolU_, 0.0, maxIter_);
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
    deviceLaplacianCoeffs(dm_, rAUf_int, pD, pU_, pL_, false);
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
                    cg[i] = rAUbndH[bi + i] * lg.magSf()[lp[pi].start + i] * P_.procDelta[pj][i];
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

    DeviceBuffer<scalar> pSol;
    deviceCopy(pSol, dp_);
    const DeviceLduView pv = deviceLduView(dm_, pDiagC, pU_, pL_);
    const scalar nfp = deviceParallelNormFactor(pv, halo_, pIfCoeff, pSol, pb, ones_, P_.globalNCells);
    deviceParallelJacobiPCG(pv, halo_, pIfCoeff, pb, pSol, nfp, tolP_, 0.0, maxIter_);

    // ---- corrector: conservative phi, relax p, U = HbyA - rAU*grad(p) ----
    DeviceBuffer<scalar> pfluxInt, pfluxBnd;
    deviceMatrixFluxInternal(pv, pSol, pfluxInt);
    deviceMatrixFluxBoundary(dbP_, piC, pbC, pSol, pfluxBnd);
    deviceParallelMatrixFluxInterface(halo_, faceCellsD_, pIfCoeff, procStart_, pSol, pfluxBnd);
    deviceAxpy(-1.0, pfluxInt, phiHbyAint);          // phi = phiHbyA - pflux
    deviceAxpy(-1.0, pfluxBnd, phiHbyAbnd);
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
    }
}

} // namespace brae
