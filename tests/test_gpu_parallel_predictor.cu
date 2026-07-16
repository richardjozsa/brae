// Phase 4a L8: the DEVICE distributed momentum PREDICTOR reproduces the host predictor.
//
// This is the first composed stage of the distributed SIMPLE step -- every ingredient below was validated
// individually (L1 assembly, relax, grad(p) coupling, L2 BiCGStab, the coupled bcType); this checks they
// compose into the same U the validated host parallelSimpleStepLaminar predictor produces:
//
//   M    = div(phi,U) - laplacian(nuEff,U)          (+ processor interface coeffs)
//   relax(M, relaxU)                                 -> relaxed diag + source += delta*U
//   per component k:  (diag + iC_k) x = delta*U_k - V*grad(p)_k + bC_k
//
// The host solves with PBiCGStab/DILU and the device with Jacobi-BiCGStab, so the converged U agrees to the
// solver tolerance, not bit-exactly.
//
// Run: mpirun -np {1,2,4,8} test_gpu_parallel_predictor <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "scotch_decomposition.cuh"
#include "parallel_simple.cuh"        // Partition, outwardFlux, pcmp/pset
#include "local_assembly.cuh"         // momentumDistributed
#include "parallel_matrix_ops.cuh"    // parallelRelaxMatrix
#include "parallel_pbicgstab.cuh"
#include "parallel_pcg.cuh"
#include "field_distribute.cuh"
#include "device_mesh.cuh"
#include "device_ldu.cuh"
#include "device_blas.cuh"
#include "device_pcg.cuh"
#include "device_simple.cuh"
#include "device_boundary.cuh"
#include "device_halo.cuh"
#include "device_buffer.cuh"
#include "parallel_device_simple.cuh"
#include "cf_pstream.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv)
{
    Pstream::init(argc, argv);
    const int rank = Pstream::myProcNo(), nproc = Pstream::nProcs();
    if (argc < 2)
    {
        if (Pstream::master()) std::printf("usage: %s <caseDir>\n", argv[0]);
        Pstream::finalize();
        return 2;
    }
    if (!Pstream::nvshmemActive())
    {
        if (Pstream::master()) std::printf("test_gpu_parallel_predictor: NVSHMEM inactive -- SKIP\n");
        Pstream::finalize();
        return 0;
    }
    const std::string caseDir = argv[1];
    const scalar nu = 1e-3, relaxU = 0.7, tolU = 1e-9, tolP = 1e-9;
    const int maxIter = 3000;

    PrimitiveMesh gm;  gm.read(caseDir + "/constant/polyMesh");
    const label nC = gm.nCells();
    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    const Partition P(gm, cellToPart, rank);
    const PrimitiveMesh& m = P.Lm.mesh;
    const FvGeometry& lg = P.lg;
    const std::vector<FvPatch>& lp = P.lp;
    const label lnC = m.nCells(), nIf = m.nInternalFaces();

    // Distributed fields.
    GeometricField<vector> U = distributeField<vector>(readField<vector>(caseDir + "/282/U"),
                                                       gm.patches(), P.Lm, lp, P.procW, rank);
    U.evaluateBoundary();
    GeometricField<scalar> p = distributeField<scalar>(readField<scalar>(caseDir + "/282/p"),
                                                       gm.patches(), P.Lm, lp, P.procW, rank);
    p.evaluateBoundary();
    const SurfaceScalarField phi = fvc::flux(U, m, lg, lp);

    const std::vector<scalar> nuEffCell(lnC, nu);
    SurfaceScalarField nuEfff0 = fvc::interpolate(nuEffCell, m, lg, lp);
    std::vector<std::vector<scalar>> nuEffBndPatch(lp.size());
    for (std::size_t pi = 0; pi < lp.size(); ++pi) nuEffBndPatch[pi].assign(lp[pi].size, nu);
    nuEfff0.boundary = nuEffBndPatch;
    const std::vector<scalar> nuF(m.nFaces(), nu);

    // ---- HOST predictor (the oracle) ----
    const std::vector<scalar> outPhi = outwardFlux(P, phi);
    FvVectorMatrix Ml = fvm::div(phi.internal, phi.boundary, U, m, lp);
    addEqual(Ml, fvm::laplacian(nuEfff0, U, m, lg, lp), -1.0);
    DistributedMatrix L = momentumDistributed(Ml, P.Lm, lg, lp, outPhi, nuF, P.procDelta);
    parallelRelaxMatrix(L, Ml, U.internal, P.Lm, lp, relaxU);
    std::vector<vector> Uhost = U.internal;
    std::vector<scalar> rAUhost(lnC);
    std::vector<vector> HbyAhost(lnC);
    SurfaceScalarField phiHbyAhost;
    std::vector<scalar> phost, rAUfHost;
    {
        std::vector<ProcessorInterface> ifs = P.ifs;
        const std::vector<vector> gP = fvc::gaussGrad(p, m, lg, lp);
        for (int k = 0; k < 3; ++k)
        {
            DistributedMatrix Lc = L;
            Lc.diagC = L.diag;
            Lc.b.assign(lnC, 0.0);
            for (label c = 0; c < lnC; ++c)
                Lc.b[c] = pcmp(Ml.source[c], k) - lg.V()[c] * pcmp(gP[c], k);
            for (std::size_t pi = 0; pi < lp.size(); ++pi)
                for (label i = 0; i < lp[pi].size; ++i)
                {
                    Lc.diagC[lp[pi].faceCells[i]] += pcmp(Ml.internalCoeffs[pi][i], k);
                    Lc.b[lp[pi].faceCells[i]]     += pcmp(Ml.boundaryCoeffs[pi][i], k);
                }
            std::vector<scalar> x(lnC);
            for (label c = 0; c < lnC; ++c) x[c] = pcmp(U.internal[c], k);
            parallelPBiCGStab(Lc, x, ifs, P.globalNCells, tolU, 0.0, maxIter);
            for (label c = 0; c < lnC; ++c) pset(Uhost[c], k, x[c]);
        }

        // rAU = 1/A, A = (diag + cmptAv(iC))/V
        DistributedMatrix La = L;
        La.diagC = L.diag;
        for (std::size_t pi = 0; pi < lp.size(); ++pi)
            for (label i = 0; i < lp[pi].size; ++i)
                La.diagC[lp[pi].faceCells[i]] += cmptAv(Ml.internalCoeffs[pi][i]);
        const std::vector<scalar> A = parallelMatrixA(La, lg.V());
        for (label c = 0; c < lnC; ++c) rAUhost[c] = 1.0 / A[c];

        // HbyA = rAU * H(U_post) / V, with H's source = Ml.source (= delta*U_old, set at relax time)
        for (int k = 0; k < 3; ++k)
        {
            std::vector<scalar> Uk(lnC);
            for (label c = 0; c < lnC; ++c) Uk[c] = pcmp(Uhost[c], k);
            const std::vector<scalar> Apsi = parallelAmul(La, Uk, ifs);
            std::vector<scalar> Hk(lnC);
            for (label c = 0; c < lnC; ++c)
                Hk[c] = La.diag[c] * Uk[c] - Apsi[c] + pcmp(Ml.source[c], k);
            for (std::size_t pi = 0; pi < lp.size(); ++pi)
                for (label i = 0; i < lp[pi].size; ++i)
                {
                    const vector ic = Ml.internalCoeffs[pi][i];
                    const label fc = lp[pi].faceCells[i];
                    Hk[fc] += (cmptAv(ic) - pcmp(ic, k)) * Uk[fc] + pcmp(Ml.boundaryCoeffs[pi][i], k);
                }
            for (label c = 0; c < lnC; ++c) pset(HbyAhost[c], k, rAUhost[c] * Hk[c] / lg.V()[c]);
        }
        GeometricField<vector> HbyAf = distributeField<vector>(readField<vector>(caseDir + "/282/U"),
                                                               gm.patches(), P.Lm, lp, P.procW, rank);
        HbyAf.internal = HbyAhost;
        HbyAf.evaluateBoundary();
        phiHbyAhost = fvc::flux(HbyAf, m, lg, lp);

        // ---- pEqn: laplacian(rAU,p) == div(phiHbyA) ----
        GeometricField<scalar> rAUfld = distributeFromCells<scalar>(rAUhost, P);
        SurfaceScalarField rAUf = fvc::interpolate(rAUhost, m, lg, lp);
        std::vector<scalar> rAUFf(m.nFaces(), 0.0);
        for (label f = 0; f < nIf; ++f) rAUFf[f] = rAUf.internal[f];
        for (std::size_t pi = 0; pi < lp.size(); ++pi)
            if (lp[pi].type == "processor")
                for (label i = 0; i < lp[pi].size; ++i)
                    rAUFf[lp[pi].start + i] = rAUfld.boundary[pi]->value()[i];
        rAUfHost = rAUFf;

        FvScalarMatrix Mpl = fvm::laplacian(rAUf, p, m, lg, lp);
        const std::vector<scalar> dphi = fvc::div(phiHbyAhost, m, lg, lp);
        for (label c = 0; c < lnC; ++c) Mpl.source[c] += lg.V()[c] * dphi[c];
        DistributedMatrix Lp = assembleLocalLaplacianF(P.Lm, lg, lp, P.procDelta, rAUFf);
        Lp.diag  = Mpl.diag;
        Lp.upper = Mpl.upper;
        Lp.lower = Mpl.lower;
        {
            std::size_t pj2 = 0;
            for (std::size_t pi = 0; pi < lp.size(); ++pi)
            {
                if (lp[pi].type != "processor") continue;
                for (label i = 0; i < lp[pi].size; ++i)
                    Lp.diag[lp[pi].faceCells[i]] -= rAUFf[lp[pi].start + i] * lg.magSf()[lp[pi].start + i] * P.procDelta[pj2][i];
                ++pj2;
            }
        }
        Lp.diagC = Lp.diag;
        Lp.b.resize(lnC);
        for (label c = 0; c < lnC; ++c) Lp.b[c] = Mpl.source[c];
        for (std::size_t pi = 0; pi < lp.size(); ++pi)
            for (label i = 0; i < lp[pi].size; ++i)
            {
                Lp.diagC[lp[pi].faceCells[i]] += Mpl.internalCoeffs[pi][i];
                Lp.b[lp[pi].faceCells[i]]     += Mpl.boundaryCoeffs[pi][i];
            }
        phost = p.internal;
        parallelPCG(Lp, phost, ifs, P.globalNCells, tolP, 0.0, maxIter);
    }

    // ---- DEVICE predictor ----
    scalar gMax = 0, gMag = 0;
    scalar relRAU = 0, relHbyA = 0, relPhi = 0, relP = 0;
    {
        const DeviceMesh dm = buildDeviceMesh(m, lg, lp);
        const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, lp, lg);
        const DeviceBoundary       dbP = buildDeviceBoundary(p, lp, lg);

        std::vector<scalar> uxH(lnC), uyH(lnC), uzH(lnC);
        for (label c = 0; c < lnC; ++c)
        {
            uxH[c] = U.internal[c].x;
            uyH[c] = U.internal[c].y;
            uzH[c] = U.internal[c].z;
        }
        DeviceBuffer<scalar> Uk[3];
        Uk[0].copyFrom(uxH);
        Uk[1].copyFrom(uyH);
        Uk[2].copyFrom(uzH);
        DeviceBuffer<scalar> dp(p.internal), ones_d(std::vector<scalar>(lnC, 1.0));
        DeviceBuffer<scalar> phiInt_d(std::vector<scalar>(phi.internal.begin(), phi.internal.begin() + nIf));
        DeviceBuffer<scalar> nuEff_fd(std::vector<scalar>(nuEfff0.internal.begin(), nuEfff0.internal.begin() + nIf));

        // flattened boundary arrays (device gather order), + processor patch offsets for the halo injection
        std::vector<scalar> phiBndH, nuEffBndH;
        std::vector<label>  procStart;
        label bidx = 0;
        for (std::size_t pi = 0; pi < lp.size(); ++pi)
        {
            if (lp[pi].type == "cyclic" || lp[pi].type == "cyclicAMI") continue;
            if (lp[pi].type == "processor") procStart.push_back(bidx);
            for (label i = 0; i < lp[pi].size; ++i)
            {
                phiBndH.push_back(phi.boundary[pi][i]);
                nuEffBndH.push_back(nu);
            }
            bidx += lp[pi].size;
        }
        DeviceBuffer<scalar> phiBnd_d(phiBndH), nuEffBnd_d(nuEffBndH);

        // momentum assembly + processor interface
        DeviceBuffer<scalar> mDiag, mUp, mLo, lD, lU, lL;
        deviceDivUpwindCoeffs(dm, phiInt_d, mDiag, mUp, mLo);
        deviceLaplacianCoeffs(dm, nuEff_fd, lD, lU, lL, false);
        deviceAxpy(-1.0, lD, mDiag);
        deviceAxpy(-1.0, lU, mUp);
        deviceAxpy(-1.0, lL, mLo);

        std::vector<int> nbrParts;
        for (int q : P.Lm.procNbr) nbrParts.push_back(q);
        DeviceHalo halo(rank, nbrParts, P.Lm.procFaceCells);
        std::vector<DeviceBuffer<label>>  faceCellsD(P.Lm.procFaceCells.size());
        std::vector<DeviceBuffer<scalar>> phiF, coeffGeo, weightsD(P.procW.size());
        for (std::size_t i = 0; i < P.Lm.procFaceCells.size(); ++i) faceCellsD[i].copyFrom(P.Lm.procFaceCells[i]);
        for (std::size_t i = 0; i < P.procW.size(); ++i) weightsD[i].copyFrom(P.procW[i]);
        std::size_t pj = 0;
        for (std::size_t pi = 0; pi < lp.size(); ++pi)
        {
            if (lp[pi].type != "processor") continue;
            DeviceBuffer<scalar> pf;
            pf.copyFrom(std::vector<scalar>(phi.boundary[pi].begin(), phi.boundary[pi].end()));
            phiF.push_back(std::move(pf));
            std::vector<scalar> cg(lp[pi].size);
            for (label f = 0; f < lp[pi].size; ++f)
                cg[f] = nu * lg.magSf()[lp[pi].start + f] * P.procDelta[pj][f];
            DeviceBuffer<scalar> cgd;
            cgd.copyFrom(cg);
            coeffGeo.push_back(std::move(cgd));
            ++pj;
        }
        std::vector<DeviceBuffer<scalar>> ifCoeff;
        deviceMomentumInterface(halo, faceCellsD, phiF, coeffGeo, mDiag, ifCoeff);

        // relax (interface counts toward diagonal dominance)
        DeviceBuffer<scalar> r0IC, r0BC, r0lIC, r0lBC;
        deviceBCDivCoeffs(dbU.comp[0], phiBnd_d, r0IC, r0BC);
        deviceBCLaplacianCoeffsFace(dbU.comp[0], nuEffBnd_d, r0lIC, r0lBC);
        deviceAxpy(-1.0, r0lIC, r0IC);
        DeviceBuffer<scalar> sumOff(std::vector<scalar>(lnC, 0.0));
        deviceInterfaceOffDiagSum(halo, faceCellsD, ifCoeff, sumOff);
        DeviceBuffer<scalar> mDiagR, delta;
        deviceRelaxDiag(deviceLduView(dm, mDiag, mUp, mLo), dm, r0IC, relaxU, mDiagR, delta, sumOff.data());

        // grad(p): coupled processor boundary value injected by the halo
        DeviceBuffer<scalar> pbv;
        deviceBCValue(dbP, dp, pbv);                                     // real BCs (bcType 8 faces untouched)
        halo.exchange(dp.data());
        halo.scatterBoundaryValues(dp.data(), weightsD, procStart, pbv.data());
        halo.waitExchange();                                             // protect recv buf before next exchange
        DeviceBuffer<scalar> gx, gy, gz;
        deviceGaussGrad(dm, dp, pbv, gx, gy, gz);
        DeviceBuffer<scalar>* gg[3] = { &gx, &gy, &gz };

        // Per-component boundary coeffs + the relax source. relaxSrc = delta*U_OLD must be snapshotted for ALL
        // components BEFORE any solve: the host sets Ml.source once at relax time and H() later reads that same
        // (old-U) source while using the POST-predictor U.
        DeviceBuffer<scalar> iC[3], bC[3], relaxSrc[3];
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> lIC, lBC;
            deviceBCDivCoeffs(dbU.comp[k], phiBnd_d, iC[k], bC[k]);
            deviceBCLaplacianCoeffsFace(dbU.comp[k], nuEffBnd_d, lIC, lBC);
            deviceAxpy(-1.0, lIC, iC[k]);
            deviceAxpy(-1.0, lBC, bC[k]);
            deviceHadamard(relaxSrc[k], delta, Uk[k]);                   // delta * U_old
        }

        // per-component predictor solve
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> s;
            deviceHadamard(s, dm.V, *gg[k]);
            deviceScale(s, -1.0);
            deviceAxpy(1.0, relaxSrc[k], s);                             // s = -V*grad(p)_k + delta*U_k

            DeviceBuffer<scalar> diagC, b;
            deviceFold(dm, mDiagR, s, iC[k], bC[k], diagC, b);
            const DeviceLduView mv = deviceLduView(dm, diagC, mUp, mLo);
            const scalar nf = deviceParallelNormFactor(mv, halo, ifCoeff, Uk[k], b, ones_d, P.globalNCells);
            deviceParallelJacobiBiCGStab(mv, halo, ifCoeff, b, Uk[k], nf, tolU, 0.0, maxIter);
        }

        // ---- rAU = V/diagA,  diagA = relaxedDiag + cmptAv(iC) ----
        DeviceBuffer<scalar> cmptAvIC;
        deviceCopy(cmptAvIC, iC[0]);
        deviceAxpy(1.0, iC[1], cmptAvIC);
        deviceAxpy(1.0, iC[2], cmptAvIC);
        deviceScale(cmptAvIC, 1.0 / 3.0);
        DeviceBuffer<scalar> zeroSrc(std::vector<scalar>(lnC, 0.0)), zeroBnd(std::vector<scalar>(dbU.n, 0.0));
        DeviceBuffer<scalar> diagA, dumb, rAU;
        deviceFold(dm, mDiagR, zeroSrc, cmptAvIC, zeroBnd, diagA, dumb);
        deviceReciprocalV(dm, diagA, rAU);

        // ---- HbyA_k = rAU * H_k   (H's bdDiag = cmptAv(iC) - iC_k, bdSrc = bC_k) ----
        DeviceBuffer<scalar> HbyA[3];
        const DeviceLduView av = deviceLduView(dm, diagA, mUp, mLo);
        for (int k = 0; k < 3; ++k)
        {
            DeviceBuffer<scalar> bdDiag;
            deviceCopy(bdDiag, cmptAvIC);
            deviceAxpy(-1.0, iC[k], bdDiag);
            DeviceBuffer<scalar> Hk;
            deviceParallelMatrixH(av, dm, halo, faceCellsD, ifCoeff, Uk[k], relaxSrc[k], bdDiag, bC[k], Hk);
            deviceHadamard(HbyA[k], rAU, Hk);
        }

        // ---- phiHbyA = flux(HbyA): internal + boundary (processor faces get the coupled value) ----
        DeviceBuffer<scalar> hb[3];
        for (int k = 0; k < 3; ++k)
        {
            deviceBCValue(dbU.comp[k], HbyA[k], hb[k]);                  // real BCs; bcType 8 untouched
            halo.exchange(HbyA[k].data());
            halo.scatterBoundaryValues(HbyA[k].data(), weightsD, procStart, hb[k].data());
            halo.waitExchange();                                         // recv-buffer hazard between components
        }
        DeviceBuffer<scalar> phiHbyAint, phiHbyAbnd;
        deviceVectorFlux(dm, HbyA[0], HbyA[1], HbyA[2], phiHbyAint);
        deviceBoundaryFlux(dm, hb[0], hb[1], hb[2], phiHbyAbnd);

        // ---- pEqn: laplacian(rAU,p) == div(phiHbyA) ----
        // rAU at faces: interpolate internally; at boundaries the adjacent-cell value (host rAUfld is
        // zeroGradient on real patches), with processor faces getting the halo-interpolated value.
        DeviceBuffer<scalar> rAUf_int;
        deviceInterpolate(dm, rAU, rAUf_int);
        GeometricField<scalar> rAUfldDev = distributeFromCells<scalar>(rAUhost, P);   // zeroGradient + processor
        const DeviceBoundary dbRAU = buildDeviceBoundary(rAUfldDev, lp, lg);
        DeviceBuffer<scalar> rAUbnd;
        deviceBCValue(dbRAU, rAU, rAUbnd);                          // real patches: adjacent-cell value
        halo.exchange(rAU.data());
        halo.scatterBoundaryValues(rAU.data(), weightsD, procStart, rAUbnd.data());
        halo.waitExchange();

        DeviceBuffer<scalar> pD, pU_, pL_;
        deviceLaplacianCoeffs(dm, rAUf_int, pD, pU_, pL_, false);   // local laplacian(rAU, .)
        // processor coupling: coeff = rAU_f * magSf * procDelta  (rAU_f = the halo-interpolated face value)
        const std::vector<scalar> rAUbndH = rAUbnd.host();
        std::vector<DeviceBuffer<scalar>> pCoeffGeo;
        {
            std::size_t pj3 = 0;
            label bi = 0;
            for (std::size_t pi = 0; pi < lp.size(); ++pi)
            {
                if (lp[pi].type == "cyclic" || lp[pi].type == "cyclicAMI") continue;
                if (lp[pi].type == "processor")
                {
                    std::vector<scalar> cg(lp[pi].size);
                    for (label i = 0; i < lp[pi].size; ++i)
                        cg[i] = rAUbndH[bi + i] * lg.magSf()[lp[pi].start + i] * P.procDelta[pj3][i];
                    DeviceBuffer<scalar> cgd;
                    cgd.copyFrom(cg);
                    pCoeffGeo.push_back(std::move(cgd));
                    ++pj3;
                }
                bi += lp[pi].size;
            }
        }
        std::vector<DeviceBuffer<scalar>> pIfCoeff;
        deviceLaplacianInterface(halo, faceCellsD, pCoeffGeo, pD, pIfCoeff);

        // source = V * div(phiHbyA); boundary coeffs from laplacian(rAU) on p's BCs
        DeviceBuffer<scalar> dphi, psrc;
        deviceDiv(dm, phiHbyAint, phiHbyAbnd, dphi);
        deviceHadamard(psrc, dm.V, dphi);
        DeviceBuffer<scalar> piC, pbC;
        deviceBCLaplacianCoeffsFace(dbP, rAUbnd, piC, pbC);
        DeviceBuffer<scalar> pDiagC, pb;
        deviceFold(dm, pD, psrc, piC, pbC, pDiagC, pb);

        DeviceBuffer<scalar> pSol;
        deviceCopy(pSol, dp);                                        // solve from the current p
        const DeviceLduView pv = deviceLduView(dm, pDiagC, pU_, pL_);
        const scalar nfp = deviceParallelNormFactor(pv, halo, pIfCoeff, pSol, pb, ones_d, P.globalNCells);
        deviceParallelJacobiPCG(pv, halo, pIfCoeff, pb, pSol, nfp, tolP, 0.0, maxIter);
        cudaDeviceSynchronize();

        // compare p to the host pEqn solution
        {
            const std::vector<scalar> pd = pSol.host();
            scalar n4 = 0, d4 = 0;
            for (label c = 0; c < lnC; ++c)
            {
                n4 = std::fmax(n4, std::fabs(pd[c] - phost[c]));
                d4 = std::fmax(d4, std::fabs(phost[c]));
            }
            relP = Pstream::allReduce(d4 > 0 ? n4 / d4 : n4, ReduceOp::Max);
        }

        // ---- compare rAU / HbyA / phiHbyA to the host (EVERY stage is gated, not just printed) ----
        {
            const std::vector<scalar> rAUd = rAU.host();
            scalar n1 = 0, d1 = 0;
            for (label c = 0; c < lnC; ++c)
            {
                n1 = std::fmax(n1, std::fabs(rAUd[c] - rAUhost[c]));
                d1 = std::fmax(d1, std::fabs(rAUhost[c]));
            }
            relRAU = Pstream::allReduce(d1 > 0 ? n1 / d1 : n1, ReduceOp::Max);

            const std::vector<scalar> hx = HbyA[0].host(), hy = HbyA[1].host(), hz = HbyA[2].host();
            scalar n2 = 0, d2 = 0;
            for (label c = 0; c < lnC; ++c)
            {
                const vector dd{hx[c] - HbyAhost[c].x, hy[c] - HbyAhost[c].y, hz[c] - HbyAhost[c].z};
                n2 = std::fmax(n2, brae::mag(dd));
                d2 = std::fmax(d2, brae::mag(HbyAhost[c]));
            }
            relHbyA = Pstream::allReduce(d2 > 0 ? n2 / d2 : n2, ReduceOp::Max);

            const std::vector<scalar> pf = phiHbyAint.host();
            scalar n3 = 0, d3 = 0;
            for (label f = 0; f < nIf; ++f)
            {
                n3 = std::fmax(n3, std::fabs(pf[f] - phiHbyAhost.internal[f]));
                d3 = std::fmax(d3, std::fabs(phiHbyAhost.internal[f]));
            }
            relPhi = Pstream::allReduce(d3 > 0 ? n3 / d3 : n3, ReduceOp::Max);
        }

        const std::vector<scalar> ux = Uk[0].host(), uy = Uk[1].host(), uz = Uk[2].host();
        scalar maxAbs = 0, mag = 0;
        for (label c = 0; c < lnC; ++c)
        {
            const vector d{ux[c] - Uhost[c].x, uy[c] - Uhost[c].y, uz[c] - Uhost[c].z};
            maxAbs = std::fmax(maxAbs, brae::mag(d));
            mag    = std::fmax(mag, brae::mag(Uhost[c]));
        }
        gMax = Pstream::allReduce(maxAbs, ReduceOp::Max);
        gMag = Pstream::allReduce(mag, ReduceOp::Max);
    }

    const scalar rel = gMag > 0 ? gMax / gMag : gMax;
    // Every composed stage is gated. rAU is pure local arithmetic -> machine precision; U/HbyA/phiHbyA carry
    // the predictor's solver tolerance (host PBiCGStab/DILU vs device Jacobi-BiCGStab).
    const bool pass = (rel < 1e-6) && (relRAU < 1e-11) && (relHbyA < 1e-6) && (relPhi < 1e-6)
                   && (relP < 1e-6);
    if (Pstream::master())
    {
        std::printf("test_gpu_parallel_predictor np=%d: U rel=%.3e | rAU rel=%.3e | HbyA rel=%.3e | "
                    "phiHbyA rel=%.3e | p rel=%.3e\n", nproc, rel, relRAU, relHbyA, relPhi, relP);
        std::printf("%s\n", pass ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return pass ? 0 : 1;
}
