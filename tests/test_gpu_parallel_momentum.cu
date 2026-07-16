// Phase 4a L1: the DEVICE momentum-matrix assembly on a decomposed mesh == the host momentumDistributed.
//
// Assemble M = div(phi,U) - laplacian(nuEff,U) on the local mesh: local coeffs via deviceDivUpwindCoeffs +
// deviceLaplacianCoeffs, processor coupling via deviceMomentumInterface. Compare diag/upper/lower and the
// per-interface coeffs to the host DistributedMatrix (the validated parallelSimpleStepLaminar assembly).
//
// Run: mpirun -np {1,2,4,8} test_gpu_parallel_momentum <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "scotch_decomposition.cuh"
#include "parallel_simple.cuh"       // Partition, outwardFlux
#include "local_assembly.cuh"        // momentumDistributed
#include "field_distribute.cuh"
#include "device_mesh.cuh"
#include "device_ldu.cuh"
#include "device_blas.cuh"
#include "device_halo.cuh"
#include "device_boundary.cuh"
#include "device_simple.cuh"
#include "parallel_matrix_ops.cuh"
#include "device_buffer.cuh"
#include "parallel_device_simple.cuh"
#include "cf_pstream.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

static scalar relDiff(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar num = 0, den = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
    {
        num = std::fmax(num, std::fabs(a[i] - b[i]));
        den = std::fmax(den, std::fabs(b[i]));
    }
    return den > 0 ? num / den : num;
}

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
        if (Pstream::master()) std::printf("test_gpu_parallel_momentum: NVSHMEM inactive -- SKIP\n");
        Pstream::finalize();
        return 0;
    }
    const std::string caseDir = argv[1];
    const scalar nu = 1e-3;

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

    // Distributed U + its flux phi (processor faces carry the coupled interpolated value / flux).
    const FieldData<vector> Ufd = readField<vector>(caseDir + "/282/U");
    GeometricField<vector> U = distributeField<vector>(Ufd, gm.patches(), P.Lm, lp, P.procW, rank);
    U.evaluateBoundary();
    const SurfaceScalarField phi = fvc::flux(U, m, lg, lp);

    // nuEff = uniform nu: internal face field = nu, real boundary = nu, all faces = nu.
    const std::vector<scalar> nuEffCell(lnC, nu);
    SurfaceScalarField nuEfff0 = fvc::interpolate(nuEffCell, m, lg, lp);
    std::vector<std::vector<scalar>> nuEffBnd(lp.size());
    for (std::size_t pi = 0; pi < lp.size(); ++pi) nuEffBnd[pi].assign(lp[pi].size, nu);
    nuEfff0.boundary = nuEffBnd;
    const std::vector<scalar> nuF(m.nFaces(), nu);

    // HOST oracle: M = div(phi,U) - laplacian(nuEff,U), distributed.
    const std::vector<scalar> outPhi = outwardFlux(P, phi);
    FvVectorMatrix Ml = fvm::div(phi.internal, phi.boundary, U, m, lp);
    addEqual(Ml, fvm::laplacian(nuEfff0, U, m, lg, lp), -1.0);
    const DistributedMatrix L = momentumDistributed(Ml, P.Lm, lg, lp, outPhi, nuF, P.procDelta);

    int fails = 0;
    {
        const DeviceMesh dm = buildDeviceMesh(m, lg, lp);
        // local coeffs: div(phi) - laplacian(nuEff).
        DeviceBuffer<scalar> phiInt_d(std::vector<scalar>(phi.internal.begin(), phi.internal.begin() + nIf));
        DeviceBuffer<scalar> nuEff_fd(std::vector<scalar>(nuEfff0.internal.begin(), nuEfff0.internal.begin() + nIf));
        DeviceBuffer<scalar> mDiag, mUp, mLo, lD, lU, lL;
        deviceDivUpwindCoeffs(dm, phiInt_d, mDiag, mUp, mLo);
        deviceLaplacianCoeffs(dm, nuEff_fd, lD, lU, lL, false);
        deviceAxpy(-1.0, lD, mDiag);
        deviceAxpy(-1.0, lU, mUp);
        deviceAxpy(-1.0, lL, mLo);

        // processor coupling: gather each processor patch's flux + geometric coeff, in interface order.
        std::vector<int> nbrParts;
        for (int q : P.Lm.procNbr) nbrParts.push_back(q);
        DeviceHalo halo(rank, nbrParts, P.Lm.procFaceCells);

        std::vector<DeviceBuffer<label>>  faceCellsD;
        std::vector<DeviceBuffer<scalar>> phiF, coeffGeo;
        std::size_t pj = 0;
        for (std::size_t pi = 0; pi < lp.size(); ++pi)
        {
            if (lp[pi].type != "processor") continue;
            DeviceBuffer<label> fcd;
            fcd.copyFrom(P.Lm.procFaceCells[pj]);
            faceCellsD.push_back(std::move(fcd));
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
        cudaDeviceSynchronize();

        // compare assembled coeffs to the host oracle.
        const scalar rDiag = relDiff(mDiag.host(), L.diag);
        const scalar rUp   = relDiff(mUp.host(), L.upper);
        const scalar rLo   = relDiff(mLo.host(), L.lower);
        scalar rIf = 0;
        for (int i = 0; i < halo.nInterfaces(); ++i)
            if (halo.size(i) > 0) rIf = std::fmax(rIf, relDiff(ifCoeff[i].host(), L.interfaceCoeffs[i]));

        if (rDiag > 1e-11) { std::printf("[rank %d] FAIL diag rel=%.3e\n", rank, rDiag); ++fails; }
        if (rUp   > 1e-11) { std::printf("[rank %d] FAIL upper rel=%.3e\n", rank, rUp); ++fails; }
        if (rLo   > 1e-11) { std::printf("[rank %d] FAIL lower rel=%.3e\n", rank, rLo); ++fails; }
        if (rIf   > 1e-11) { std::printf("[rank %d] FAIL ifCoeff rel=%.3e\n", rank, rIf); ++fails; }

        // The pressure equation's coupling: a laplacian(gamma, .) interface (host assembleLocalLaplacianF).
        const scalar gam = 0.5;
        const std::vector<scalar> gammaF(m.nFaces(), gam);
        const DistributedMatrix Lp = assembleLocalLaplacianF(P.Lm, lg, lp, P.procDelta, gammaF);

        DeviceBuffer<scalar> gam_fd(std::vector<scalar>(nIf, gam));
        DeviceBuffer<scalar> pD, pU, pL_;
        deviceLaplacianCoeffs(dm, gam_fd, pD, pU, pL_, false);      // local internal-face laplacian
        std::vector<DeviceBuffer<scalar>> pCoeffGeo;
        pj = 0;
        for (std::size_t pi = 0; pi < lp.size(); ++pi)
        {
            if (lp[pi].type != "processor") continue;
            std::vector<scalar> cg(lp[pi].size);
            for (label f = 0; f < lp[pi].size; ++f)
                cg[f] = gam * lg.magSf()[lp[pi].start + f] * P.procDelta[pj][f];
            DeviceBuffer<scalar> cgd;
            cgd.copyFrom(cg);
            pCoeffGeo.push_back(std::move(cgd));
            ++pj;
        }
        std::vector<DeviceBuffer<scalar>> pIfCoeff;
        deviceLaplacianInterface(halo, faceCellsD, pCoeffGeo, pD, pIfCoeff);
        cudaDeviceSynchronize();

        const scalar rpDiag = relDiff(pD.host(), Lp.diag);
        scalar rpIf = 0;
        for (int i = 0; i < halo.nInterfaces(); ++i)
            if (halo.size(i) > 0) rpIf = std::fmax(rpIf, relDiff(pIfCoeff[i].host(), Lp.interfaceCoeffs[i]));
        if (rpDiag > 1e-11) { std::printf("[rank %d] FAIL pEqn diag rel=%.3e\n", rank, rpDiag); ++fails; }
        if (rpIf   > 1e-11) { std::printf("[rank %d] FAIL pEqn ifCoeff rel=%.3e\n", rank, rpIf); ++fails; }

        // fvMatrix::relax on the distributed matrix == host parallelRelaxMatrix. The processor interface must
        // count toward diagonal dominance (deviceInterfaceOffDiagSum -> deviceRelaxDiag's cycSumOff hook), and
        // the boundary iC must be ZERO on processor faces (bcType 8) or the diagonal double-counts.
        const scalar relaxU = 0.7;
        const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, lp, lg);
        std::vector<scalar> phiBndH, nuEffBndH;
        for (std::size_t pi = 0; pi < lp.size(); ++pi)
        {
            if (lp[pi].type == "cyclic" || lp[pi].type == "cyclicAMI") continue;
            for (label i = 0; i < lp[pi].size; ++i)
            {
                phiBndH.push_back(phi.boundary[pi][i]);
                nuEffBndH.push_back(nu);
            }
        }
        DeviceBuffer<scalar> phiBnd_d(phiBndH), nuEffBnd_d(nuEffBndH);
        DeviceBuffer<scalar> r0IC, r0BC, r0lIC, r0lBC;
        deviceBCDivCoeffs(dbU.comp[0], phiBnd_d, r0IC, r0BC);              // iC = divIC - lapIC  (as device_simple_foam)
        deviceBCLaplacianCoeffsFace(dbU.comp[0], nuEffBnd_d, r0lIC, r0lBC);
        deviceAxpy(-1.0, r0lIC, r0IC);

        DeviceBuffer<scalar> sumOff(std::vector<scalar>(lnC, 0.0));
        deviceInterfaceOffDiagSum(halo, faceCellsD, ifCoeff, sumOff);
        DeviceBuffer<scalar> mDiagR, delta;
        deviceRelaxDiag(deviceLduView(dm, mDiag, mUp, mLo), dm, r0IC, relaxU, mDiagR, delta, sumOff.data());
        cudaDeviceSynchronize();

        // Host oracle: parallelRelaxMatrix mutates L.diag in place (and Ml.source, unused here).
        DistributedMatrix Lr = L;
        FvVectorMatrix Mr = Ml;
        parallelRelaxMatrix(Lr, Mr, U.internal, P.Lm, lp, relaxU);
        const scalar rRelax = relDiff(mDiagR.host(), Lr.diag);
        if (rRelax > 1e-11) { std::printf("[rank %d] FAIL relaxed diag rel=%.3e\n", rank, rRelax); ++fails; }
    }

    const label totalFail = Pstream::allReduce(static_cast<label>(fails), ReduceOp::Sum);
    if (Pstream::master())
        std::printf("test_gpu_parallel_momentum: nproc=%d  %s (failures=%d)\n",
                    nproc, totalFail == 0 ? "PASS" : "FAIL", static_cast<int>(totalFail));

    Pstream::finalize();
    return totalFail == 0 ? 0 : 1;
}
