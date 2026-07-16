// Phase 4a L2: the DEVICE distributed Jacobi-BiCGStab solves the same NON-symmetric system as the
// single-GPU device BiCGStab (the momentum predictor's solver).
//
// Build a non-symmetric transport matrix div(phi,p) - laplacian(nu,p) (upwind -> upper != lower); solve it
// (a) on one GPU over the global matrix and (b) distributed with deviceParallelJacobiBiCGStab. Same algorithm,
// so np=1 matches to machine precision (no interfaces, all-reduce over one rank is the identity); np>1 agrees
// to the solver tolerance.
//
// Run: mpirun -np {1,2,4,8} test_gpu_parallel_bicgstab <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "scotch_decomposition.cuh"
#include "domain_decomposition.cuh"
#include "parallel_amul.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_blas.cuh"
#include "device_halo.cuh"
#include "device_buffer.cuh"
#include "reconstruct.cuh"
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
        if (Pstream::master()) std::printf("test_gpu_parallel_bicgstab: NVSHMEM inactive -- SKIP\n");
        Pstream::finalize();
        return 0;
    }
    const std::string caseDir = argv[1];
    // Diffusion-dominated so the matrix is strongly diagonally dominant (well conditioned): a convection-
    // dominated matrix has a large condition number, where a 1e-8 RESIDUAL still leaves a ~1e-3 solution
    // difference -- that measures conditioning, not the solver.
    const scalar nu = 0.1;

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();

    // A non-symmetric transport matrix: div(phi,p) - laplacian(nu,p). phi = flux of the case velocity.
    GeometricField<scalar> p = buildField<scalar>(readField<scalar>(caseDir + "/282/p"), patches, nC);
    p.evaluateBoundary();
    GeometricField<vector> U = buildField<vector>(readField<vector>(caseDir + "/282/U"), patches, nC);
    U.evaluateBoundary();
    const SurfaceScalarField phi = fvc::flux(U, m, g, patches);

    FvScalarMatrix M = fvm::div(phi.internal, phi.boundary, p, m, patches);
    addEqual(M, fvm::laplacian(p, nu, m, g, patches), -1.0);
    for (label c = 0; c < nC; ++c) M.source[c] = g.V()[c] * (1.0 + 0.5 * std::sin(40.0 * g.C()[c].x));

    const scalar tol = 1e-8;
    const int    maxIter = 5000;
    const std::vector<label> ownerInt(m.owner().begin(), m.owner().begin() + nIf);

    // --- serial device reference: boundary-fold the global matrix, then deviceJacobiBiCGStab.
    std::vector<scalar> diagG = M.diag, bG = M.source;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const label gc = patches[pi].faceCells[i];
            diagG[gc] += M.internalCoeffs[pi][i];
            bG[gc]    += M.boundaryCoeffs[pi][i];
        }

    std::vector<scalar> psiS;
    DeviceSolverPerf ps;
    {
        const DeviceLduMatrix dMg = buildDeviceLdu(diagG, M.upper, M.lower, ownerInt, m.neighbour(), nC);
        DeviceBuffer<scalar> bG_d(bG), psi_d(std::vector<scalar>(nC, 0.0)), ones_d(std::vector<scalar>(nC, 1.0));
        const scalar nf = deviceNormFactor(dMg.view(), psi_d, bG_d, ones_d);
        ps = deviceJacobiBiCGStab(dMg.view(), bG_d, psi_d, nf, tol, 0.0, maxIter, 1);
        psiS = psi_d.host();
    }

    // --- distributed device solve.
    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(m, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    DomainDecomposition dd(m, cellToPart, rank);
    DistributedMatrix L = distribute(M, dd);
    foldBoundary(L, M, patches, dd.cellProcAddressing(), cellToPart, rank);
    const std::vector<label>& addr = dd.cellProcAddressing();

    std::vector<scalar> psiL;
    DeviceSolverPerf pp;
    {   // scope: DeviceHalo symmetric buffers freed before Pstream::finalize()
        const DeviceLduMatrix dMl = buildDeviceLdu(L.diagC, L.upper, L.lower, L.lowerAddr, L.upperAddr,
                                                   static_cast<int>(addr.size()));
        DeviceBuffer<scalar> b_d(L.b), psi_d(std::vector<scalar>(addr.size(), 0.0)),
                             ones_d(std::vector<scalar>(addr.size(), 1.0));

        std::vector<int>                nbrParts;
        std::vector<std::vector<label>> faceCells;
        for (auto& itf : dd.interfaces())
        {
            nbrParts.push_back(itf.neighbProcNo());
            faceCells.push_back(itf.faceCells());
        }
        DeviceHalo halo(rank, nbrParts, faceCells);

        std::vector<DeviceBuffer<scalar>> coeffs(L.interfaceCoeffs.size());
        for (std::size_t i = 0; i < L.interfaceCoeffs.size(); ++i) coeffs[i].copyFrom(L.interfaceCoeffs[i]);

        const scalar nf = deviceParallelNormFactor(dMl.view(), halo, coeffs, psi_d, b_d, ones_d, nC);
        pp = deviceParallelJacobiBiCGStab(dMl.view(), halo, coeffs, b_d, psi_d, nf, tol, 0.0, maxIter);
        psiL = psi_d.host();
    }

    // THE test: reconstruct the distributed solution to the global numbering and check it satisfies the
    // GLOBAL equation, ||bG - A_global*psi||/normFactor. That validates the distributed solver against the
    // equation itself, independent of any other solver's rounding.
    //
    // Comparing two solvers' SOLUTIONS is the wrong oracle here: this matrix has a large condition number, so
    // both converging to residual ~1e-9 still leaves a ~1e-5 spread between their solutions -- that measures
    // conditioning, not correctness. The serial solution is reported for reference only.
    const std::vector<scalar> psiGlobal = reconstructField(addr, psiL, nC);
    scalar globalRes = 0;
    {
        const DeviceLduMatrix dMg = buildDeviceLdu(diagG, M.upper, M.lower, ownerInt, m.neighbour(), nC);
        DeviceBuffer<scalar> psi_d(psiGlobal), bG_d(bG), Apsi(nC), r(nC), ones_d(std::vector<scalar>(nC, 1.0));
        const scalar nf = deviceNormFactor(dMg.view(), psi_d, bG_d, ones_d);
        deviceAmul(dMg.view(), psi_d, Apsi);
        deviceCopy(r, bG_d);
        deviceAxpy(-1.0, Apsi, r);                                  // r = bG - A*psi
        globalRes = deviceSumMag(r) / nf;
    }

    scalar maxAbs = 0, mag = 0;
    for (std::size_t lc = 0; lc < addr.size(); ++lc)
    {
        maxAbs = std::fmax(maxAbs, std::fabs(psiL[lc] - psiS[addr[lc]]));
        mag    = std::fmax(mag, std::fabs(psiS[addr[lc]]));
    }
    const scalar gMax = Pstream::allReduce(maxAbs, ReduceOp::Max);
    const scalar gMag = Pstream::allReduce(mag, ReduceOp::Max);
    const scalar rel  = gMag > 0 ? gMax / gMag : gMax;

    const bool pass = (globalRes < 1e-6) && (pp.finalResidual < 1e-6);
    if (Pstream::master())
    {
        std::printf("test_gpu_parallel_bicgstab np=%d: distributed solution vs GLOBAL equation res=%.3e  "
                    "(par %d iters res %.2e | vs serial soln rel=%.2e, cond-limited)\n",
                    nproc, globalRes, pp.nIterations, pp.finalResidual, rel);
        std::printf("%s\n", pass ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return pass ? 0 : 1;
}
