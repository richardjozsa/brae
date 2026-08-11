// Phase 3 gate: the DEVICE distributed Jacobi-PCG solves the same system as the single-GPU device PCG.
//
// Build a symmetric laplacian with a deterministic RHS; solve it (a) on one GPU with deviceJacobiPCG over the
// global matrix and (b) distributed with deviceParallelJacobiPCG (deviceParallelAmul + global reductions) over
// the SCOTCH-decomposed partitions. Same Jacobi algorithm, so np=1 must match to machine precision (identical
// computation: no interfaces, all-reduce over one rank is the identity); np>1 agrees to the solver tolerance.
//
// Run: mpirun -np {1,2,4,8} test_gpu_parallel_pcg <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "scotch_decomposition.cuh"
#include "domain_decomposition.cuh"
#include "parallel_amul.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_halo.cuh"
#include "device_buffer.cuh"
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
        if (Pstream::master()) std::printf("test_gpu_parallel_pcg: NVSHMEM inactive -- SKIP\n");
        Pstream::finalize();
        return 0;
    }
    const std::string caseDir = argv[1];

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();

    GeometricField<scalar> p = buildField<scalar>(readField<scalar>(caseDir + "/282/p"), patches, nC);
    p.evaluateBoundary();
    FvScalarMatrix M = fvm::laplacian(p, 1.0, m, g, patches);          // symmetric (orthogonal laplacian)
    for (label c = 0; c < nC; ++c) M.source[c] = g.V()[c] * (1.0 + 0.5 * std::sin(40.0 * g.C()[c].x));

    const scalar tol = 1e-8;
    const int    maxIter = 5000;
    const std::vector<label> ownerInt(m.owner().begin(), m.owner().begin() + nIf);

    // --- serial device reference: boundary-fold the global matrix, then deviceJacobiPCG over the whole mesh.
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
        ps = deviceJacobiPCG(dMg.view(), bG_d, psi_d, nf, tol, 0.0, maxIter);
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
        for (auto& itf : dd.interfaces()) { nbrParts.push_back(itf.neighbProcNo()); faceCells.push_back(itf.faceCells()); }
        DeviceHalo halo(rank, nbrParts, faceCells);

        std::vector<DeviceBuffer<scalar>> coeffs(L.interfaceCoeffs.size());
        for (std::size_t i = 0; i < L.interfaceCoeffs.size(); ++i) coeffs[i].copyFrom(L.interfaceCoeffs[i]);

        const scalar nf = deviceParallelNormFactor(dMl.view(), halo, coeffs, psi_d, b_d, ones_d, nC);
        pp = deviceParallelJacobiPCG(dMl.view(), halo, coeffs, b_d, psi_d, nf, tol, 0.0, maxIter);
        psiL = psi_d.host();
    }

    // Compare local solution to the serial solution on the same cells.
    scalar maxAbs = 0, mag = 0;
    for (std::size_t lc = 0; lc < addr.size(); ++lc)
    {
        maxAbs = std::fmax(maxAbs, std::fabs(psiL[lc] - psiS[addr[lc]]));
        mag    = std::fmax(mag, std::fabs(psiS[addr[lc]]));
    }
    const scalar gMax = Pstream::allReduce(maxAbs, ReduceOp::Max);
    const scalar gMag = Pstream::allReduce(mag, ReduceOp::Max);
    const scalar rel  = gMag > 0 ? gMax / gMag : gMax;
    const scalar gate = (nproc == 1) ? 1e-10 : 1e-6;      // np1 identical algorithm; np>1 ~ solver tol
    const bool   pass = rel < gate;
    if (Pstream::master())
    {
        std::printf("test_gpu_parallel_pcg np=%d: device parallel vs serial PCG rel=%.3e  (serial %d iters | par %d iters)\n",
                    nproc, rel, ps.nIterations, pp.nIterations);
        std::printf("%s\n", pass ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return pass ? 0 : 1;
}
