// lduInterface increment 2: parallel PCG (distributed) solves the same system as the serial PCG.
// Build a symmetric laplacian, solve it serially and distributed-parallel from psi=0, and compare
// the converged solutions. np=1 must match to machine precision (identical algorithm); np>1 agrees
// to the solver tolerance (parallel DIC is local per OpenFOAM, so the convergence path differs).
// Run: mpirun -np {1,2,4,8} test_parallel_pcg <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "pcg.cuh"
#include "scotch_decomposition.cuh"
#include "domain_decomposition.cuh"
#include "parallel_amul.cuh"
#include "parallel_pcg.cuh"
#include "cf_pstream.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    Pstream::init(argc, argv);
    const int rank = Pstream::myProcNo(), nproc = Pstream::nProcs();
    if (argc < 2) { if (Pstream::master()) std::printf("usage: %s <caseDir>\n", argv[0]); Pstream::finalize(); return 2; }
    const std::string caseDir = argv[1];

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    GeometricField<scalar> p = buildField<scalar>(readField<scalar>(caseDir + "/282/p"), patches, nC);  p.evaluateBoundary();
    FvScalarMatrix M = fvm::laplacian(p, 1.0, m, g, patches);   // symmetric (orthogonal laplacian)
    // Non-trivial deterministic RHS (same on every rank: C/V are global) -> a real Poisson solve.
    for (label c = 0; c < nC; ++c) M.source[c] = g.V()[c] * (1.0 + 0.5 * std::sin(40.0 * g.C()[c].x));

    const scalar tol = 1e-8;
    // Serial reference.
    std::vector<scalar> psiS(nC, 0.0);
    const SolverPerformance ps = pcg(M, psiS, m, patches, tol, 0.0, 2000);

    // Distributed parallel.
    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(m, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    DomainDecomposition dd(m, cellToPart, rank);
    DistributedMatrix L = distribute(M, dd);
    foldBoundary(L, M, patches, dd.cellProcAddressing(), cellToPart, rank);

    std::vector<scalar> psiL(dd.nLocalCells(), 0.0);
    const SolverPerformance pp = parallelPCG(L, psiL, dd.interfaces(), nC, tol, 0.0, 2000);

    // Compare local solution to the serial solution on the same cells.
    const std::vector<label>& addr = dd.cellProcAddressing();
    scalar maxAbs = 0, mag = 0;
    for (std::size_t lc = 0; lc < addr.size(); ++lc) {
        maxAbs = std::fmax(maxAbs, std::fabs(psiL[lc] - psiS[addr[lc]]));
        mag    = std::fmax(mag, std::fabs(psiS[addr[lc]]));
    }
    const scalar gMax = Pstream::allReduce(maxAbs, ReduceOp::Max);
    const scalar gMag = Pstream::allReduce(mag, ReduceOp::Max);

    if (Pstream::master()) {
        const scalar rel = gMag > 0 ? gMax / gMag : gMax;
        const scalar gate = (nproc == 1) ? 1e-12 : 1e-6;   // np1 identical; np>1 ~ solver tol
        std::printf("test_parallel_pcg np=%d: parallel vs serial PCG rel=%.3e  (serial %d iters res %.1e->%.1e | par %d iters %.1e->%.1e)\n",
                    nproc, rel, ps.nIterations, ps.initialResidual, ps.finalResidual, pp.nIterations, pp.initialResidual, pp.finalResidual);
        std::printf("%s\n", rel < gate ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return 0;
}
