// lduInterface increment 2b: parallel PBiCGStab (distributed) solves the same asymmetric system as
// the serial PBiCGStab. Convection-diffusion matrix div(phi,T) - laplacian(1,T); solve serially and
// distributed-parallel, compare. np=1 identical; np>1 agrees to the solver tolerance (local DILU).
// Run: mpirun -np {1,2,4,8} test_parallel_pbicgstab <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "pbicgstab.cuh"
#include "scotch_decomposition.cuh"
#include "domain_decomposition.cuh"
#include "parallel_amul.cuh"
#include "parallel_pbicgstab.cuh"
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

    GeometricField<vector> U = buildField<vector>(readField<vector>(caseDir + "/282/U"), patches, nC);  U.evaluateBoundary();
    GeometricField<scalar> T = buildField<scalar>(readField<scalar>(caseDir + "/282/p"), patches, nC);  T.evaluateBoundary();
    const SurfaceScalarField phi = fvc::flux(U, m, g, patches);

    // Asymmetric convection-diffusion matrix (upwind div => upper != lower).
    FvScalarMatrix M = fvm::div(phi.internal, phi.boundary, T, m, patches);
    addEqual(M, fvm::laplacian(T, 1.0, m, g, patches), -1.0);
    for (label c = 0; c < nC; ++c) M.source[c] = g.V()[c] * (1.0 + 0.5 * std::sin(40.0 * g.C()[c].x));

    const scalar tol = 1e-8;
    std::vector<scalar> psiS(nC, 0.0);
    const SolverPerformance ps = pbicgstab(M, psiS, m, patches, tol, 0.0, 2000);

    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(m, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    DomainDecomposition dd(m, cellToPart, rank);
    DistributedMatrix L = distribute(M, dd);
    foldBoundary(L, M, patches, dd.cellProcAddressing(), cellToPart, rank);

    std::vector<scalar> psiL(dd.nLocalCells(), 0.0);
    const SolverPerformance pp = parallelPBiCGStab(L, psiL, dd.interfaces(), nC, tol, 0.0, 2000);

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
        const scalar gate = (nproc == 1) ? 1e-12 : 1e-6;
        std::printf("test_parallel_pbicgstab np=%d: parallel vs serial rel=%.3e  (serial %d iters | par %d iters)\n",
                    nproc, rel, ps.nIterations, pp.nIterations);
        std::printf("%s\n", rel < gate ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return 0;
}
