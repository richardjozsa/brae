// lduInterface increment 1: distributed (MPI) matrix-vector product == serial Amul.
// Build a global laplacian matrix, SCOTCH-decompose, distribute it, run the interface-coupled
// parallelAmul on each rank, and check every local cell reproduces the serial global Amul.
// Run: mpirun -np {1,2,4,8} test_parallel_spmv <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "ldu_spmv.cuh"
#include "scotch_decomposition.cuh"
#include "domain_decomposition.cuh"
#include "parallel_amul.cuh"
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

    // Global laplacian(1, p) matrix (every rank assembles it; psi = the 282/p field).
    GeometricField<scalar> p = buildField<scalar>(readField<scalar>(caseDir + "/282/p"), patches, nC);  p.evaluateBoundary();
    const FvScalarMatrix M = fvm::laplacian(p, 1.0, m, g, patches);
    const std::vector<scalar>& psiG = p.internal;

    // Serial reference (raw lduMatrix Amul, no interfaces).
    const std::vector<scalar> ApsiSerial = Amul(M, psiG, m);

    // Decompose on master, broadcast.
    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(m, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);

    DomainDecomposition dd(m, cellToPart, rank);
    const DistributedMatrix L = distribute(M, dd);

    // Local psi from the global field; parallel Amul with interface coupling.
    const std::vector<label>& addr = dd.cellProcAddressing();
    std::vector<scalar> psiL(addr.size());
    for (std::size_t lc = 0; lc < addr.size(); ++lc) psiL[lc] = psiG[addr[lc]];
    const std::vector<scalar> ApsiLocal = parallelAmul(L, psiL, dd.interfaces());

    // Each local cell must match the serial global Amul.
    scalar maxAbs = 0, mag = 0;
    for (std::size_t lc = 0; lc < addr.size(); ++lc) {
        maxAbs = std::fmax(maxAbs, std::fabs(ApsiLocal[lc] - ApsiSerial[addr[lc]]));
        mag    = std::fmax(mag, std::fabs(ApsiSerial[addr[lc]]));
    }
    const scalar gMax = Pstream::allReduce(maxAbs, ReduceOp::Max);
    const scalar gMag = Pstream::allReduce(mag, ReduceOp::Max);
    const label  nIf  = Pstream::allReduce((label)dd.nLocalInternalFaces(), ReduceOp::Sum);
    label nCut = 0; for (label f = 0; f < m.nInternalFaces(); ++f) if (cellToPart[m.owner()[f]] != cellToPart[m.neighbour()[f]]) ++nCut;

    if (Pstream::master()) {
        const scalar rel = gMag > 0 ? gMax / gMag : gMax;
        std::printf("test_parallel_spmv np=%d: distributed Amul vs serial rel=%.3e  (localIntFaces=%d cutFaces=%d)\n",
                    nproc, rel, (int)nIf, (int)nCut);
        std::printf("%s\n", rel < 1e-12 ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return 0;
}
