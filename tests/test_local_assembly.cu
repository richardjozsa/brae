// lduInterface increment 4c: a LOCALLY-assembled laplacian (each rank builds its own matrix from its
// local mesh, processor faces -> interface coeffs via the exchanged delta) reproduces the serial
// global Amul. This is the first REAL parallel matrix (not the distribute()-sliced global one) and
// the point where actual per-rank assembly work appears.
// Run: mpirun -np {1,2,4,8} test_local_assembly <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "ldu_spmv.cuh"
#include "scotch_decomposition.cuh"
#include "local_mesh.cuh"
#include "local_assembly.cuh"
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

    PrimitiveMesh gm;  gm.read(caseDir + "/constant/polyMesh");
    FvGeometry gg;     gg.build(gm);
    const std::vector<FvPatch> gpatches = buildPatches(gm, gg);
    const label nC = gm.nCells();

    // Serial reference: global laplacian(1) and a deterministic psi.
    GeometricField<scalar> p = buildField<scalar>(readField<scalar>(caseDir + "/282/p"), gpatches, nC);  p.evaluateBoundary();
    const FvScalarMatrix M = fvm::laplacian(p, 1.0, gm, gg, gpatches);
    std::vector<scalar> psiG(nC);
    for (label c = 0; c < nC; ++c) psiG[c] = std::sin(40.0 * gg.C()[c].x) + 0.5 * gg.C()[c].y;
    const std::vector<scalar> ApsiSerial = Amul(M, psiG, gm);

    // Local mesh + geometry + the locally assembled matrix.
    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    LocalMesh Lm = buildLocalMesh(gm, cellToPart, rank);
    FvGeometry lg; lg.build(Lm.mesh);
    const std::vector<FvPatch> lpatches = buildPatches(Lm.mesh, lg);
    const auto procDelta = computeProcDeltaCoeffs(Lm, lg, lpatches);
    DistributedMatrix L = assembleLocalLaplacian(Lm, lg, lpatches, procDelta, 1.0);

    std::vector<ProcessorInterface> interfaces;
    for (std::size_t j = 0; j < Lm.procNbr.size(); ++j)
        interfaces.emplace_back(rank, Lm.procNbr[j], Lm.procFaceCells[j]);

    const std::vector<label>& addr = Lm.cellProcAddr;
    std::vector<scalar> psiL(addr.size());
    for (std::size_t lc = 0; lc < addr.size(); ++lc) psiL[lc] = psiG[addr[lc]];
    const std::vector<scalar> ApsiLocal = parallelAmul(L, psiL, interfaces);

    scalar maxAbs = 0, mag = 0;
    for (std::size_t lc = 0; lc < addr.size(); ++lc) {
        maxAbs = std::fmax(maxAbs, std::fabs(ApsiLocal[lc] - ApsiSerial[addr[lc]]));
        mag    = std::fmax(mag, std::fabs(ApsiSerial[addr[lc]]));
    }
    const scalar gMax = Pstream::allReduce(maxAbs, ReduceOp::Max);
    const scalar gMag = Pstream::allReduce(mag, ReduceOp::Max);

    if (Pstream::master()) {
        const scalar rel = gMag > 0 ? gMax / gMag : gMax;
        std::printf("test_local_assembly np=%d: locally-assembled Amul vs serial rel=%.3e\n", nproc, rel);
        std::printf("%s\n", rel < 1e-12 ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return 0;
}
