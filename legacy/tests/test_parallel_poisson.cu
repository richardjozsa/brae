// lduInterface increment 4d: a full parallel SOLVE pipeline on a real PDE. Each rank LOCALLY
// assembles a variable-gamma laplacian from its own mesh (processor faces -> interface coeffs),
// folds the boundary, and solves with the parallel PCG; the gathered solution matches the serial
// pcg. This is the end-to-end parallel solver (local mesh -> local assembly -> parallel solve).
// Run: mpirun -np {1,2,4,8} test_parallel_poisson <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "pcg.cuh"
#include "scotch_decomposition.cuh"
#include "local_mesh.cuh"
#include "local_assembly.cuh"
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

    PrimitiveMesh gm;  gm.read(caseDir + "/constant/polyMesh");
    FvGeometry gg;     gg.build(gm);
    const std::vector<FvPatch> gpatches = buildPatches(gm, gg);
    const label nC = gm.nCells(), nIf = gm.nInternalFaces();

    // Variable-gamma laplacian + a deterministic RHS; serial reference solve.
    GeometricField<scalar> p = buildField<scalar>(readField<scalar>(caseDir + "/282/p"), gpatches, nC);  p.evaluateBoundary();
    std::vector<scalar> gammaCell(nC);
    for (label c = 0; c < nC; ++c) gammaCell[c] = 1.0 + 0.5 * std::sin(30.0 * gg.C()[c].x);
    const SurfaceScalarField gammaf = fvc::interpolate(gammaCell, gm, gg, gpatches);
    FvScalarMatrix M = fvm::laplacian(gammaf, p, gm, gg, gpatches);
    for (label c = 0; c < nC; ++c) M.source[c] = gg.V()[c] * (1.0 + 0.5 * std::sin(40.0 * gg.C()[c].x));

    const scalar tol = 1e-8;
    std::vector<scalar> psiS(nC, 0.0);
    const SolverPerformance ps = pcg(M, psiS, gm, gpatches, tol, 0.0, 2000);

    // Local assembly + parallel solve.
    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    LocalMesh Lm = buildLocalMesh(gm, cellToPart, rank);
    FvGeometry lg; lg.build(Lm.mesh);
    const std::vector<FvPatch> lpatches = buildPatches(Lm.mesh, lg);
    const auto procDelta = computeProcDeltaCoeffs(Lm, lg, lpatches);

    std::vector<scalar> localGammaF(Lm.mesh.nFaces(), 0.0);   // face gamma (internal + processor)
    for (label f = 0; f < Lm.mesh.nFaces(); ++f) { const label gf = Lm.faceGlobal[f]; if (gf < nIf) localGammaF[f] = gammaf.internal[gf]; }
    DistributedMatrix L = assembleLocalLaplacianF(Lm, lg, lpatches, procDelta, localGammaF);
    foldBoundary(L, M, gpatches, Lm.cellProcAddr, cellToPart, rank);   // boundary diag/source

    std::vector<ProcessorInterface> interfaces;
    for (std::size_t j = 0; j < Lm.procNbr.size(); ++j) interfaces.emplace_back(rank, Lm.procNbr[j], Lm.procFaceCells[j]);

    std::vector<scalar> psiL(Lm.mesh.nCells(), 0.0);
    const SolverPerformance pp = parallelPCG(L, psiL, interfaces, nC, tol, 0.0, 2000);

    const std::vector<label>& addr = Lm.cellProcAddr;
    scalar maxAbs = 0, mag = 0;
    for (std::size_t lc = 0; lc < addr.size(); ++lc) {
        maxAbs = std::fmax(maxAbs, std::fabs(psiL[lc] - psiS[addr[lc]]));
        mag    = std::fmax(mag, std::fabs(psiS[addr[lc]]));
    }
    const scalar gMax = Pstream::allReduce(maxAbs, ReduceOp::Max);
    const scalar gMag = Pstream::allReduce(mag, ReduceOp::Max);

    if (Pstream::master()) {
        const scalar rel = gMag > 0 ? gMax / gMag : gMax;
        const scalar gate = (nproc == 1) ? 1e-9 : 1e-6;   // np1: two solver impls, ~1e-12 FP drift
        std::printf("test_parallel_poisson np=%d: local-assembled parallel solve vs serial rel=%.3e  (serial %d it | par %d it)\n",
                    nproc, rel, ps.nIterations, pp.nIterations);
        std::printf("%s\n", rel < gate ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return 0;
}
