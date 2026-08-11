// lduInterface increment 4d: a parallel scalar-transport SOLVE with the combined operator
// div(phi,T) - laplacian(gamma,T) = S (the momentum's structure). Each rank locally assembles
// convection + (-1)*laplacian, folds the boundary, and solves with the parallel PBiCGStab; the
// gathered solution matches serial. Run: mpirun -np {1,2,4,8} test_parallel_transport <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "pbicgstab.cuh"
#include "scotch_decomposition.cuh"
#include "local_mesh.cuh"
#include "local_assembly.cuh"
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

    PrimitiveMesh gm;  gm.read(caseDir + "/constant/polyMesh");
    FvGeometry gg;     gg.build(gm);
    const std::vector<FvPatch> gpatches = buildPatches(gm, gg);
    const label nC = gm.nCells(), nIf = gm.nInternalFaces();

    GeometricField<vector> U = buildField<vector>(readField<vector>(caseDir + "/282/U"), gpatches, nC);  U.evaluateBoundary();
    GeometricField<scalar> T = buildField<scalar>(readField<scalar>(caseDir + "/282/p"), gpatches, nC);  T.evaluateBoundary();
    const SurfaceScalarField phi = fvc::flux(U, gm, gg, gpatches);
    std::vector<scalar> gammaCell(nC);
    for (label c = 0; c < nC; ++c) gammaCell[c] = 1.0 + 0.5 * std::sin(30.0 * gg.C()[c].x);
    const SurfaceScalarField gammaf = fvc::interpolate(gammaCell, gm, gg, gpatches);

    FvScalarMatrix M = fvm::div(phi.internal, phi.boundary, T, gm, gpatches);
    addEqual(M, fvm::laplacian(gammaf, T, gm, gg, gpatches), -1.0);
    for (label c = 0; c < nC; ++c) M.source[c] = gg.V()[c] * (1.0 + 0.5 * std::sin(40.0 * gg.C()[c].x));

    const scalar tol = 1e-8;
    std::vector<scalar> psiS(nC, 0.0);
    const SolverPerformance ps = pbicgstab(M, psiS, gm, gpatches, tol, 0.0, 2000);

    // Local combined assembly.
    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    LocalMesh Lm = buildLocalMesh(gm, cellToPart, rank);
    FvGeometry lg; lg.build(Lm.mesh);
    const std::vector<FvPatch> lpatches = buildPatches(Lm.mesh, lg);
    const auto procDelta = computeProcDeltaCoeffs(Lm, lg, lpatches);

    std::vector<scalar> localPhi(Lm.mesh.nFaces(), 0.0), localGammaF(Lm.mesh.nFaces(), 0.0);
    for (label f = 0; f < Lm.mesh.nFaces(); ++f) {
        const label gf = Lm.faceGlobal[f];
        if (gf < nIf) { localPhi[f] = phi.internal[gf] * (Lm.faceFlip[f] ? -1.0 : 1.0); localGammaF[f] = gammaf.internal[gf]; }
    }
    DistributedMatrix L = assembleLocalConvection(Lm, lpatches, localPhi);
    addEqualDist(L, assembleLocalLaplacianF(Lm, lg, lpatches, procDelta, localGammaF), -1.0);
    foldBoundary(L, M, gpatches, Lm.cellProcAddr, cellToPart, rank);

    std::vector<ProcessorInterface> interfaces;
    for (std::size_t j = 0; j < Lm.procNbr.size(); ++j) interfaces.emplace_back(rank, Lm.procNbr[j], Lm.procFaceCells[j]);

    std::vector<scalar> psiL(Lm.mesh.nCells(), 0.0);
    const SolverPerformance pp = parallelPBiCGStab(L, psiL, interfaces, nC, tol, 0.0, 2000);

    const std::vector<label>& addr = Lm.cellProcAddr;
    scalar maxAbs = 0, mag = 0;
    for (std::size_t lc = 0; lc < addr.size(); ++lc) { maxAbs = std::fmax(maxAbs, std::fabs(psiL[lc] - psiS[addr[lc]])); mag = std::fmax(mag, std::fabs(psiS[addr[lc]])); }
    const scalar gMax = Pstream::allReduce(maxAbs, ReduceOp::Max), gMag = Pstream::allReduce(mag, ReduceOp::Max);

    if (Pstream::master()) {
        const scalar rel = gMag > 0 ? gMax / gMag : gMax;
        const scalar gate = (nproc == 1) ? 1e-8 : 1e-6;   // combined assembly vs serial M: FP order
        std::printf("test_parallel_transport np=%d: local div-laplacian parallel solve vs serial rel=%.3e  (serial %d | par %d it)\n",
                    nproc, rel, ps.nIterations, pp.nIterations);
        std::printf("%s\n", rel < gate ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return 0;
}
