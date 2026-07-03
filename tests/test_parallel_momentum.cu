// lduInterface increment 4d: the parallel MOMENTUM matrix div(phi,U) - laplacian(nu,U). Each rank
// runs the serial fvm operators on its LOCAL mesh + field (correct internal + real-boundary coeffs;
// zero at processor patches) and adds the processor faces; parallelAmul reproduces the serial Amul.
// Run: mpirun -np {1,2,4,8} test_parallel_momentum <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "ldu_spmv.cuh"
#include "scotch_decomposition.cuh"
#include "local_mesh.cuh"
#include "local_assembly.cuh"
#include "field_distribute.cuh"
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
    const scalar nu = 1e-5;

    PrimitiveMesh gm;  gm.read(caseDir + "/constant/polyMesh");
    FvGeometry gg;     gg.build(gm);
    const std::vector<FvPatch> gpatches = buildPatches(gm, gg);
    const label nC = gm.nCells(), nIf = gm.nInternalFaces();

    const FieldData<vector> Ufd = readField<vector>(caseDir + "/282/U");
    GeometricField<vector> Ug = buildField<vector>(Ufd, gpatches, nC);  Ug.evaluateBoundary();
    const SurfaceScalarField phi = fvc::flux(Ug, gm, gg, gpatches);
    // Global momentum matrix + serial Amul of each component.
    FvVectorMatrix M = fvm::div(phi.internal, phi.boundary, Ug, gm, gpatches);
    addEqual(M, fvm::laplacian(Ug, nu, gm, gg, gpatches), -1.0);
    std::vector<scalar> Ux(nC); for (label c = 0; c < nC; ++c) Ux[c] = Ug.internal[c].x;
    FvScalarMatrix Ms; Ms.diag = M.diag; Ms.upper = M.upper; Ms.lower = M.lower;   // shared scalar lduMatrix
    const std::vector<scalar> AserialX = Amul(Ms, Ux, gm);

    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    LocalMesh Lm = buildLocalMesh(gm, cellToPart, rank);
    FvGeometry lg; lg.build(Lm.mesh);
    const std::vector<FvPatch> lpatches = buildPatches(Lm.mesh, lg);
    const auto procDelta = computeProcDeltaCoeffs(Lm, lg, lpatches);
    const auto procW     = computeProcWeights(Lm, lg, lpatches);

    GeometricField<vector> Ul = distributeField<vector>(Ufd, gm.patches(), Lm, lpatches, procW, rank);  Ul.evaluateBoundary();

    // Local phi: SurfaceScalarField (for the serial local fvm) + per-face outward flux (for the processor add).
    std::vector<scalar> gBndPhi(gm.nFaces(), 0.0);
    for (std::size_t gp = 0; gp < gpatches.size(); ++gp)
        for (label i = 0; i < gpatches[gp].size; ++i) gBndPhi[gpatches[gp].start + i] = phi.boundary[gp][i];
    SurfaceScalarField lphi; lphi.internal.resize(Lm.mesh.nInternalFaces()); lphi.boundary.resize(lpatches.size());
    std::vector<scalar> outwardPhi(Lm.mesh.nFaces(), 0.0);
    for (label f = 0; f < Lm.mesh.nFaces(); ++f) { const label gf = Lm.faceGlobal[f]; if (gf < nIf) outwardPhi[f] = phi.internal[gf] * (Lm.faceFlip[f] ? -1.0 : 1.0); }
    for (label f = 0; f < Lm.mesh.nInternalFaces(); ++f) lphi.internal[f] = outwardPhi[f];
    for (std::size_t pi = 0; pi < lpatches.size(); ++pi) {
        lphi.boundary[pi].resize(lpatches[pi].size);
        for (label i = 0; i < lpatches[pi].size; ++i) { const label gf = Lm.faceGlobal[lpatches[pi].start + i]; lphi.boundary[pi][i] = (gf < nIf) ? 0.0 : gBndPhi[gf]; }
    }

    FvVectorMatrix Ml = fvm::div(lphi.internal, lphi.boundary, Ul, Lm.mesh, lpatches);
    addEqual(Ml, fvm::laplacian(Ul, nu, Lm.mesh, lg, lpatches), -1.0);
    const std::vector<scalar> localNuEffF(Lm.mesh.nFaces(), nu);
    DistributedMatrix L = momentumDistributed(Ml, Lm, lg, lpatches, outwardPhi, localNuEffF, procDelta);

    std::vector<ProcessorInterface> interfaces;
    for (std::size_t j = 0; j < Lm.procNbr.size(); ++j) interfaces.emplace_back(rank, Lm.procNbr[j], Lm.procFaceCells[j]);
    std::vector<scalar> psiL(Lm.mesh.nCells());
    for (label lc = 0; lc < Lm.mesh.nCells(); ++lc) psiL[lc] = Ux[Lm.cellProcAddr[lc]];
    const std::vector<scalar> AlocalX = parallelAmul(L, psiL, interfaces);

    scalar maxAbs = 0, mag = 0;
    for (label lc = 0; lc < Lm.mesh.nCells(); ++lc) { const label gc = Lm.cellProcAddr[lc]; maxAbs = std::fmax(maxAbs, std::fabs(AlocalX[lc] - AserialX[gc])); mag = std::fmax(mag, std::fabs(AserialX[gc])); }
    const scalar gMax = Pstream::allReduce(maxAbs, ReduceOp::Max), gMag = Pstream::allReduce(mag, ReduceOp::Max);

    if (Pstream::master()) {
        const scalar rel = gMag > 0 ? gMax / gMag : gMax;
        std::printf("test_parallel_momentum np=%d: local momentum matrix Amul vs serial rel=%.3e\n", nproc, rel);
        std::printf("%s\n", rel < 1e-12 ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return 0;
}
