// lduInterface increment 4d: the parallel MOMENTUM PREDICTOR. solve(UEqn == -grad(p)) with the
// simplified laminar momentum div(phi,U)-laplacian(nu,U): relax, then per-component parallel solve.
// The distributed result matches the serial momentum predictor.
// Run: mpirun -np {1,2,4,8} test_parallel_predictor <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"
#include "solve_vector.cuh"
#include "scotch_decomposition.cuh"
#include "local_mesh.cuh"
#include "local_assembly.cuh"
#include "field_distribute.cuh"
#include "parallel_amul.cuh"
#include "parallel_pbicgstab.cuh"
#include "parallel_matrix_ops.cuh"
#include "cf_pstream.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

static scalar comp(const vector& v, int c) { return c == 0 ? v.x : (c == 1 ? v.y : v.z); }

int main(int argc, char** argv) {
    Pstream::init(argc, argv);
    const int rank = Pstream::myProcNo(), nproc = Pstream::nProcs();
    if (argc < 2) { if (Pstream::master()) std::printf("usage: %s <caseDir>\n", argv[0]); Pstream::finalize(); return 2; }
    const std::string caseDir = argv[1];
    const scalar nu = 1e-5, relaxU = 0.7, tol = 1e-9;

    PrimitiveMesh gm;  gm.read(caseDir + "/constant/polyMesh");
    FvGeometry gg;     gg.build(gm);
    const std::vector<FvPatch> gpatches = buildPatches(gm, gg);
    const label nC = gm.nCells(), nIf = gm.nInternalFaces();

    const FieldData<vector> Ufd = readField<vector>(caseDir + "/282/U");
    GeometricField<vector> Ug = buildField<vector>(Ufd, gpatches, nC);  Ug.evaluateBoundary();
    GeometricField<scalar> pg = buildField<scalar>(readField<scalar>(caseDir + "/282/p"), gpatches, nC);  pg.evaluateBoundary();
    const SurfaceScalarField phi = fvc::flux(Ug, gm, gg, gpatches);

    // ---- serial predictor ----
    FvVectorMatrix UEqn = fvm::div(phi.internal, phi.boundary, Ug, gm, gpatches);
    addEqual(UEqn, fvm::laplacian(Ug, nu, gm, gg, gpatches), -1.0);
    relaxMatrix(UEqn, Ug, gm, gpatches, relaxU);
    const std::vector<vector> gradP = fvc::gaussGrad(pg, gm, gg, gpatches);
    FvVectorMatrix Mp = UEqn; for (label c = 0; c < nC; ++c) Mp.source[c] += (-gg.V()[c]) * gradP[c];
    GeometricField<vector> Userial = buildField<vector>(Ufd, gpatches, nC); Userial.evaluateBoundary();
    solveVector(Mp, Userial, gm, gpatches, tol, 0.0, 2000);

    // ---- distributed predictor ----
    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    LocalMesh Lm = buildLocalMesh(gm, cellToPart, rank);
    FvGeometry lg; lg.build(Lm.mesh);
    const std::vector<FvPatch> lpatches = buildPatches(Lm.mesh, lg);
    const auto procDelta = computeProcDeltaCoeffs(Lm, lg, lpatches);
    const auto procW     = computeProcWeights(Lm, lg, lpatches);
    GeometricField<vector> Ul = distributeField<vector>(Ufd, gm.patches(), Lm, lpatches, procW, rank);  Ul.evaluateBoundary();
    GeometricField<scalar> pl = distributeField<scalar>(readField<scalar>(caseDir + "/282/p"), gm.patches(), Lm, lpatches, procW, rank);  pl.evaluateBoundary();

    // local phi
    std::vector<scalar> gBndPhi(gm.nFaces(), 0.0);
    for (std::size_t gp = 0; gp < gpatches.size(); ++gp) for (label i = 0; i < gpatches[gp].size; ++i) gBndPhi[gpatches[gp].start + i] = phi.boundary[gp][i];
    SurfaceScalarField lphi; lphi.internal.resize(Lm.mesh.nInternalFaces()); lphi.boundary.resize(lpatches.size());
    std::vector<scalar> outwardPhi(Lm.mesh.nFaces(), 0.0);
    for (label f = 0; f < Lm.mesh.nFaces(); ++f) { const label gf = Lm.faceGlobal[f]; if (gf < nIf) outwardPhi[f] = phi.internal[gf]*(Lm.faceFlip[f]?-1.0:1.0); }
    for (label f = 0; f < Lm.mesh.nInternalFaces(); ++f) lphi.internal[f] = outwardPhi[f];
    for (std::size_t pi = 0; pi < lpatches.size(); ++pi) { lphi.boundary[pi].resize(lpatches[pi].size); for (label i = 0; i < lpatches[pi].size; ++i) { const label gf = Lm.faceGlobal[lpatches[pi].start+i]; lphi.boundary[pi][i] = (gf < nIf) ? 0.0 : gBndPhi[gf]; } }

    FvVectorMatrix Ml = fvm::div(lphi.internal, lphi.boundary, Ul, Lm.mesh, lpatches);
    addEqual(Ml, fvm::laplacian(Ul, nu, Lm.mesh, lg, lpatches), -1.0);
    const std::vector<scalar> localNuEffF(Lm.mesh.nFaces(), nu);
    DistributedMatrix L = momentumDistributed(Ml, Lm, lg, lpatches, outwardPhi, localNuEffF, procDelta);
    parallelRelaxMatrix(L, Ml, Ul.internal, Lm, lpatches, relaxU);

    std::vector<ProcessorInterface> interfaces;
    for (std::size_t j = 0; j < Lm.procNbr.size(); ++j) interfaces.emplace_back(rank, Lm.procNbr[j], Lm.procFaceCells[j]);
    const std::vector<vector> gradPl = fvc::gaussGrad(pl, Lm.mesh, lg, lpatches);
    const label lnC = Lm.mesh.nCells();
    
    for (int k = 0; k < 3; ++k) {
        DistributedMatrix Lc = L;                                       // shared scalar matrix
        Lc.diagC = L.diag; Lc.b.assign(lnC, 0.0);
        for (label c = 0; c < lnC; ++c) Lc.b[c] = comp(Ml.source[c], k) - lg.V()[c] * comp(gradPl[c], k);
        for (std::size_t pi = 0; pi < lpatches.size(); ++pi)
            for (label i = 0; i < lpatches[pi].size; ++i) {
                Lc.diagC[lpatches[pi].faceCells[i]] += comp(Ml.internalCoeffs[pi][i], k);
                Lc.b[lpatches[pi].faceCells[i]]     += comp(Ml.boundaryCoeffs[pi][i], k);
            }
        std::vector<scalar> x(lnC); for (label c = 0; c < lnC; ++c) x[c] = comp(Ul.internal[c], k);
        parallelPBiCGStab(Lc, x, interfaces, nC, tol, 0.0, 2000);
        for (label c = 0; c < lnC; ++c) { if (k==0) Ul.internal[c].x = x[c]; else if (k==1) Ul.internal[c].y = x[c]; else Ul.internal[c].z = x[c]; }
    }

    scalar maxAbs = 0, mag = 0;
    for (label lc = 0; lc < lnC; ++lc) { const label gc = Lm.cellProcAddr[lc]; maxAbs = std::fmax(maxAbs, brae::mag(Ul.internal[lc] - Userial.internal[gc])); mag = std::fmax(mag, brae::mag(Userial.internal[gc])); }
    const scalar gMax = Pstream::allReduce(maxAbs, ReduceOp::Max), gMag = Pstream::allReduce(mag, ReduceOp::Max);
    if (Pstream::master()) {
        const scalar rel = gMag > 0 ? gMax / gMag : gMax;
        const scalar gate = (nproc == 1) ? 1e-8 : 1e-6;
        std::printf("test_parallel_predictor np=%d: momentum predictor U* vs serial rel=%.3e\n", nproc, rel);
        std::printf("%s\n", rel < gate ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return 0;
}
