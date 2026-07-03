// lduInterface increment 4d: parallel kEpsilon::correct() == serial. One correct() step from 282,
// distributed, compared to the serial correct(). Run: mpirun -np {1,2,4,8} test_parallel_kepsilon <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "k_epsilon.cuh"
#include "scotch_decomposition.cuh"
#include "parallel_simple.cuh"
#include "parallel_kepsilon.cuh"
#include "cf_pstream.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

static SurfaceScalarField readPhi(const std::string& path, const std::vector<FvPatch>& patches) {
    const FieldData<scalar> phiF = readField<scalar>(path);
    SurfaceScalarField phi; phi.internal = phiF.internalField; phi.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) { phi.boundary[pi].assign(patches[pi].size, 0.0);
        for (const auto& b : phiF.boundary) if (b.name == patches[pi].name && b.hasValue && (label)b.values.size() == patches[pi].size) phi.boundary[pi] = b.values; }
    return phi;
}

int main(int argc, char** argv) {
    Pstream::init(argc, argv);
    const int rank = Pstream::myProcNo(), nproc = Pstream::nProcs();
    if (argc < 2) { if (Pstream::master()) std::printf("usage: %s <caseDir>\n", argv[0]); Pstream::finalize(); return 2; }
    const std::string caseDir = argv[1];
    const scalar nu = 1e-5, tol = 1e-10;

    PrimitiveMesh gm;  gm.read(caseDir + "/constant/polyMesh");
    FvGeometry gg;     gg.build(gm);
    const std::vector<FvPatch> gp = buildPatches(gm, gg);
    const label nC = gm.nCells(), nIf = gm.nInternalFaces();
    const FieldData<vector> Ufd = readField<vector>(caseDir + "/282/U");
    const FieldData<scalar> kfd = readField<scalar>(caseDir + "/282/k"), efd = readField<scalar>(caseDir + "/282/epsilon"), nfd = readField<scalar>(caseDir + "/282/nut");

    // ---- serial correct ----
    std::vector<scalar> kser, eser, nser;
    {
        GeometricField<vector> U = buildField<vector>(Ufd, gp, nC);  U.evaluateBoundary();
        GeometricField<scalar> k = buildField<scalar>(kfd, gp, nC);  k.evaluateBoundary();
        GeometricField<scalar> eps = buildField<scalar>(efd, gp, nC);  eps.evaluateBoundary();
        GeometricField<scalar> nut = buildField<scalar>(nfd, gp, nC);
        const SurfaceScalarField phi = readPhi(caseDir + "/282/phi", gp);
        kepsilon::correct(U, k, eps, nut, phi, nu, gm, gg, gp, 1.0, 1.0, tol, 0.0, 2000);
        kser = k.internal; eser = eps.internal; nser = nut.internal;
    }

    // ---- parallel correct ----
    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    Partition P(gm, cellToPart, rank);
    GeometricField<vector> U = distributeField<vector>(Ufd, gm.patches(), P.Lm, P.lp, P.procW, rank);  U.evaluateBoundary();
    GeometricField<scalar> k = distributeField<scalar>(kfd, gm.patches(), P.Lm, P.lp, P.procW, rank);  k.evaluateBoundary();
    GeometricField<scalar> eps = distributeField<scalar>(efd, gm.patches(), P.Lm, P.lp, P.procW, rank);  eps.evaluateBoundary();
    GeometricField<scalar> nut = distributeField<scalar>(nfd, gm.patches(), P.Lm, P.lp, P.procW, rank);  nut.evaluateBoundary();

    // local phi from the global conservative flux.
    const SurfaceScalarField gphi = readPhi(caseDir + "/282/phi", gp);
    std::vector<scalar> gBnd(gm.nFaces(), 0.0);
    for (std::size_t g2 = 0; g2 < gp.size(); ++g2) for (label i = 0; i < gp[g2].size; ++i) gBnd[gp[g2].start + i] = gphi.boundary[g2][i];
    SurfaceScalarField phi; phi.internal.resize(P.Lm.mesh.nInternalFaces()); phi.boundary.resize(P.lp.size());
    for (label f = 0; f < P.Lm.mesh.nInternalFaces(); ++f) { const label gf = P.Lm.faceGlobal[f]; phi.internal[f] = gphi.internal[gf] * (P.Lm.faceFlip[f] ? -1.0 : 1.0); }
    for (std::size_t pi = 0; pi < P.lp.size(); ++pi) { phi.boundary[pi].resize(P.lp[pi].size); for (label i = 0; i < P.lp[pi].size; ++i) { const label gf = P.Lm.faceGlobal[P.lp[pi].start + i]; phi.boundary[pi][i] = (gf < nIf) ? gphi.internal[gf] * (P.Lm.faceFlip[P.lp[pi].start + i] ? -1.0 : 1.0) : gBnd[gf]; } }

    kepsilon::parallelCorrect(P, U, k, eps, nut, phi, nu, 1.0, 1.0, tol);

    scalar mk = 0, gk = 0, me = 0, ge = 0, mn = 0, gn = 0;
    for (label lc = 0; lc < P.nCells(); ++lc) { const label gc = P.Lm.cellProcAddr[lc];
        mk = std::fmax(mk, std::fabs(k.internal[lc] - kser[gc])); gk = std::fmax(gk, std::fabs(kser[gc]));
        me = std::fmax(me, std::fabs(eps.internal[lc] - eser[gc])); ge = std::fmax(ge, std::fabs(eser[gc]));
        mn = std::fmax(mn, std::fabs(nut.internal[lc] - nser[gc])); gn = std::fmax(gn, std::fabs(nser[gc])); }
    const scalar rk = Pstream::allReduce(mk, ReduceOp::Max) / Pstream::allReduce(gk, ReduceOp::Max);
    const scalar re = Pstream::allReduce(me, ReduceOp::Max) / Pstream::allReduce(ge, ReduceOp::Max);
    const scalar rn = Pstream::allReduce(mn, ReduceOp::Max) / Pstream::allReduce(gn, ReduceOp::Max);
    if (Pstream::master()) {
        const scalar gate = (nproc == 1) ? 1e-7 : 5e-3;   // np>1: nearWallDist at split walls differs slightly
        std::printf("test_parallel_kepsilon np=%d: k rel=%.3e  eps rel=%.3e  nut rel=%.3e\n", nproc, rk, re, rn);
        std::printf("%s\n", (rk < gate && re < gate && rn < gate) ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return 0;
}
