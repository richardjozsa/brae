// lduInterface increment 4d: parallel divDevReff fvc::div(nuEff*dev2(T(grad U))) == serial. The
// processor faces use the interpolated cell sigma; real faces the snGrad-corrected boundary gradient.
// Run: mpirun -np {1,2,4,8} test_parallel_divdevreff <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "scotch_decomposition.cuh"
#include "parallel_simple.cuh"
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
    const std::vector<FvPatch> gp = buildPatches(gm, gg);
    const label nC = gm.nCells();
    const FieldData<vector> Ufd = readField<vector>(caseDir + "/282/U");
    const FieldData<scalar> nfd = readField<scalar>(caseDir + "/282/nut");

    // ---- serial divDevReff ----
    std::vector<vector> divSer;
    {
        GeometricField<vector> U = buildField<vector>(Ufd, gp, nC);  U.evaluateBoundary();
        GeometricField<scalar> nut = buildField<scalar>(nfd, gp, nC);
        std::vector<scalar> nuEffC(nC); for (label c = 0; c < nC; ++c) nuEffC[c] = nu + nut.internal[c];
        std::vector<std::vector<scalar>> nuEffB(gp.size());
        for (std::size_t pi = 0; pi < gp.size(); ++pi) { const auto& nv = nut.boundary[pi]->value(); nuEffB[pi].resize(gp[pi].size); for (label i = 0; i < gp[pi].size; ++i) nuEffB[pi][i] = nu + nv[i]; }
        const std::vector<tensor> gradC = fvc::gaussGrad(U, gm, gg, gp);
        const auto gradB = fvc::gradUBoundary(U, gradC, gm, gg, gp);
        std::vector<tensor> sigC(nC); for (label c = 0; c < nC; ++c) sigC[c] = nuEffC[c] * dev2(transpose(gradC[c]));
        std::vector<std::vector<tensor>> sigB(gp.size());
        for (std::size_t pi = 0; pi < gp.size(); ++pi) { sigB[pi].resize(gp[pi].size); for (label i = 0; i < gp[pi].size; ++i) sigB[pi][i] = nuEffB[pi][i] * dev2(transpose(gradB[pi][i])); }
        divSer = fvc::div(sigC, sigB, gm, gg, gp);
    }

    // ---- parallel divDevReff ----
    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    Partition P(gm, cellToPart, rank);
    GeometricField<vector> U = distributeField<vector>(Ufd, gm.patches(), P.Lm, P.lp, P.procW, rank);  U.evaluateBoundary();
    GeometricField<scalar> nut = distributeField<scalar>(nfd, gm.patches(), P.Lm, P.lp, P.procW, rank);  nut.evaluateBoundary();
    std::vector<scalar> nuEffC(P.nCells()); for (label c = 0; c < P.nCells(); ++c) nuEffC[c] = nu + nut.internal[c];
    std::vector<std::vector<scalar>> nuEffB(P.lp.size());
    for (std::size_t pi = 0; pi < P.lp.size(); ++pi) { const auto& nv = nut.boundary[pi]->value(); nuEffB[pi].resize(P.lp[pi].size); for (label i = 0; i < P.lp[pi].size; ++i) nuEffB[pi][i] = nu + nv[i]; }
    const std::vector<vector> divLoc = parallelDivDevReff(P, U, nuEffC, nuEffB);

    scalar maxAbs = 0, mag = 0;
    for (label lc = 0; lc < P.nCells(); ++lc) { const label gc = P.Lm.cellProcAddr[lc]; maxAbs = std::fmax(maxAbs, brae::mag(divLoc[lc] - divSer[gc])); mag = std::fmax(mag, brae::mag(divSer[gc])); }
    const scalar rel = Pstream::allReduce(maxAbs, ReduceOp::Max) / Pstream::allReduce(mag, ReduceOp::Max);
    if (Pstream::master()) {
        std::printf("test_parallel_divdevreff np=%d: divDevReff vs serial rel=%.3e\n", nproc, rel);
        std::printf("%s\n", rel < 1e-10 ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return 0;
}
