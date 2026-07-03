// Phase 4 gate, cf GAMG vs OpenFOAM (dumpSolve with GAMG). Solve -laplacian(1,T)=0 and check
// the solution + convergence against OF's GAMG. The solution must match (both solve the same
// system to tolerance); iteration count is reported (agglomeration-dependent).
// Run: test_gamg <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "gamg.cuh"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <caseDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1];

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);

    GeometricField<scalar> T = buildField<scalar>(readField<scalar>(caseDir + "/282/T"), patches, m.nCells());
    T.evaluateBoundary();
    FvScalarMatrix M = fvm::laplacian(T, 1.0, m, g, patches);
    for (auto& v : M.diag)   v = -v;
    for (auto& v : M.upper)  v = -v;
    for (auto& v : M.lower)  v = -v;
    for (auto& v : M.source) v = -v;
    for (auto& p : M.internalCoeffs) for (auto& v : p) v = -v;
    for (auto& p : M.boundaryCoeffs) for (auto& v : p) v = -v;

    std::vector<scalar> psi = T.internal;
    const SolverPerformance perf =
        gamg(M, psi, m, g, patches, 1e-8, 0.0, 1000, /*nCellsInCoarsest*/10, 0, 2, 2);

    std::ifstream in(caseDir + "/solution.dat");
    if (!in) { std::printf("FAIL no solution.dat\n"); return 1; }
    label nC; in >> nC;
    scalar ofInit, ofFinal; int ofIters; in >> ofInit >> ofFinal >> ofIters;
    std::vector<scalar> Tof(nC);
    for (label c = 0; c < nC; ++c) in >> Tof[c];

    scalar maxAbs = 0.0, maxMag = 0.0;
    for (label c = 0; c < nC; ++c) {
        maxAbs = std::fmax(maxAbs, std::fabs(psi[c] - Tof[c]));
        maxMag = std::fmax(maxMag, std::fabs(Tof[c]));
    }
    const scalar solRel = maxAbs / maxMag;

    int fails = 0;
    if (!(perf.finalResidual < 1e-8)) { ++fails; std::printf("  FAIL did not converge (final=%.3e)\n", perf.finalResidual); }
    if (!(solRel < 1e-6))             { ++fails; std::printf("  FAIL solution relErr=%.3e\n", solRel); }

    std::printf("test_gamg: cf[init=%.4e final=%.4e iters=%d]  OF[init=%.4e final=%.4e iters=%d]\n",
                perf.initialResidual, perf.finalResidual, perf.nIterations, ofInit, ofFinal, ofIters);
    std::printf("  solution relErr=%.3e (max|T|=%.4f)\n", solRel, maxMag);
    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
