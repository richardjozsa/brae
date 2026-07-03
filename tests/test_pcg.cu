// Phase 4 gate, cf PCG (DIC) vs OpenFOAM (dumpSolve). Solve -laplacian(1,T)=0 (PD problem
// set by T's BCs) and compare the solution, iteration count and residuals to OF's DICPCG.
// Run: test_pcg <caseDir>   (caseDir must contain solution.dat from dumpSolve)
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "pcg.cuh"

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

    // Solve -laplacian (positive-definite), as OF's dumpSolve does.
    for (auto& v : M.diag)   v = -v;
    for (auto& v : M.upper)  v = -v;
    for (auto& v : M.lower)  v = -v;
    for (auto& v : M.source) v = -v;
    for (auto& p : M.internalCoeffs) for (auto& v : p) v = -v;
    for (auto& p : M.boundaryCoeffs) for (auto& v : p) v = -v;

    std::vector<scalar> psi = T.internal;   // initial guess = field value (uniform 0)
    const SolverPerformance perf = pcg(M, psi, m, patches, 1e-8, 0.0, 1000);

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
    auto gate = [&](bool ok, const char* msg) { if (!ok) { ++fails; std::printf("  FAIL %s\n", msg); } };
    gate(perf.nIterations == ofIters, "iteration count");
    gate(std::fabs(perf.initialResidual - ofInit) / ofInit < 1e-9, "initial residual");
    gate(solRel < 1e-6, "solution");

    std::printf("test_pcg: cf[init=%.6e final=%.6e iters=%d]  OF[init=%.6e final=%.6e iters=%d]\n",
                perf.initialResidual, perf.finalResidual, perf.nIterations, ofInit, ofFinal, ofIters);
    std::printf("  solution relErr=%.3e (max|T|=%.4f)\n", solRel, maxMag);
    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
