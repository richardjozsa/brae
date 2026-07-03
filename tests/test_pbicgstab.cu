// Phase 5 gate, cf PBiCGStab (DILU) vs OpenFOAM. Solve the asymmetric scalar convection-
// diffusion matrix div(phi,T)-laplacian(1,T) and compare solution + convergence to OF's
// DILUPBiCGStab.  Run: test_pbicgstab <caseDir>  (caseDir has transport.dat from dumpMomentum)
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "pbicgstab.cuh"

#include <algorithm>
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

    const FieldData<scalar> phiF = readField<scalar>(caseDir + "/282/phi");
    std::vector<std::vector<scalar>> phiBnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) {
        phiBnd[pi].assign(patches[pi].size, 0.0);
        for (const auto& b : phiF.boundary)
            if (b.name == patches[pi].name && b.hasValue && (label)b.values.size() == patches[pi].size) phiBnd[pi] = b.values;
    }

    FvScalarMatrix M = fvm::div(phiF.internalField, phiBnd, T, m, patches);
    addEqual(M, fvm::laplacian(T, 1.0, m, g, patches), -1.0);   // div(phi,T) - laplacian(1,T)

    std::vector<scalar> psi = T.internal;
    const SolverPerformance perf = pbicgstab(M, psi, m, patches, 1e-8, 0.0, 1000);

    std::ifstream in(caseDir + "/transport.dat");
    if (!in) { std::printf("FAIL no transport.dat\n"); return 1; }
    label nC; in >> nC;
    scalar ofInit, ofFinal; int ofIters; in >> ofInit >> ofFinal >> ofIters;
    std::vector<scalar> Tof(nC); for (label c = 0; c < nC; ++c) in >> Tof[c];

    scalar maxAbs = 0.0, maxMag = 0.0;
    for (label c = 0; c < nC; ++c) { maxAbs = std::fmax(maxAbs, std::fabs(psi[c] - Tof[c])); maxMag = std::fmax(maxMag, std::fabs(Tof[c])); }
    const scalar solRel = maxAbs / maxMag;

    int fails = 0;
    if (!(perf.finalResidual < 1e-8)) { ++fails; std::printf("  FAIL not converged (%.3e)\n", perf.finalResidual); }
    if (!(solRel < 1e-6))             { ++fails; std::printf("  FAIL solution relErr=%.3e\n", solRel); }

    std::printf("test_pbicgstab: cf[init=%.4e final=%.4e iters=%d]  OF[init=%.4e final=%.4e iters=%d]  solRel=%.3e\n",
                perf.initialResidual, perf.finalResidual, perf.nIterations, ofInit, ofFinal, ofIters, solRel);
    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
