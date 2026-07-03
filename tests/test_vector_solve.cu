// Phase 5 gate, segregated vector solve (component-wise PBiCGStab) vs OpenFOAM.
// Solve UEqn = div(phi,U)-laplacian(nu,U) == 0 and compare U to OF's solve.  Run: <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "solve_vector.cuh"

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

    GeometricField<vector> U = buildField<vector>(readField<vector>(caseDir + "/282/U"), patches, m.nCells());
    U.evaluateBoundary();
    const FieldData<scalar> phiF = readField<scalar>(caseDir + "/282/phi");
    std::vector<std::vector<scalar>> phiBnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) {
        phiBnd[pi].assign(patches[pi].size, 0.0);
        for (const auto& b : phiF.boundary)
            if (b.name == patches[pi].name && b.hasValue && (label)b.values.size() == patches[pi].size) phiBnd[pi] = b.values;
    }

    FvVectorMatrix UEqn = fvm::div(phiF.internalField, phiBnd, U, m, patches);
    addEqual(UEqn, fvm::laplacian(U, 1e-5, m, g, patches), -1.0);
    const SolverPerformance perf = solveVector(UEqn, U, m, patches, 1e-8, 0.0, 1000);

    std::ifstream in(caseDir + "/usolve.dat");
    if (!in) { std::printf("FAIL no usolve.dat\n"); return 1; }
    label nC; in >> nC;
    std::vector<vector> Uof(nC);
    for (label c = 0; c < nC; ++c) in >> Uof[c].x >> Uof[c].y >> Uof[c].z;

    scalar maxAbs = 0, maxMag = 0;
    for (label c = 0; c < nC; ++c) { maxAbs = std::fmax(maxAbs, mag(U.internal[c] - Uof[c])); maxMag = std::fmax(maxMag, mag(Uof[c])); }
    const scalar rel = maxAbs / maxMag;
    const bool ok = rel < 1e-6;
    std::printf("test_vector_solve: Ux iters=%d  max|U|=%.4f  solRel=%.3e  %s\n",
                perf.nIterations, maxMag, rel, ok ? "OK" : "FAIL");
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
