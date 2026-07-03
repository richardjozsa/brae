// Phase 3 gate, SpMV (lduMatrix::Amul) vs OpenFOAM (dumpDiv app, amul.dat).
// Assemble fvm::laplacian(1,T), apply to the same psi OF used, compare A*psi cell-by-cell.
// Run: test_spmv <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "ldu_spmv.cuh"

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
    const FvScalarMatrix M = fvm::laplacian(T, 1.0, m, g, patches);

    std::ifstream in(caseDir + "/amul.dat");
    if (!in) { std::printf("FAIL no amul.dat\n"); return 1; }
    label nC; in >> nC;
    if (nC != m.nCells()) { std::printf("FAIL size\n"); return 1; }
    std::vector<scalar> psi(nC), ApsiOf(nC);
    for (label c = 0; c < nC; ++c) in >> psi[c] >> ApsiOf[c];

    const std::vector<scalar> Apsi = Amul(M, psi, m);

    scalar maxAbs = 0.0, maxMag = 0.0;
    for (label c = 0; c < nC; ++c) {
        maxAbs = std::fmax(maxAbs, std::fabs(Apsi[c] - ApsiOf[c]));
        maxMag = std::fmax(maxMag, std::fabs(ApsiOf[c]));
    }
    const scalar rel = maxAbs / maxMag;
    const bool ok = rel <= 1e-12;
    std::printf("test_spmv: nCells=%d  max|A.psi|=%.4e  maxAbs=%.3e  rel=%.3e (tol 1e-12)  %s\n",
                (int)nC, maxMag, maxAbs, rel, ok ? "OK" : "FAIL");
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
