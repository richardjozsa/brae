// Phase 3 gate, fvm::laplacian matrix vs OpenFOAM (dumpLaplacian app).
// Assembles fvm::laplacian(1, T) on the matrixDump case and compares diag/upper/lower/source
// and per-patch internalCoeffs/boundaryCoeffs to OF's dumped lduMatrix, cell-by-cell.
// Run: test_fvm_laplacian <caseDir>   (caseDir must contain matrix.dat from dumpLaplacian)
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

static int g_fails = 0;
static scalar relOf(const std::vector<scalar>& cf, const std::vector<scalar>& of, const char* nm) {
    scalar maxAbs = 0.0, maxMag = 0.0;
    for (std::size_t i = 0; i < of.size(); ++i) {
        maxAbs = std::fmax(maxAbs, std::fabs(cf[i] - of[i]));
        maxMag = std::fmax(maxMag, std::fabs(of[i]));
    }
    const scalar rel = (maxMag > 0) ? maxAbs / maxMag : maxAbs;
    const bool ok = rel <= 1e-12;
    if (!ok) ++g_fails;
    std::printf("  %-16s n=%6zu  maxAbs=%.3e  rel=%.3e  %s\n", nm, of.size(), maxAbs, rel, ok ? "OK" : "FAIL");
    return rel;
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <caseDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);

    GeometricField<scalar> T = buildField<scalar>(readField<scalar>(caseDir + "/282/T"), patches, m.nCells());
    T.evaluateBoundary();
    const FvScalarMatrix M = fvm::laplacian(T, 1.0, m, g, patches);

    std::ifstream in(caseDir + "/matrix.dat");
    if (!in) { std::printf("FAIL cannot open %s/matrix.dat (run dumpLaplacian)\n", caseDir.c_str()); return 1; }
    label nC, nIf, np;
    in >> nC >> nIf >> np;
    if (nC != m.nCells() || nIf != m.nInternalFaces()) { std::printf("FAIL dimension mismatch\n"); return 1; }

    std::vector<scalar> diag(nC), upper(nIf), lower(nIf), source(nC);
    for (auto& v : diag)   in >> v;
    for (auto& v : upper)  in >> v;
    for (auto& v : lower)  in >> v;
    for (auto& v : source) in >> v;

    std::printf("test_fvm_laplacian (matrixDump, nCells=%d):\n", (int)nC);
    relOf(M.diag,   diag,   "diag");
    relOf(M.upper,  upper,  "upper");
    relOf(M.lower,  lower,  "lower");
    relOf(M.source, source, "source");

    for (label p = 0; p < np; ++p) {
        std::string name;
        label sz;
        in >> name >> sz;
        std::vector<scalar> iC(sz), bC(sz);
        for (label i = 0; i < sz; ++i) in >> iC[i] >> bC[i];
        if (sz == 0) continue;
        std::size_t pk = patches.size();
        for (std::size_t k = 0; k < patches.size(); ++k) if (patches[k].name == name) { pk = k; break; }
        if (pk == patches.size()) { std::printf("  FAIL patch %s not found\n", name.c_str()); ++g_fails; continue; }
        relOf(M.internalCoeffs[pk], iC, (name + ":intCoeffs").c_str());
        relOf(M.boundaryCoeffs[pk], bC, (name + ":bndCoeffs").c_str());
    }

    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
