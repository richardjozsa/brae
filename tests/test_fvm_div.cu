// Phase 3 gate, fvm::div(phi,T) (Gauss upwind) vs OpenFOAM (dumpDiv app, matrix_div.dat).
// Run: test_fvm_div <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

static int g_fails = 0;
static void cmp(const std::vector<scalar>& a, const std::vector<scalar>& of, const char* nm) {
    scalar maxAbs = 0.0, maxMag = 0.0;
    for (std::size_t i = 0; i < of.size(); ++i) {
        maxAbs = std::fmax(maxAbs, std::fabs(a[i] - of[i]));
        maxMag = std::fmax(maxMag, std::fabs(of[i]));
    }
    const scalar rel = (maxMag > 0) ? maxAbs / maxMag : maxAbs;
    const bool ok = rel <= 1e-12;
    if (!ok) ++g_fails;
    std::printf("  %-18s n=%6zu  maxAbs=%.3e  rel=%.3e  %s\n", nm, of.size(), maxAbs, rel, ok ? "OK" : "FAIL");
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <caseDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1];

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);

    GeometricField<scalar> T = buildField<scalar>(readField<scalar>(caseDir + "/282/T"), patches, m.nCells());
    T.evaluateBoundary();

    // Face flux phi: internal + per-patch boundary (aligned to mesh patches).
    const FieldData<scalar> phiF = readField<scalar>(caseDir + "/282/phi");
    std::vector<std::vector<scalar>> phiBnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) {
        phiBnd[pi].assign(patches[pi].size, 0.0);
        for (const auto& b : phiF.boundary)
            if (b.name == patches[pi].name && b.hasValue) {
                if (b.valueUniform) std::fill(phiBnd[pi].begin(), phiBnd[pi].end(), b.uniformValue);
                else if ((label)b.values.size() == patches[pi].size) phiBnd[pi] = b.values;
            }
    }

    const FvScalarMatrix M = fvm::div(phiF.internalField, phiBnd, T, m, patches);

    std::ifstream in(caseDir + "/matrix_div.dat");
    if (!in) { std::printf("FAIL no matrix_div.dat\n"); return 1; }
    label nC, nIf, np;  in >> nC >> nIf >> np;
    std::vector<scalar> diag(nC), upper(nIf), lower(nIf), source(nC);
    for (auto& v : diag)   in >> v;
    for (auto& v : upper)  in >> v;
    for (auto& v : lower)  in >> v;
    for (auto& v : source) in >> v;

    std::printf("test_fvm_div (matrixDump, Gauss upwind):\n");
    cmp(M.diag, diag, "diag"); cmp(M.upper, upper, "upper"); cmp(M.lower, lower, "lower"); cmp(M.source, source, "source");
    for (label p = 0; p < np; ++p) {
        std::string name; label sz; in >> name >> sz;
        std::vector<scalar> iC(sz), bC(sz);
        for (label i = 0; i < sz; ++i) in >> iC[i] >> bC[i];
        if (sz == 0) continue;
        std::size_t pk = patches.size();
        for (std::size_t k = 0; k < patches.size(); ++k) if (patches[k].name == name) { pk = k; break; }
        cmp(M.internalCoeffs[pk], iC, (name + ":intCoeffs").c_str());
        cmp(M.boundaryCoeffs[pk], bC, (name + ":bndCoeffs").c_str());
    }
    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
