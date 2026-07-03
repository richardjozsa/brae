// Phase 5 gate, cf vector momentum matrix div(phi,U)-laplacian(nu,U) vs OpenFOAM
// (dumpMomentum, momentum.dat). Compares scalar diag/upper/lower and vector source +
// per-patch internal/boundary coeffs cell-by-cell.  Run: test_momentum_matrix <caseDir>
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
static void cmpS(const std::vector<scalar>& a, const std::vector<scalar>& of, const char* nm) {
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < of.size(); ++i) { mx = std::fmax(mx, std::fabs(a[i] - of[i])); mg = std::fmax(mg, std::fabs(of[i])); }
    const scalar rel = mg > 0 ? mx / mg : mx; if (rel > 1e-11) ++g_fails;
    std::printf("  %-18s n=%6zu rel=%.3e %s\n", nm, of.size(), rel, rel <= 1e-11 ? "OK" : "FAIL");
}
static void cmpV(const std::vector<vector>& a, const std::vector<vector>& of, const char* nm) {
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < of.size(); ++i) { mx = std::fmax(mx, mag(a[i] - of[i])); mg = std::fmax(mg, mag(of[i])); }
    const scalar rel = mg > 0 ? mx / mg : mx; if (rel > 1e-11) ++g_fails;
    std::printf("  %-18s n=%6zu rel=%.3e %s\n", nm, of.size(), rel, rel <= 1e-11 ? "OK" : "FAIL");
}

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

    FvVectorMatrix M = fvm::div(phiF.internalField, phiBnd, U, m, patches);
    addEqual(M, fvm::laplacian(U, 1e-5, m, g, patches), -1.0);

    std::ifstream in(caseDir + "/momentum.dat");
    if (!in) { std::printf("FAIL no momentum.dat\n"); return 1; }
    label nC, nIf, np; in >> nC >> nIf >> np;
    std::vector<scalar> diag(nC), upper(nIf), lower(nIf);
    std::vector<vector> source(nC);
    for (auto& v : diag)  in >> v;
    for (auto& v : upper) in >> v;
    for (auto& v : lower) in >> v;
    for (auto& v : source) in >> v.x >> v.y >> v.z;

    std::printf("test_momentum_matrix:\n");
    cmpS(M.diag, diag, "diag"); cmpS(M.upper, upper, "upper"); cmpS(M.lower, lower, "lower");
    cmpV(M.source, source, "source");

    for (label p = 0; p < np; ++p) {
        std::string name; label sz; in >> name >> sz;
        std::vector<vector> iC(sz), bC(sz);
        for (label i = 0; i < sz; ++i) in >> iC[i].x >> iC[i].y >> iC[i].z >> bC[i].x >> bC[i].y >> bC[i].z;
        if (sz == 0) continue;
        std::size_t pk = patches.size();
        for (std::size_t k = 0; k < patches.size(); ++k) if (patches[k].name == name) { pk = k; break; }
        cmpV(M.internalCoeffs[pk], iC, (name + ":intCoeffs").c_str());
        cmpV(M.boundaryCoeffs[pk], bC, (name + ":bndCoeffs").c_str());
    }
    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
