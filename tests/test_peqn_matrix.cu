// Phase 5 gate, pressure equation matrix laplacian(rAU,p) vs OpenFOAM (peqn.dat).
// rAU = 1/UEqn.A(); gammaf = interpolate(rAU); pEqn = laplacian(gammaf, p).  Run: <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

static int g_fails = 0;
static void cmp(const std::vector<scalar>& a, const std::vector<scalar>& of, const char* nm) {
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < of.size(); ++i) { mx = std::fmax(mx, std::fabs(a[i] - of[i])); mg = std::fmax(mg, std::fabs(of[i])); }
    const scalar rel = mg > 0 ? mx / mg : mx; if (rel > 1e-11) ++g_fails;
    std::printf("  %-16s n=%6zu rel=%.3e %s\n", nm, of.size(), rel, rel <= 1e-11 ? "OK" : "FAIL");
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <caseDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1];

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);

    GeometricField<vector> U = buildField<vector>(readField<vector>(caseDir + "/282/U"), patches, m.nCells());
    U.evaluateBoundary();
    GeometricField<scalar> p = buildField<scalar>(readField<scalar>(caseDir + "/282/p"), patches, m.nCells());
    p.evaluateBoundary();
    const FieldData<scalar> phiF = readField<scalar>(caseDir + "/282/phi");
    std::vector<std::vector<scalar>> phiBnd(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) {
        phiBnd[pi].assign(patches[pi].size, 0.0);
        for (const auto& b : phiF.boundary)
            if (b.name == patches[pi].name && b.hasValue && (label)b.values.size() == patches[pi].size) phiBnd[pi] = b.values;
    }

    FvVectorMatrix UEqn = fvm::div(phiF.internalField, phiBnd, U, m, patches);
    addEqual(UEqn, fvm::laplacian(U, 1e-5, m, g, patches), -1.0);

    const std::vector<scalar> A = matrixA(UEqn, m, g, patches);
    std::vector<scalar> rAU(m.nCells());
    for (label c = 0; c < m.nCells(); ++c) rAU[c] = 1.0 / A[c];
    const SurfaceScalarField gammaf = fvc::interpolate(rAU, m, g, patches);
    const FvScalarMatrix pEqn = fvm::laplacian(gammaf, p, m, g, patches);

    std::ifstream in(caseDir + "/peqn.dat");
    if (!in) { std::printf("FAIL no peqn.dat\n"); return 1; }
    label nC, nIf, np; in >> nC >> nIf >> np;
    std::vector<scalar> diag(nC), upper(nIf), lower(nIf), source(nC);
    for (auto& v : diag)  in >> v;
    for (auto& v : upper) in >> v;
    for (auto& v : lower) in >> v;
    for (auto& v : source) in >> v;

    std::printf("test_peqn_matrix:\n");
    cmp(pEqn.diag, diag, "diag"); cmp(pEqn.upper, upper, "upper"); cmp(pEqn.lower, lower, "lower"); cmp(pEqn.source, source, "source");
    for (label pp = 0; pp < np; ++pp) {
        std::string name; label sz; in >> name >> sz;
        std::vector<scalar> iC(sz), bC(sz);
        for (label i = 0; i < sz; ++i) in >> iC[i] >> bC[i];
        if (sz == 0) continue;
        std::size_t pk = patches.size();
        for (std::size_t k = 0; k < patches.size(); ++k) if (patches[k].name == name) { pk = k; break; }
        cmp(pEqn.internalCoeffs[pk], iC, (name + ":iC").c_str());
        cmp(pEqn.boundaryCoeffs[pk], bC, (name + ":bC").c_str());
    }
    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
