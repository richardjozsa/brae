// Phase 6 gate, k-epsilon transport matrices (div - laplacian + Sp == production) vs OpenFOAM
// (dumpKEpsilonEqn keqn.dat/epseqn.dat). Validates fvm::Sp + the equation assembly.  Run: <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "k_epsilon.cuh"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

static int g_fails = 0;
static void cmpMatrix(const FvScalarMatrix& M, const std::string& path,
                      const PrimitiveMesh& m, const std::vector<FvPatch>& patches, const char* tag) {
    std::ifstream in(path);
    if (!in) { std::printf("  FAIL no %s\n", path.c_str()); ++g_fails; return; }
    label nC, nIf, np; in >> nC >> nIf >> np;
    std::vector<scalar> diag(nC), upper(nIf), lower(nIf), source(nC);
    for (auto& v : diag)  in >> v;  for (auto& v : upper) in >> v;
    for (auto& v : lower) in >> v;  for (auto& v : source) in >> v;
    auto rel = [&](const std::vector<scalar>& a, const std::vector<scalar>& b) {
        scalar mx = 0, mg = 0; for (std::size_t i = 0; i < b.size(); ++i) { mx = std::fmax(mx, std::fabs(a[i]-b[i])); mg = std::fmax(mg, std::fabs(b[i])); }
        return mg > 0 ? mx / mg : mx;
    };
    const scalar rD = rel(M.diag, diag), rU = rel(M.upper, upper), rL = rel(M.lower, lower), rS = rel(M.source, source);
    scalar rC = 0;
    for (label p = 0; p < np; ++p) {
        std::string name; label sz; in >> name >> sz;
        std::vector<scalar> iC(sz), bC(sz);
        for (label i = 0; i < sz; ++i) in >> iC[i] >> bC[i];
        if (sz == 0) continue;
        std::size_t pk = patches.size();
        for (std::size_t kk = 0; kk < patches.size(); ++kk) if (patches[kk].name == name) { pk = kk; break; }
        rC = std::fmax(rC, std::fmax(rel(M.internalCoeffs[pk], iC), rel(M.boundaryCoeffs[pk], bC)));
    }
    const scalar worst = std::fmax(std::fmax(rD, rU), std::fmax(std::fmax(rL, rS), rC));
    const bool ok = worst <= 1e-11;
    if (!ok) ++g_fails;
    std::printf("  %-7s diag=%.2e upper=%.2e lower=%.2e source=%.2e coeffs=%.2e  %s\n",
                tag, rD, rU, rL, rS, rC, ok ? "OK" : "FAIL");
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <caseDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1];

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);

    GeometricField<vector> U = buildField<vector>(readField<vector>(caseDir + "/282/U"), patches, m.nCells());  U.evaluateBoundary();
    GeometricField<scalar> k = buildField<scalar>(readField<scalar>(caseDir + "/282/k"), patches, m.nCells());  k.evaluateBoundary();
    GeometricField<scalar> eps = buildField<scalar>(readField<scalar>(caseDir + "/282/epsilon"), patches, m.nCells());  eps.evaluateBoundary();
    const std::vector<scalar> nut = readField<scalar>(caseDir + "/282/nut").internalField;
    const FieldData<scalar> phiF = readField<scalar>(caseDir + "/282/phi");
    SurfaceScalarField phi; phi.internal = phiF.internalField; phi.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) {
        phi.boundary[pi].assign(patches[pi].size, 0.0);
        for (const auto& b : phiF.boundary)
            if (b.name == patches[pi].name && b.hasValue && (label)b.values.size() == patches[pi].size) phi.boundary[pi] = b.values;
    }

    const std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
    const std::vector<scalar> GbyNu = kepsilon::GbyNu(gradU);
    const std::vector<scalar> G     = kepsilon::production(nut, GbyNu);
    const scalar nu = 1e-5;

    const FvScalarMatrix kEqn   = kepsilon::kEqn(k, eps.internal, G, nut, phi, nu, m, g, patches);
    const FvScalarMatrix epsEqn = kepsilon::epsilonEqn(eps, k.internal, GbyNu, nut, phi, nu, m, g, patches);

    std::printf("test_kepsilon_eqn:\n");
    cmpMatrix(kEqn,   caseDir + "/keqn.dat",   m, patches, "kEqn");
    cmpMatrix(epsEqn, caseDir + "/epseqn.dat", m, patches, "epsEqn");
    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
