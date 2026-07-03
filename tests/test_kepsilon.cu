// Phase 6 gate, k-epsilon production + eddy viscosity vs OpenFOAM (dumpKEpsilon).
// grad(U), GbyNu = gradU && devTwoSymm(gradU), nut = Cmu*k^2/eps, G = nut*GbyNu.
// Run: test_kepsilon <caseDir>
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
static void cmp(const std::vector<scalar>& a, const std::vector<scalar>& of, const char* nm) {
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < of.size(); ++i) { mx = std::fmax(mx, std::fabs(a[i] - of[i])); mg = std::fmax(mg, std::fabs(of[i])); }
    const scalar rel = mg > 0 ? mx / mg : mx; if (rel > 1e-11) ++g_fails;
    std::printf("  %-10s n=%6zu rel=%.3e %s\n", nm, of.size(), rel, rel <= 1e-11 ? "OK" : "FAIL");
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <caseDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1];

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);

    GeometricField<vector> U = buildField<vector>(readField<vector>(caseDir + "/282/U"), patches, m.nCells());
    U.evaluateBoundary();
    const std::vector<scalar> k   = readField<scalar>(caseDir + "/282/k").internalField;
    const std::vector<scalar> eps = readField<scalar>(caseDir + "/282/epsilon").internalField;

    const std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
    const std::vector<scalar> GbyNu = kepsilon::GbyNu(gradU);
    const std::vector<scalar> nut   = kepsilon::nut(k, eps);
    const std::vector<scalar> G     = kepsilon::production(nut, GbyNu);

    std::ifstream in(caseDir + "/kepsilon.dat");
    if (!in) { std::printf("FAIL no kepsilon.dat\n"); return 1; }
    label nC; in >> nC;
    std::vector<scalar> gUof(9 * nC), gUcf(9 * nC), GbyNuOf(nC), nutOf(nC), Gof(nC);
    for (label c = 0; c < nC; ++c) {
        in >> gUof[9*c+0] >> gUof[9*c+1] >> gUof[9*c+2] >> gUof[9*c+3] >> gUof[9*c+4]
           >> gUof[9*c+5] >> gUof[9*c+6] >> gUof[9*c+7] >> gUof[9*c+8];
        const tensor& t = gradU[c];
        gUcf[9*c+0]=t.xx; gUcf[9*c+1]=t.xy; gUcf[9*c+2]=t.xz; gUcf[9*c+3]=t.yx; gUcf[9*c+4]=t.yy;
        gUcf[9*c+5]=t.yz; gUcf[9*c+6]=t.zx; gUcf[9*c+7]=t.zy; gUcf[9*c+8]=t.zz;
    }
    for (auto& v : GbyNuOf) in >> v;
    for (auto& v : nutOf)   in >> v;
    for (auto& v : Gof)     in >> v;

    std::printf("test_kepsilon:\n");
    cmp(gUcf, gUof, "grad(U)");
    cmp(GbyNu, GbyNuOf, "GbyNu");
    cmp(nut, nutOf, "nut");
    cmp(G, Gof, "G");
    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
