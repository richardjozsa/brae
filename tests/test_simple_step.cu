// Phase 5 gate, one cf SIMPLE step vs OpenFOAM (dumpSimpleStep). From step 282, do one
// identical SIMPLE iteration (relax, predictor, pEqn, flux, corrector) and compare p/U/phi.
// Run: test_simple_step <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "simple_foam.cuh"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

static int g_fails = 0;
static void cmp(const std::vector<scalar>& a, const std::vector<scalar>& of, const char* nm, scalar tol = 1e-5) {
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < of.size(); ++i) { mx = std::fmax(mx, std::fabs(a[i] - of[i])); mg = std::fmax(mg, std::fabs(of[i])); }
    const scalar rel = mg > 0 ? mx / mg : mx; if (rel > tol) ++g_fails;
    std::printf("  %-14s n=%6zu maxAbs=%.3e rel=%.3e %s\n", nm, of.size(), mx, rel, rel <= tol ? "OK" : "FAIL");
}

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <caseDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1];

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);

    const FieldData<vector> Udata = readField<vector>(caseDir + "/282/U");
    GeometricField<vector> U = buildField<vector>(Udata, patches, m.nCells());  U.evaluateBoundary();
    GeometricField<scalar> p = buildField<scalar>(readField<scalar>(caseDir + "/282/p"), patches, m.nCells());  p.evaluateBoundary();
    const FieldData<scalar> phiF = readField<scalar>(caseDir + "/282/phi");
    SurfaceScalarField phi; phi.internal = phiF.internalField; phi.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) {
        phi.boundary[pi].assign(patches[pi].size, 0.0);
        for (const auto& b : phiF.boundary)
            if (b.name == patches[pi].name && b.hasValue && (label)b.values.size() == patches[pi].size) phi.boundary[pi] = b.values;
    }

    SimpleControls ctl;  // nu 1e-5, relaxU 0.7, relaxP 0.3, tol 1e-10
    simpleStep(U, p, phi, Udata, m, g, patches, ctl);

    std::ifstream in(caseDir + "/step.dat");
    if (!in) { std::printf("FAIL no step.dat\n"); return 1; }
    label nC, nIf, np; in >> nC >> nIf >> np;
    std::vector<scalar> pof(nC), phiIntOf(nIf);
    std::vector<scalar> Uof(3 * nC), Ucf(3 * nC);
    for (auto& v : pof) in >> v;
    for (label c = 0; c < nC; ++c) { in >> Uof[3*c] >> Uof[3*c+1] >> Uof[3*c+2];
                                     Ucf[3*c] = U.internal[c].x; Ucf[3*c+1] = U.internal[c].y; Ucf[3*c+2] = U.internal[c].z; }
    for (auto& v : phiIntOf) in >> v;

    std::printf("test_simple_step:\n");
    cmp(p.internal, pof, "p");
    cmp(Ucf, Uof, "U");
    cmp(phi.internal, phiIntOf, "phi:internal");
    for (label pp = 0; pp < np; ++pp) {
        std::string name; label sz; in >> name >> sz;
        std::vector<scalar> fb(sz); for (label i = 0; i < sz; ++i) in >> fb[i];
        if (sz == 0) continue;
        std::size_t pk = patches.size();
        for (std::size_t k = 0; k < patches.size(); ++k) if (patches[k].name == name) { pk = k; break; }
        cmp(phi.boundary[pk], fb, (name + ":phi").c_str());
    }
    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
