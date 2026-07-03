// Phase 6 gate, cf kEpsilon correct() (wall functions + constraint + solve + bound) vs the real
// OpenFOAM kEpsilon model (dumpKEpsilonCorrect). One correct() from step 282; compare k/eps/nut.
// Run: test_kepsilon_correct <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "k_epsilon.cuh"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

static scalar relErr(const std::vector<scalar>& a, const std::vector<scalar>& of) {
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < of.size(); ++i) { mx = std::fmax(mx, std::fabs(a[i] - of[i])); mg = std::fmax(mg, std::fabs(of[i])); }
    return mg > 0 ? mx / mg : mx;
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
    GeometricField<scalar> nut = buildField<scalar>(readField<scalar>(caseDir + "/282/nut"), patches, m.nCells());
    const FieldData<scalar> phiF = readField<scalar>(caseDir + "/282/phi");
    SurfaceScalarField phi; phi.internal = phiF.internalField; phi.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) {
        phi.boundary[pi].assign(patches[pi].size, 0.0);
        for (const auto& b : phiF.boundary)
            if (b.name == patches[pi].name && b.hasValue && (label)b.values.size() == patches[pi].size) phi.boundary[pi] = b.values;
    }

    kepsilon::correct(U, k, eps, nut, phi, 1e-5, m, g, patches, /*relaxEps*/1.0, /*relaxK*/1.0, 1e-10, 0.0, 2000);

    std::ifstream in(caseDir + "/correct.dat");
    if (!in) { std::printf("FAIL no correct.dat\n"); return 1; }
    label nC; in >> nC;
    std::vector<scalar> kof(nC), eof(nC), nof(nC);
    for (auto& v : kof) in >> v;  for (auto& v : eof) in >> v;  for (auto& v : nof) in >> v;

    // Near-wall epsilon is pinned by the epsilonWallFunction constraint (setValues): it must match
    // OF to machine precision. Interior + k carry the iterative-solve / interpolation FP floor.
    std::vector<char> isWall(m.nCells(), 0);
    for (const FvPatch& fp : patches) if (fp.type == "wall") for (label i = 0; i < fp.size; ++i) isWall[fp.faceCells[i]] = 1;
    std::vector<scalar> eWall, eWallOf;
    for (label c = 0; c < m.nCells(); ++c) if (isWall[c]) { eWall.push_back(eps.internal[c]); eWallOf.push_back(eof[c]); }

    const scalar rk = relErr(k.internal, kof), re = relErr(eps.internal, eof), rn = relErr(nut.internal, nof);
    const scalar reWall = relErr(eWall, eWallOf);   // wall-constraint pin: expect ~1e-13
    const scalar tol = 6e-3;
    const bool ok = rk < tol && re < tol && rn < tol && reWall < 1e-9;
    std::printf("test_kepsilon_correct: k rel=%.3e  epsilon rel=%.3e  nut rel=%.3e (tol %.0e)\n", rk, re, rn, tol);
    std::printf("  epsilon near-wall (constraint pin) rel=%.3e (must be ~1e-13)\n", reWall);
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
