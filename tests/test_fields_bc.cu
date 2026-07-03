// Phase 2 gate, field reader + GeometricField + BC evaluation.
// Reads the converged pitzDaily (step 282) U and p, builds the fields with their boundary
// conditions, evaluates them, and checks:
//   (1) internal field size == nCells; boundary types parsed;
//   (2) faceCells addressing geometrically valid (boundary face outward of its owner cell);
//   (3) BC evaluation per definition (fixedValue / noSlip / zeroGradient);
//   (4) boundary-flux continuity: sum_b U_b . Sf ~ 0  (external physical oracle that ties
//       together faceCells, Sf orientation and BC evaluation against OF's solution).
// Run: test_fields_bc <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"

#include <cmath>
#include <cstdio>
#include <string>

using namespace brae;

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <caseDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);

    GeometricField<vector> U = buildField<vector>(readField<vector>(caseDir + "/282/U"), patches, m.nCells());
    GeometricField<scalar> p = buildField<scalar>(readField<scalar>(caseDir + "/282/p"), patches, m.nCells());
    U.evaluateBoundary();
    p.evaluateBoundary();

    int fails = 0;
    auto check = [&](bool ok, const char* msg) { if (!ok) { ++fails; std::printf("  FAIL %s\n", msg); } };

    // (1) sizes
    check((label)U.internal.size() == m.nCells(), "U internal size");
    check((label)p.internal.size() == m.nCells(), "p internal size");

    // (2) faceCells geometric validity: boundary face outward of its owner cell.
    scalar minOutward = 1e30;
    for (const FvPatch& fp : patches)
        for (label i = 0; i < fp.size; ++i) {
            const label f = fp.start + i;
            minOutward = std::fmin(minOutward, dot(g.Cf()[f] - g.C()[fp.faceCells[i]], g.Sf()[f]));
        }
    check(minOutward > 0.0, "all boundary faces outward of owner");

    // (3) BC evaluation by definition.
    for (std::size_t pi = 0; pi < patches.size(); ++pi) {
        const FvPatch& fp = patches[pi];
        const auto&    Uv = U.boundary[pi]->value();
        const auto&    pv = p.boundary[pi]->value();
        if (fp.name == "inlet") {                      // U fixedValue (10 0 0), p zeroGradient
            for (label i = 0; i < fp.size; ++i)
                check(Uv[i].x == 10.0 && Uv[i].y == 0.0 && Uv[i].z == 0.0, "inlet U == (10 0 0)");
            for (label i = 0; i < fp.size; ++i)
                check(pv[i] == p.internal[fp.faceCells[i]], "inlet p == internal (zeroGradient)");
        } else if (fp.name == "outlet") {              // U zeroGradient, p fixedValue 0
            for (label i = 0; i < fp.size; ++i)
                check(Uv[i].x == U.internal[fp.faceCells[i]].x, "outlet U == internal (zeroGradient)");
            for (label i = 0; i < fp.size; ++i) check(pv[i] == 0.0, "outlet p == 0");
        } else if (fp.name == "upperWall" || fp.name == "lowerWall") {   // U noSlip
            for (label i = 0; i < fp.size; ++i)
                check(Uv[i].x == 0.0 && Uv[i].y == 0.0 && Uv[i].z == 0.0, "wall U == 0 (noSlip)");
        }
    }

    // (4) boundary-flux continuity: sum_b U_b . Sf  vs  inlet flux magnitude.
    scalar totalFlux = 0.0, inletFluxMag = 0.0;
    for (std::size_t pi = 0; pi < patches.size(); ++pi) {
        const FvPatch& fp = patches[pi];
        const auto&    Uv = U.boundary[pi]->value();
        scalar pf = 0.0;
        for (label i = 0; i < fp.size; ++i) pf += dot(Uv[i], g.Sf()[fp.start + i]);
        totalFlux += pf;
        if (fp.name == "inlet") inletFluxMag = std::fabs(pf);
    }
    const scalar contErr = std::fabs(totalFlux) / inletFluxMag;
    check(contErr < 5e-3, "boundary flux continuity");

    std::printf("test_fields_bc: nCells=%d  minOutward(Cf-C).Sf=%.3e  inletFlux=%.5e  contErr=%.3e\n",
                (int)m.nCells(), minOutward, inletFluxMag, contErr);
    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
