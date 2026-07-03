// Phase 2 gate, nutkWallFunction vs OpenFOAM. On the converged pitzDaily (step 282), compute
// nut at each wall patch from the k field and compare to OF's written nut wall values.
// Run: test_wall_function <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "foam_field_reader.cuh"
#include "nut_wall_function.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <caseDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1];
    const scalar nu = 1e-5;   // pitzDaily transportProperties

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);

    const FieldData<scalar> kf   = readField<scalar>(caseDir + "/282/k");
    const FieldData<scalar> nutf = readField<scalar>(caseDir + "/282/nut");
    std::vector<scalar> kInt = kf.internalUniform
                                   ? std::vector<scalar>(m.nCells(), kf.internalUniformValue)
                                   : kf.internalField;

    int fails = 0;
    scalar maxAbs = 0.0, maxRel = 0.0, maxNut = 0.0;
    for (const PatchFieldData<scalar>& bd : nutf.boundary) {
        if (bd.type != "nutkWallFunction") continue;
        const FvPatch* fp = nullptr;
        for (const FvPatch& p : patches) if (p.name == bd.name) { fp = &p; break; }
        if (!fp || !bd.hasValue) { std::printf("  FAIL %s missing patch/value\n", bd.name.c_str()); ++fails; continue; }

        // Near-wall distance = orthogonal distance from cell centre to the wall face plane.
        std::vector<scalar> y(fp->size);
        for (label i = 0; i < fp->size; ++i) {
            const label f = fp->start + i;
            const vector nHat = g.Sf()[f] / g.magSf()[f];
            y[i] = dot(g.Cf()[f] - g.C()[fp->faceCells[i]], nHat);
        }
        const std::vector<scalar> cfNut = nutkWallFunction(*fp, y, kInt, nu);
        for (label i = 0; i < fp->size; ++i) {
            const scalar of = bd.valueUniform ? bd.uniformValue : bd.values[i];
            const scalar d  = std::fabs(cfNut[i] - of);
            maxAbs = std::fmax(maxAbs, d);
            maxNut = std::fmax(maxNut, std::fabs(of));
            if (std::fabs(of) > 1e-12) maxRel = std::fmax(maxRel, d / std::fabs(of));
        }
        std::printf("  patch %-10s faces=%d\n", bd.name.c_str(), (int)fp->size);
    }

    // Field-scale relative error (maxAbs/maxNut) is the meaningful metric: per-face maxRel
    // blows up at viscous/log-transition faces where nut->0. The ~2e-6 floor is the converged
    // 282/k input precision (writePrecision 6); a formula error would be O(0.1-1).
    const scalar fieldRel = maxAbs / maxNut;
    const scalar tol = 1e-4;
    const bool ok = (fieldRel <= tol);
    if (!ok) ++fails;
    std::printf("test_wall_function: maxNut=%.4e maxAbs=%.3e  fieldRel=maxAbs/maxNut=%.3e"
                " (tol %.0e)  perFaceMaxRel=%.2e  %s\n",
                maxNut, maxAbs, fieldRel, tol, maxRel, ok ? "OK" : "FAIL");
    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
