// Phase 3 gate, fvc::grad(p) (Gauss linear) vs OpenFOAM grad(p).
// On the converged pitzDaily (step 282), read p, evaluate its BCs, compute the Gauss
// gradient and compare to OF's written grad(p). Both use the same 282/p, so agreement
// should be ~machine precision (limited only by weights/Sf, validated to ~1e-14).
// Run: test_fvc_grad <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: %s <caseDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1];

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);

    GeometricField<scalar> p = buildField<scalar>(readField<scalar>(caseDir + "/282/p"), patches, m.nCells());
    p.evaluateBoundary();

    const std::vector<vector> grad   = fvc::gaussGrad(p, m, g, patches);
    const std::vector<vector> gradOf = readField<vector>(caseDir + "/282/grad(p)").internalField;

    if ((label)gradOf.size() != m.nCells()) { std::printf("FAIL ref size %zu\n", gradOf.size()); return 1; }

    scalar maxAbs = 0.0, maxMag = 0.0;
    for (label c = 0; c < m.nCells(); ++c) {
        maxAbs = std::fmax(maxAbs, mag(grad[c] - gradOf[c]));
        maxMag = std::fmax(maxMag, mag(gradOf[c]));
    }
    const scalar relErr = maxAbs / maxMag;
    const scalar tol = 1e-9;
    const bool ok = (relErr <= tol);

    std::printf("test_fvc_grad: nCells=%d  max|grad|=%.4e  maxAbsErr=%.3e  relErr=%.3e (tol %.0e)  %s\n",
                (int)m.nCells(), maxMag, maxAbs, relErr, tol, ok ? "OK" : "FAIL");
    std::printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
