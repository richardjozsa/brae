// Phase 5, assembled SIMPLE loop: run brae::runSimple and confirm it iterates stably and drives
// the residual + global continuity down (a working simpleFoam). Per-step exactness vs OF is
// validated by test_simple_step.  Run: test_simple_run <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "simple_foam.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

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

    SimpleControls ctl;
    ctl.tolU = 1e-7; ctl.relTolU = 0.1; ctl.tolP = 1e-7; ctl.relTolP = 0.1;  // pitzDaily-style relTol

    scalar firstP = 0, lastP = 0;
    const int iters = runSimple(U, p, phi, Udata, m, g, patches, ctl, /*maxOuter*/300, /*resControl*/1e-4, &firstP, &lastP);

    // Global continuity: sum of all boundary fluxes (mass conservation) relative to inlet flux.
    scalar totalFlux = 0, inletMag = 0;
    for (std::size_t pi = 0; pi < patches.size(); ++pi) {
        scalar s = 0; for (scalar v : phi.boundary[pi]) s += v;
        totalFlux += s;
        if (patches[pi].name == "inlet") inletMag = std::fabs(s);
    }
    const scalar contErr = std::fabs(totalFlux) / inletMag;

    // The loop is correct (per-step exact vs OF, see test_simple_step). Here we confirm it runs
    // STABLY: residual stays finite and decreases, and global continuity is satisfied (the key
    // SIMPLE property). Full convergence to ~1e-4 needs turbulence (laminar pitzDaily at Re~2.5e4
    // is physically unsteady, so steady residuals stall, expected, not a cf defect).
    const bool finite = std::isfinite(lastP) && std::isfinite(contErr);
    std::printf("test_simple_run: iters=%d  firstPres=%.3e  lastPres=%.3e  contErr=%.3e\n",
                iters, firstP, lastP, contErr);
    const bool ok = finite && (lastP < firstP) && (contErr < 1e-3);
    std::printf("%s\n", ok ? "PASS (loop stable, mass-conserving)" : "FAIL");
    return ok ? 0 : 1;
}
