// Device MRF gate, the GPU SIMPLE loop with an MRF rotating zone. A closed box whose walls move with the
// frame (U_wall = Omega x r) must spin up to SOLID-BODY ROTATION U = Omega x (C - origin) (zero relative
// velocity, the analytic MRF solution). Mirrors the host ctest mrf_rotation, but through DeviceSimpleSolver
// (Coriolis source + relative resident flux). Run: test_gpu_mrf <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvc.cuh"
#include "mrf_zone.cuh"
#include "device_simple_foam.cuh"

#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    const std::string caseDir = argc > 1 ? argv[1] : "validation/mrfBox";
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();
    const vector axis{0, 0, 1}, origin{0, 0, 0}; const scalar omega = 1.0;

    std::vector<label> allCells(nC); for (label c = 0; c < nC; ++c) allCells[c] = c;
    const MRFZone z = buildMRFZone(m, allCells, axis, omega, origin);

    // U: walls move with the frame (fixedValue = Omega x r); empty z; interior at rest.
    GeometricField<vector> U; U.internal.assign(nC, vector{0, 0, 0});
    for (const FvPatch& p : fvp) {
        if (p.type == "empty") { U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(p)); continue; }
        std::vector<vector> wv(p.size);
        for (label i = 0; i < p.size; ++i) wv[i] = cross(z.Omega, g.Cf()[p.start + i] - origin);
        U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(p, false, vector{0, 0, 0}, wv));
    }
    U.evaluateBoundary();
    GeometricField<scalar> p; p.internal.assign(nC, 0.0);
    for (const FvPatch& fp : fvp) {
        if (fp.type == "empty") p.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(fp));
        else                    p.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(fp));
    }
    p.evaluateBoundary();
    SurfaceScalarField phi = fvc::flux(U, m, g, fvp);     // solver.setMRF makes it relative

    DeviceSimpleControls ctl;
    ctl.nu = 0.1; ctl.relaxU = 0.7; ctl.relaxP = 0.3; ctl.tolU = 1e-13; ctl.tolP = 1e-13;
    ctl.turbulent = false; ctl.useGraph = false;
    ctl.needRef = true; ctl.pRefCell = 0; ctl.pRefValue = 0.0;   // closed box: no fixedValue-p

    DeviceSimpleSolver solver(m, g, fvp, U, p, phi, ctl);
    solver.setMRF(z, m, g);

    std::vector<vector> Uprev = U.internal; scalar ch = 1.0; int it = 0;
    for (; it < 4000; ++it) {
        solver.step();
        if (it % 25 == 0 || it > 3990) {
            const std::vector<vector> Un = solver.U();
            ch = 0; for (label c = 0; c < nC; ++c) ch = std::fmax(ch, mag(Un[c] - Uprev[c]));
            Uprev = Un; if (ch < 1e-10) { ++it; break; }
        }
    }

    const std::vector<vector> Ufin = solver.U();
    scalar nerr = 0, dref = 0;
    for (label c = 0; c < nC; ++c) { const vector ue = cross(z.Omega, g.C()[c] - origin);
        nerr += magSqr(Ufin[c] - ue); dref += magSqr(ue); }
    const scalar solidErr = std::sqrt(nerr / dref);

    std::printf("device MRF rotation (nC=%d, %d SIMPLE iters, lastdU=%.1e):\n", (int)nC, it, ch);
    std::printf("  U vs solid-body Omega x r : %.3e (host mrf_rotation: 2.0e-2 @12x12; discretisation-limited)\n", solidErr);
    const bool pass = solidErr < 3e-2;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
