// MRF increment M2+M3: MRF in the full SIMPLE loop. A closed box whose walls move with the rotating
// frame (U_wall = Omega x r) must spin up to SOLID-BODY ROTATION U = Omega x (C - origin), the relative
// velocity is zero everywhere (the analytic MRF solution). Checks: (1) converged U matches the solid-body
// field; (2) the relative flux is ~zero (fluid stationary in the rotating frame).
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvc.cuh"
#include "mrf_zone.cuh"
#include "mrf_simple.cuh"
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
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const vector axis{0, 0, 1}, origin{0, 0, 0}; const scalar omega = 1.0, nu = 0.1;

    std::vector<label> allCells(nC); for (label c = 0; c < nC; ++c) allCells[c] = c;
    const MRFZone z = buildMRFZone(m, allCells, axis, omega, origin);

    // U: walls move with the frame (fixedValue = Omega x r); empty z; interior starts at rest.
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

    SurfaceScalarField phi = fvc::flux(U, m, g, fvp); makeRelative(phi, z, g, fvp);

    int it = 0; scalar ch = 1.0;
    for (; it < 4000; ++it) {
        std::vector<vector> Up = U.internal;
        mrfSimpleStep(m, g, fvp, z, U, p, phi, nu, 0.7, 0.3, 1e-13, 1e-13);
        ch = 0; for (label c = 0; c < nC; ++c) ch = std::fmax(ch, mag(U.internal[c] - Up[c]));
        if (ch < 1e-11) { ++it; break; }
    }

    // (1) converged U == solid body Omega x (C - origin).
    scalar nerr = 0, dref = 0;
    for (label c = 0; c < nC; ++c) { const vector ue = cross(z.Omega, g.C()[c] - origin);
        nerr += magSqr(U.internal[c] - ue); dref += magSqr(ue); }
    const scalar solidErr = std::sqrt(nerr / dref);

    // (2) relative flux of the converged field is ~zero.
    SurfaceScalarField phiAbs = fvc::flux(U, m, g, fvp); makeRelative(phiAbs, z, g, fvp);
    scalar maxRel = 0, scaleF = 0;
    for (label f = 0; f < nIf; ++f) { maxRel = std::fmax(maxRel, std::fabs(phiAbs.internal[f]));
        scaleF = std::fmax(scaleF, std::fabs(fvc::flux(U, m, g, fvp).internal[f])); }

    std::printf("MRF rotation (nC=%d, %d SIMPLE iters, lastdU=%.1e):\n", nC, it, ch);
    std::printf("  (1) U vs solid-body Omega x r : %.3e (discretisation-limited, O(h^2) -> 0 on refine)\n", solidErr);
    std::printf("  (2) relative flux (abs max)   : %.3e (fluid ~stationary in rotating frame)\n", maxRel);
    const bool pass = solidErr < 3e-2 && maxRel < 1e-3;   // 12x12: 2.0e-2 / 2.4e-4; 24x24: 3.6e-3 / 2.9e-5
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
