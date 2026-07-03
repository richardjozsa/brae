// Cyclic increment C3c: cyclic in the FULL SIMPLE loop. A periodic channel (cyclic in x, no-slip walls
// in y) driven by a uniform body force g_x converges to laminar Poiseuille flow. Two checks:
//   (1) x-INVARIANCE (the cyclic-correctness gate): the converged U_x must be identical for all cells
//       at the same y, a wrong cyclic coupling would break periodicity. Machine precision.
//   (2) the y-profile matches the analytic parabola u(y) = (g/2nu) y(1-y) (discretisation-limited).
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "cyclic_interface.cuh"
#include "cyclic_field.cuh"
#include "fvc.cuh"
#include "cyclic_simple.cuh"
#include <cmath>
#include <cstdio>
#include <map>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    const std::string caseDir = argc > 1 ? argv[1] : "validation/cyclicChannel";
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const std::vector<CyclicInterface> cyclics = buildCyclicInterfaces(m, g, fvp);
    const label nC = m.nCells();
    const scalar nu = 0.01, gx = 0.1, relaxU = 0.7, relaxP = 0.3, tol = 1e-14;

    GeometricField<vector> U = buildCyclicField<vector>(std::vector<vector>(nC), fvp, cyclics, /*wallNoSlip*/true); U.evaluateBoundary();
    GeometricField<scalar> p = buildCyclicField<scalar>(std::vector<scalar>(nC, 0.0), fvp, cyclics); p.evaluateBoundary();
    SurfaceScalarField phi = fvc::flux(U, m, g, fvp);

    scalar lastChange = 1.0; int it = 0;
    for (; it < 2000; ++it) {
        std::vector<scalar> Uprev(nC); for (label c = 0; c < nC; ++c) Uprev[c] = U.internal[c].x;
        cyclicSimpleStep(m, g, fvp, cyclics, U, p, phi, nu, gx, relaxU, relaxP, tol, tol);
        scalar ch = 0; for (label c = 0; c < nC; ++c) ch = std::fmax(ch, std::fabs(U.internal[c].x - Uprev[c]));
        lastChange = ch; if (ch < 1e-11) { ++it; break; }
    }

    // (1) x-invariance: group cells by y, measure spread of U_x within each group.
    std::map<long, std::vector<scalar>> byY;
    for (label c = 0; c < nC; ++c) byY[std::lround(g.C()[c].y * 1e6)].push_back(U.internal[c].x);
    scalar maxSpread = 0, umax = 0;
    for (auto& kv : byY) { scalar lo = kv.second[0], hi = kv.second[0];
        for (scalar v : kv.second) { lo = std::fmin(lo, v); hi = std::fmax(hi, v); umax = std::fmax(umax, std::fabs(v)); }
        maxSpread = std::fmax(maxSpread, hi - lo); }

    // (2) analytic Poiseuille parabola.
    scalar nerr = 0, dref = 0;
    for (label c = 0; c < nC; ++c) { const scalar y = g.C()[c].y; const scalar ue = (gx/(2.0*nu))*y*(1.0-y);
        nerr += (U.internal[c].x - ue)*(U.internal[c].x - ue); dref += ue*ue; }
    const scalar parErr = std::sqrt(nerr/dref);

    std::printf("cyclic channel (nC=%d, %d SIMPLE iters, lastdU=%.1e, Umax=%.4f):\n", nC, it, lastChange, umax);
    std::printf("  (1) x-invariance spread : %.3e   (cyclic-correctness gate)\n", maxSpread);
    std::printf("  (2) parabola L2 vs analytic: %.3e (discretisation-limited)\n", parErr);
    const bool pass = maxSpread < 1e-9 && parErr < 3e-2;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
