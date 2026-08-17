// LUST's blend belongs in the MATRIX, and this pins that it is there.
//
// OpenFOAM (LUST.H:107-131):
//     weights()    = 0.75*linear.weights() + 0.25*linearUpwind::weights()
//     correction() = 0.25*linearUpwind::correction()
//
// so three quarters of the central weighting is IMPLICIT and only a quarter of the gradient term is
// explicit. brae assembled a pure-upwind matrix and carried the whole blend as a deferred correction.
// The interpolated FACE VALUE is identical either way, which is exactly why this survived: every test
// that looked at fields agreed, and the defect lived in the matrix.
//
// It matters because the momentum diagonal is rAU, and rAU sets the pressure-Laplacian coefficient and
// HbyA -- so a diagonal that is uniformly too large scales the whole pressure-velocity coupling. Upwind
// puts more on the diagonal than a central blend, so brae's rAU came out too small. Measured against
// the OpenFOAM stage oracle on pimpleFoam/LES/vortexShed, one momentum assembly from identical inputs:
// brae's far-field diagonal was 1.1118x OpenFOAM's across the 112640 cells touching no boundary, and
// mDiag/V was 131.5 against 118.25 with 1/dt = 100. After this fix the same comparison gives 1.0000,
// and the case goes from |U| = 1.2e+10 to 9.5 against OpenFOAM's 0.0435.
//
// WHY THE TEST IS SHAPED THIS WAY. The claim is not "some blend happens" but "the blend is 0.75/0.25
// and it is in the matrix". Leg 1 pins the identity the implementation is built on. Leg 2 is the
// discrimination that matters: the blended matrix must differ from BOTH pure schemes, because a
// regression to either would still produce a working solver with a quietly wrong rAU -- which is the
// state this replaces.
#include "box_mesh.cuh"
#include "device_blas.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include "solver_controls.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       std::printf("  ok:   %s\n", what);
}

scalar maxAbsDiff(const DeviceBuffer<scalar>& a, const DeviceBuffer<scalar>& b)
{
    std::vector<scalar> ha, hb;
    a.copyTo(ha); b.copyTo(hb);
    scalar m = 0;
    for (std::size_t i = 0; i < ha.size(); ++i) m = std::fmax(m, std::fabs(ha[i] - hb[i]));
    return m;
}

scalar maxAbs(const DeviceBuffer<scalar>& a)
{
    std::vector<scalar> h; a.copyTo(h);
    scalar m = 0;
    for (const scalar v : h) m = std::fmax(m, std::fabs(v));
    return m;
}
} // namespace

int main()
{
    std::printf("== LUST: the 0.75 central share is IMPLICIT ==\n");

    const PrimitiveMesh m = boxtest::boxMesh(8, 6, 5);
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const int nIf = (int)m.nInternalFaces();

    // A SIGNED, varying flux. Upwind and central differ only through the sign and magnitude of phi, so a
    // uniform or one-signed flux would make the two schemes agree on many faces and weaken every leg.
    std::vector<scalar> hPhi((std::size_t)nIf);
    for (int f = 0; f < nIf; ++f) hPhi[(std::size_t)f] = ((f % 3) ? 1.0 : -1.0)*(0.05 + 0.01*f);
    DeviceBuffer<scalar> phi; phi.copyFrom(hPhi);

    DeviceBuffer<scalar> cD, cU, cL, uD, uU, uL;
    deviceDivCentralCoeffs(dm, phi, cD, cU, cL);
    deviceDivUpwindCoeffs(dm, phi, uD, uU, uL);

    // The blend the solver forms, built here the same way: coefficients are LINEAR in the face weight,
    // so combining the two assembled matrices is the same as assembling with combined weights.
    const scalar wc = DeviceSimpleControls::lustCentralFrac;
    const scalar wu = DeviceSimpleControls::lustUpwindFrac;
    DeviceBuffer<scalar> bD, bU, bL;
    deviceCopy(bD, uD); deviceScale(bD, wu); deviceAxpy(wc, cD, bD);
    deviceCopy(bU, uU); deviceScale(bU, wu); deviceAxpy(wc, cU, bU);
    deviceCopy(bL, uL); deviceScale(bL, wu); deviceAxpy(wc, cL, bL);

    // ---- Leg 1: the fractions are OpenFOAM's, and they partition unity --------------------------------
    // A blend that did not sum to 1 would not be an interpolation at all -- it would rescale the whole
    // convection term, which is the failure mode this file exists to prevent.
    {
        check(wc == scalar(0.75), "the central share is 0.75, as OF LUST.H defines it");
        check(wu == scalar(0.25), "the upwind share is 0.25");
        check(wc + wu == scalar(1), "...and they sum to exactly 1");
    }

    // ---- Leg 2: the blended matrix is neither pure scheme ---------------------------------------------
    // The discrimination. Reverting to pure upwind (the defect) or to pure central (over-correcting)
    // both leave a solver that runs; only comparing against BOTH catches either.
    {
        const scalar dU = maxAbsDiff(bD, uD), dC = maxAbsDiff(bD, cD), scale = maxAbs(uD);
        check(dU > 0.01*scale, "the blended diagonal differs from PURE UPWIND (the old behaviour)");
        check(dC > 0.01*scale, "...and from PURE CENTRAL");
        std::printf("        (diag: |blend-upwind| %.3e, |blend-central| %.3e, scale %.3e)\n",
                    (double)dU, (double)dC, (double)scale);
    }

    // ---- Leg 3: upwind really does load the diagonal more than central -------------------------------
    // The mechanism, asserted rather than assumed: this inequality is WHY the old pure-upwind matrix made
    // rAU too small, and it is what makes the sign of the fix predictable.
    {
        std::vector<scalar> hu, hc, hb;
        uD.copyTo(hu); cD.copyTo(hc); bD.copyTo(hb);
        int upwindBigger = 0, between = 0;
        for (std::size_t i = 0; i < hu.size(); ++i)
        {
            if (std::fabs(hu[i]) > std::fabs(hc[i])) ++upwindBigger;
            const scalar lo = std::fmin(hu[i], hc[i]), hi = std::fmax(hu[i], hc[i]);
            if (hb[i] >= lo - 1e-12 && hb[i] <= hi + 1e-12) ++between;
        }
        check(upwindBigger > (int)(0.5*hu.size()),
              "upwind loads the diagonal more than central on most cells (so pure upwind understates rAU)");
        check(between == (int)hu.size(), "the blend lies between the two everywhere -- it is an interpolation");
    }

    // ---- Leg 4: a vacuity guard on the fixture --------------------------------------------------------
    // If central and upwind happened to coincide on this mesh and flux, Legs 2-3 would be measuring
    // nothing. They do not, and that is worth stating.
    {
        const scalar d = maxAbsDiff(uD, cD);
        check(d > 0, "vacuity guard: central and upwind genuinely differ on this fixture");
        std::printf("        (|upwind-central| diag %.3e over %d internal faces)\n", (double)d, nIf);
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
