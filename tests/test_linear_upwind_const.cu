// CONSTANT PRESERVATION of the scalar linearUpwind deferred correction.
//
// A convection scheme must transport a uniform field exactly. For Gauss linearUpwind that is structural:
// the deferred correction is
//     corr_f = (Cf - C_upwindCell) & grad(field)_upwindCell
// so a constant field has grad == 0 and the correction is IDENTICALLY ZERO. OF gets this exactly -- one
// SIMPLE iteration on pitzDaily from a uniform k gives bit-identical results with `Gauss upwind` and
// `Gauss linearUpwind grad(k)`.
//
// brae did not: the same comparison differed by 9.4e-03, because the correction was non-zero on a
// constant field. That is why linearUpwind on k diverged from a COLD start (a uniform initial field is
// the worst case -- the spurious source is the entire signal) while still agreeing to 1.6e-06 when
// started from an already-converged state, where genuine gradients dominate the spurious part. The
// symptom looked like a stability problem and was gated off as one; it was a consistency bug.
//
// This test is deliberately at the operator level rather than the solver level. The solver-level version
// is a 60-iteration two-code comparison; this is three kernels and an exact zero, and it fails loudly for
// the actual reason instead of "the case diverged".

#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvc.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_simple.cuh"   // deviceLinearUpwindCorr
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

scalar maxAbs(const std::vector<scalar>& v)
{
    scalar m = 0;
    for (scalar x : v) m = std::fmax(m, std::fabs(x));
    return m;
}

}   // namespace

int main(int argc, char** argv)
{
    const std::string caseDir = argc > 1 ? argv[1] : "validation/pitzDaily";
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    // A UNIFORM scalar field, with every patch carrying the same constant. This is exactly the state a
    // cold start begins from (pitzDaily 0/k is `internalField uniform 0.375`).
    const scalar K = 0.375;
    GeometricField<scalar> k; k.internal.assign(nC, K);
    for (const FvPatch& q : fvp)
    {
        if (q.type == "empty")     k.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q));
        else if (q.type == "wall") k.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
        else                       k.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(q, true, K, std::vector<scalar>{}));
    }
    k.evaluateBoundary();

    // A non-trivial flux, so the correction is exercised on every face and both upwind directions occur.
    GeometricField<vector> U; U.internal.resize(nC);
    for (label c = 0; c < nC; ++c) U.internal[c] = { std::sin(0.01*c), std::cos(0.013*c), 0.0 };
    for (const FvPatch& q : fvp)
    {
        if (q.type == "empty") U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
        else                   U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
    }
    U.evaluateBoundary();
    const SurfaceScalarField phi = fvc::flux(U, m, g, fvp);

    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    std::vector<std::vector<scalar>> kbnd;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) kbnd.push_back(k.boundary[pi]->value());
    DeviceBuffer<scalar> dk(k.internal), dkb(flattenBoundary(kbnd));
    DeviceBuffer<scalar> dphi(phi.internal);

    // 1. grad of a constant must be zero -- the correction is built from it.
    DeviceBuffer<scalar> gx, gy, gz;
    deviceGaussGrad(dm, dk, dkb, gx, gy, gz);
    const scalar gmax = std::fmax(maxAbs(gx.host()), std::fmax(maxAbs(gy.host()), maxAbs(gz.host())));
    // Gauss grad of a constant is K*sum(Sf)/V. sum(Sf) over a closed cell cancels only to the mesh's own
    // closure error, and dividing by a small V amplifies it, so the bar is the measured roundoff floor
    // (4.7e-13 on pitzDaily) with margin -- NOT exact zero. A tighter bar fails on mesh arithmetic alone.
    const scalar gtol = 1e-10;
    if (gmax <= gtol)
    {
        std::printf("  OK   gaussGrad(uniform %.3f) = 0        (max |grad| %.3e <= %.1e)\n", K, gmax, gtol);
    }
    else
    {
        std::printf("  FAIL gaussGrad(uniform %.3f) is NOT zero (max |grad| %.3e > %.1e)\n"
                    "       -> sum(Sf*value) over the cell's faces does not cancel; a patch is contributing\n"
                    "          a boundary value other than the uniform one (or none at all)\n", K, gmax, gtol);
        failures++;
    }

    // 2. THE CLAIM: the deferred correction itself must be identically zero on a constant field.
    DeviceBuffer<scalar> luCorr;
    deviceLinearUpwindCorr(dm, dphi, gx, gy, gz, luCorr);
    const scalar cmax = maxAbs(luCorr.host());
    // corr = (Cf - C_up) & grad with |grad| at the roundoff floor above, so this inherits that floor
    // scaled by the face-to-cell distance and the flux; it is ~1e-22 here, not exactly 0.
    if (cmax <= 1e-16)
    {
        std::printf("  OK   linearUpwind correction on a uniform field = %.3e (roundoff)\n", cmax);
    }
    else
    {
        std::printf("  FAIL linearUpwind correction on a uniform field = %.6e, must be 0.\n"
                    "       A uniform field must convect exactly; a non-zero correction is a spurious\n"
                    "       source term. From a cold start (uniform initial field) it IS the whole signal,\n"
                    "       which is what made div(phi,k) linearUpwind diverge on pitzDaily.\n", cmax);
        failures++;
    }

    // 3. NEGATIVE CONTROL: on a NON-constant field the correction must be non-zero, or this test would
    // pass just as well against a correction that was hard-wired to zero (which would silently turn
    // linearUpwind into plain upwind everywhere -- the exact 1.6e-02 nut error the downgrade caused).
    GeometricField<scalar> kv; kv.internal.resize(nC);
    for (label c = 0; c < nC; ++c) kv.internal[c] = K + 0.1 * std::sin(0.02 * c);
    for (const FvPatch& q : fvp)
    {
        if (q.type == "empty")     kv.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q));
        else if (q.type == "wall") kv.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
        else                       kv.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(q, true, K, std::vector<scalar>{}));
    }
    kv.evaluateBoundary();
    std::vector<std::vector<scalar>> kvb;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) kvb.push_back(kv.boundary[pi]->value());
    DeviceBuffer<scalar> dkv(kv.internal), dkvb(flattenBoundary(kvb));
    DeviceBuffer<scalar> vx, vy, vz, luVar;
    deviceGaussGrad(dm, dkv, dkvb, vx, vy, vz);
    deviceLinearUpwindCorr(dm, dphi, vx, vy, vz, luVar);
    const scalar vmax = maxAbs(luVar.host());
    if (vmax > 0)
    {
        std::printf("  OK   varying field still produces a correction (max %.3e) -- not stubbed to zero\n", vmax);
    }
    else
    {
        std::printf("  FAIL correction is zero on a VARYING field too -- linearUpwind is doing nothing,\n"
                    "       so the constant-preservation check above proves nothing\n");
        failures++;
    }

    std::printf("linear_upwind_const: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
