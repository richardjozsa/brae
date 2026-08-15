// constrainHbyA at an inletOutlet outlet -- which patches OpenFOAM constrains, and which it pointedly
// does not.
//
// OF fvMatrices.C::constrainHbyA overwrites HbyA's boundary with U's boundary value on exactly the
// patches where
//     !U.boundaryField()[patchi].assignable()
// and assignable() is NOT the same question as "is the value fixed":
//     fixedValue          assignable() = false   -> constrained     (HbyA_b = U_b)
//     mixed               assignable() = false   -> constrained
//     outletInlet         mixed, no override     -> constrained
//     inletOutlet         OVERRIDES back to true -> NOT constrained (inletOutletFvPatchField.H:164)
// So at an inletOutlet patch HbyA keeps the boundary it was born with: the extrapolatedCalculated
// boundary of rAU*UEqn.H(), i.e. the ADJACENT CELL value -- even on a face where U itself is clamped to
// inletValue because the flux reversed.
//
// WHY IT MATTERS. phiHbyA_b = HbyA_b & Sf is the flux the pressure equation starts from. Reading it
// through U's inletOutlet descriptor gives ZERO on every backflow face, which is a different pressure
// equation, not a rounding difference: on pimpleFoam/RAS/TJunction, whose outlet1 reverses at step 2,
// it put p in the boundary-adjacent cell 0.54 below OpenFOAM's, tripled the backflow velocity there
// (0.837 vs 0.285 m/s) and left the whole field at U L2 1.0e-01. With HbyA extrapolated: 7.8e-04.
//
// Leg 3 is the discriminating one: outletInlet is the same mixed switch with the opposite sense, and OF
// DOES constrain it. A fix that extrapolated "all the flux-switching patches" would pass legs 1-2 and
// fail leg 3.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "device_buffer.cuh"
#include "device_boundary.cuh"
#include "box_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       std::printf("  ok:   %s\n", what);
}
} // namespace

int main()
{
    std::printf("== constrainHbyA at inletOutlet / outletInlet ==\n");

    PrimitiveMesh m = boxtest::boxMesh(4, 4, 3);
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    // Distinctive constants so every possible answer is distinguishable by inspection:
    //   U cell field, HbyA cell field, inletValue and the fixed inlet velocity are all different.
    const vector INLET_VALUE{7, 8, 9};        // inletOutlet's clamped backflow value
    const vector OUTLET_VALUE{-3, -4, -5};    // outletInlet's clamped outflow value
    const vector FIXED_U{2, 0, 0};            // a plain fixedValue inlet

    GeometricField<vector> U;  U.internal.resize(nC);
    GeometricField<vector> H;  H.internal.resize(nC);
    for (label c = 0; c < nC; ++c)
    {
        U.internal[c] = {1.0 + 0.01*c, 0.5, -0.25};
        H.internal[c] = {100.0 + c, 200.0 + c, 300.0 + c};   // nothing like U
    }

    std::size_t ioPatch = 0, oioPatch = 0, fixPatch = 0;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        const FvPatch& q = fvp[pi];
        auto mk = [&](auto&& f) { U.boundary.push_back(f()); H.boundary.push_back(f()); };
        if (pi == 0)
        {
            fixPatch = pi;
            mk([&]{ return std::make_unique<FixedValuePatchField<vector>>(q, true, FIXED_U, std::vector<vector>{}); });
        }
        else if (pi == 1)
        {
            ioPatch = pi;
            mk([&]{ return std::make_unique<InletOutletPatchField<vector>>(q, true, INLET_VALUE, std::vector<vector>{}); });
        }
        else if (pi == 2)
        {
            oioPatch = pi;
            mk([&]{ return std::make_unique<OutletInletPatchField<vector>>(q, true, OUTLET_VALUE, std::vector<vector>{}); });
        }
        else
        {
            mk([&]{ return std::make_unique<ZeroGradientPatchField<vector>>(q); });
        }
    }
    U.evaluateBoundary();
    H.evaluateBoundary();
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);

    // Flatten the boundary bookkeeping the same way the device builder does, so a face index means the
    // same thing on both sides.
    std::vector<label> patchOf, cellOf;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i) { patchOf.push_back((label)pi); cellOf.push_back(fvp[pi].faceCells[i]); }
    const int nB = (int)patchOf.size();

    DeviceBuffer<scalar> Hc[3], hb[3], ext[3];
    for (int k = 0; k < 3; ++k)
    {
        std::vector<scalar> comp(nC);
        for (label c = 0; c < nC; ++c) comp[c] = component(H.internal[c], k);
        Hc[k].copyFrom(comp);
        // HbyA evaluated through U's OWN descriptors -- what the solver does before the fix.
        deviceBCValue(dbU.comp[k], Hc[k], hb[k]);
        // ...and the extrapolatedCalculated boundary OF actually gives HbyA: the adjacent cell value.
        std::vector<scalar> e(nB);
        for (int i = 0; i < nB; ++i) e[i] = comp[cellOf[i]];
        ext[k].copyFrom(e);
    }

    std::vector<scalar> before[3];
    for (int k = 0; k < 3; ++k) hb[k].copyTo(before[k]);

    // ---- negative control: without the fix, the io faces carry the CLAMPED inlet value -------------
    {
        bool clamped = true, anyIO = false;
        for (int i = 0; i < nB; ++i)
            if (patchOf[i] == (label)ioPatch)
            {
                anyIO = true;
                clamped = clamped && std::fabs(before[0][i] - INLET_VALUE.x) < 1e-12
                                  && std::fabs(before[1][i] - INLET_VALUE.y) < 1e-12;
            }
        check(anyIO, "vacuity guard: the fixture really has inletOutlet faces");
        check(clamped, "negative control: evaluating HbyA through U's descriptor gives inletValue, not HbyA_cell");
    }

    deviceExtrapolateIOHbyA(dbU, ext[0], ext[1], ext[2], hb[0], hb[1], hb[2]);

    std::vector<scalar> after[3];
    for (int k = 0; k < 3; ++k) hb[k].copyTo(after[k]);

    // ---- Leg 1: inletOutlet faces now carry HbyA's own adjacent-cell value -------------------------
    {
        scalar err = 0;
        for (int i = 0; i < nB; ++i)
            if (patchOf[i] == (label)ioPatch)
                for (int k = 0; k < 3; ++k)
                    err = std::max(err, std::fabs(after[k][i] - component(H.internal[cellOf[i]], k)));
        check(err < 1e-12, "inletOutlet: HbyA_b == the adjacent-cell HbyA (OF leaves it assignable)");
    }

    // ---- Leg 2: the fixedValue patch is untouched -- constrainHbyA still owns it -------------------
    {
        scalar err = 0;
        for (int i = 0; i < nB; ++i)
            if (patchOf[i] == (label)fixPatch)
            {
                err = std::max(err, std::fabs(after[0][i] - FIXED_U.x));
                err = std::max(err, std::fabs(after[1][i] - FIXED_U.y));
            }
        check(err < 1e-12, "fixedValue: HbyA_b stays U_b (still constrained, as OF does)");
    }

    // ---- Leg 3: outletInlet is NOT extrapolated -- OF constrains it --------------------------------
    {
        scalar moved = 0;
        for (int i = 0; i < nB; ++i)
            if (patchOf[i] == (label)oioPatch)
                for (int k = 0; k < 3; ++k) moved = std::max(moved, std::fabs(after[k][i] - before[k][i]));
        check(moved == 0.0, "outletInlet: untouched -- mixed without an assignable() override IS constrained");
    }

    // ---- Leg 4: nothing else moved ----------------------------------------------------------------
    {
        scalar moved = 0;
        for (int i = 0; i < nB; ++i)
            if (patchOf[i] != (label)ioPatch)
                for (int k = 0; k < 3; ++k) moved = std::max(moved, std::fabs(after[k][i] - before[k][i]));
        check(moved == 0.0, "every non-inletOutlet face is left exactly as it was");
    }

    // ---- Leg 5: a case with no inletOutlet patch is a no-op ----------------------------------------
    {
        GeometricField<vector> U2; U2.internal = U.internal;
        for (const FvPatch& q : fvp) U2.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
        U2.evaluateBoundary();
        const DeviceVectorBoundary db2 = buildDeviceVectorBoundary(U2, fvp, g);
        DeviceBuffer<scalar> h2[3];
        std::vector<scalar> pre[3], post[3];
        for (int k = 0; k < 3; ++k) { deviceBCValue(db2.comp[k], Hc[k], h2[k]); h2[k].copyTo(pre[k]); }
        deviceExtrapolateIOHbyA(db2, ext[0], ext[1], ext[2], h2[0], h2[1], h2[2]);
        scalar moved = 0;
        for (int k = 0; k < 3; ++k)
        {
            h2[k].copyTo(post[k]);
            for (int i = 0; i < nB; ++i) moved = std::max(moved, std::fabs(post[k][i] - pre[k][i]));
        }
        check(moved == 0.0, "no inletOutlet patch: the call changes nothing at all");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
