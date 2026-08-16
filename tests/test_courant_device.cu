// The Courant number, on the device, with the coupled-interface flux included.
//
// OF CourantNo.H:
//     CoNum     = 0.5*max(surfaceSum(mag(phi))/V)*deltaT
//     meanCoNum = 0.5*(sum(surfaceSum(mag(phi)))/sum(V))*deltaT
//
// TWO THINGS WERE WRONG WITH THE HOST VERSION, and only one of them was about speed.
//
// The visible one: it pulled the whole internal and boundary flux arrays back from the device every
// step of an adaptive run and looped over faces and cells serially.
//
// The one that matters: it read those two arrays and NOTHING ELSE. cyclic and AMI flux live in their
// own buffers, so on a coupled mesh every interface cell had its Courant number computed from a partial
// flux sum. That is not a diagnostic inaccuracy -- on an `adjustTimeStep` case maxCo chooses the next
// deltaT, so an understated Co near a rotating interface hands the solver a step it cannot take. The
// cells worst affected are precisely the ones a sliding or rotating interface makes fastest.
//
// Leg 3 is therefore the load-bearing one: it asserts the interface flux CHANGES the answer. Leg 2
// (agreement with an independent host reference) would pass just as happily on a reduction that
// silently ignored the interfaces, which is exactly the state being fixed.
#include "box_mesh.cuh"
#include "device_buffer.cuh"
#include "device_blas.cuh"
#include "device_ddt.cuh"
#include "device_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
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
} // namespace

int main()
{
    std::printf("== Courant number on the device, interfaces included ==\n");

    PrimitiveMesh m = boxtest::boxMesh(5, 4, 3);
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const label nC = m.nCells();
    const int nIf = (int)m.nInternalFaces();
    const int nBf = dm.nBndFaces;

    // Signed, varying flux: |phi| is what the sum uses, so a sign pattern proves the magnitude is taken.
    std::vector<scalar> hPhiI(nIf), hPhiB(nBf);
    for (int f = 0; f < nIf; ++f) hPhiI[f] = ((f % 3) ? 1.0 : -1.0)*(0.2 + 0.01*f);
    for (int b = 0; b < nBf; ++b) hPhiB[b] = ((b % 2) ? -1.0 : 1.0)*(0.5 + 0.02*b);
    DeviceBuffer<scalar> phiI, phiB;
    phiI.copyFrom(hPhiI); phiB.copyFrom(hPhiB);

    // A synthetic interface carrying a LARGE flux on a few cells -- the situation a rotating AMI
    // creates, and the one the old sum could not see.
    const int nIface = 5;
    std::vector<label> ifOwn(nIface);
    std::vector<scalar> ifPhi(nIface);
    for (int i = 0; i < nIface; ++i) { ifOwn[i] = (label)(2*i); ifPhi[i] = -(30.0 + 5.0*i); }
    DeviceBuffer<label> dIfOwn; DeviceBuffer<scalar> dIfPhi;
    dIfOwn.copyFrom(ifOwn); dIfPhi.copyFrom(ifPhi);

    // Independent host reference, written from OF's definition rather than from brae's kernel.
    auto hostSum = [&](bool withIface)
    {
        std::vector<scalar> s(static_cast<std::size_t>(nC), 0.0);
        for (int f = 0; f < nIf; ++f)
        {
            const scalar a = std::fabs(hPhiI[f]);
            s[m.owner()[f]] += a;
            s[m.neighbour()[f]] += a;
        }
        for (int b = 0; b < nBf; ++b) s[m.owner()[m.nInternalFaces() + b]] += std::fabs(hPhiB[b]);
        if (withIface)
            for (int i = 0; i < nIface; ++i) s[ifOwn[i]] += std::fabs(ifPhi[i]);
        return s;
    };

    DeviceBuffer<scalar> sumNo, sumWith;
    deviceSurfaceSumMagPhi(dm, phiI, phiB, nullptr, nullptr, nullptr, nullptr, sumNo);
    deviceSurfaceSumMagPhi(dm, phiI, phiB, &dIfOwn, &dIfPhi, nullptr, nullptr, sumWith);
    std::vector<scalar> gotNo, gotWith;
    sumNo.copyTo(gotNo); sumWith.copyTo(gotWith);

    // ---- Leg 1: the max-ratio reduction itself ------------------------------------------------------
    // The Courant number is a MAXIMUM, which cannot be assembled from the sum/dot reductions that
    // already existed, so this is genuinely new arithmetic and gets its own check.
    {
        const std::vector<scalar>& V = g.V();
        scalar want = 0;
        for (label c = 0; c < nC; ++c) if (V[c] > 0) want = std::max(want, gotWith[c]/V[c]);
        const scalar got = deviceMaxRatio(sumWith, dm.V);
        check(std::fabs(got - want) <= 1e-12*std::max(want, scalar(1)), "deviceMaxRatio == max(x/V)");
        check(want > 0, "vacuity guard: the maximum is not zero");
    }

    // ---- Leg 2: the surface sum matches an independent host reference -------------------------------
    {
        const std::vector<scalar> refNo = hostSum(false), refWith = hostSum(true);
        scalar eNo = 0, eWith = 0;
        for (label c = 0; c < nC; ++c)
        {
            eNo   = std::max(eNo,   std::fabs(gotNo[c]   - refNo[c]));
            eWith = std::max(eWith, std::fabs(gotWith[c] - refWith[c]));
        }
        check(eNo   < 1e-12, "surfaceSum(mag(phi)) matches the host reference (no interface)");
        check(eWith < 1e-12, "...and with an interface coupled in");
    }

    // ---- Leg 3: the interface CHANGES the answer ----------------------------------------------------
    // The discrimination. A reduction that ignored the interface -- the state being fixed -- would pass
    // Leg 2's first half and produce an identical Courant number here.
    {
        const std::vector<scalar>& V = g.V();
        scalar maxNo = 0, maxWith = 0;
        for (label c = 0; c < nC; ++c)
            if (V[c] > 0)
            {
                maxNo   = std::max(maxNo,   gotNo[c]/V[c]);
                maxWith = std::max(maxWith, gotWith[c]/V[c]);
            }
        check(maxWith > 1.5*maxNo,
              "the interface flux RAISES the Courant number -- omitting it understates the step limit");
        std::printf("        (max sumPhi/V  without %.4e  with %.4e)\n", (double)maxNo, (double)maxWith);
    }

    // ---- Leg 4: |phi| is used, not phi --------------------------------------------------------------
    // Every interface flux above is negative. A sum that added the signed value would come out smaller
    // than the no-interface case, not larger, so Leg 3 already implies this -- but pinning it directly
    // keeps a future sign change from being absorbed silently.
    {
        scalar minCell = 1e300;
        for (label c = 0; c < nC; ++c) minCell = std::min(minCell, gotWith[c]);
        check(minCell >= 0, "the sum is over magnitudes: no cell total is negative");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
