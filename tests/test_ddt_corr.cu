// fvc::ddtCorr -- the ddt flux correction pimpleFoam adds to phiHbyA, which brae did not have.
//
//     phiHbyA += fvc::interpolate(rAU)*fvc::ddtCorr(U, phi, Uf)          (pEqn.H, guarded by PIMPLE/ddtCorr)
//
// EulerDdtScheme::fvcDdtPhiCorr + ddtScheme::fvcDdtPhiCoeff, as verified term for term against OpenFOAM
// v2412 on pimpleFoam/RAS/oscillatingInletACMI2D (21464 internal faces, agreement 1.2e-10 L2):
//
//     phiCorr = phi_old - (interpolate(U_old) & Sf)
//     coeff   = 1 - min(|phiCorr| / (|phi_old| + SMALL), 1)
//     ddtCorr = coeff * (1/deltaT) * phiCorr
//
// WHY IT MATTERS ENOUGH TO TEST. It is ~1% of the flux on nearly every internal face (3.5% of the peak,
// 0.31% rms on that case), and omitting it was the whole remaining difference from OpenFOAM once the
// cyclicACMI defects were fixed: the static case went from 1.1e-02 to 6.5e-07 in velocity when this was
// added. A term that small is exactly the kind that looks like noise and is not.
//
// WHAT EACH LEG PINS.
//  1. the coefficient's TWO limits, which are the whole character of the term: it is 1 where the stored
//     flux already agrees with the interpolated velocity (a genuine ddt correction) and 0 where they
//     disagree by as much as the flux itself (a face whose flux is pure Rhie-Chow contributes nothing).
//     A version that dropped the coefficient entirely passes neither.
//  2. the boundary mask, which is where OF's rule is surprising: ddtScheme::fvcDdtPhiCoeff zeroes the
//     coefficient on fixesValue and on `isA<cyclicAMIFvPatch>` patches -- and cyclicACMIFvPatch derives
//     from coupledFvPatch, NOT from cyclicAMIFvPatch, so an ACMI patch is NOT exempt. Measured: OF's
//     term is 0 on inlet/outlet/walls/blockage and max 1.07e-04 on the two coupled patches.
//  3. sign and scale against a hand-computed value, so a transposed subtraction or a missing 1/deltaT
//     cannot hide behind the ratios in leg 1.
#include "device_ddt.cuh"
#include "device_buffer.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

void near(const char* what, scalar got, scalar want, scalar tol)
{
    if (std::fabs(got - want) <= tol) return;
    std::printf("  FAIL %s: got %.12g, want %.12g (tol %.1e)\n", what, (double)got, (double)want, (double)tol);
    ++failures;
}

}   // namespace

int main()
{
    const scalar dt = 2.5e-3, rDeltaT = scalar(1)/dt;

    // Four faces chosen to land on the two limits and in between:
    //   0: phiCorr == 0        -> coeff 1, but the correction is 0 anyway (the common, quiet case)
    //   1: |phiCorr| = 0.25|phi| -> coeff 0.75      (a real, partial correction)
    //   2: |phiCorr| = |phi|   -> coeff 0           (fully Rhie-Chow: contributes nothing)
    //   3: |phiCorr| > |phi|   -> coeff clipped at 0, NOT negative
    const std::vector<scalar> phiOld   = { 2.0,  4.0,  3.0,  1.0};
    const std::vector<scalar> fluxUold = { 2.0,  3.0,  0.0, -2.0};   // phiCorr = 0, 1, 3, 3
    const std::vector<scalar> rAUf     = {1e-3, 2e-3, 5e-4, 1e-3};
    const std::vector<scalar> ones     = { 1.0,  1.0,  1.0,  1.0};

    DeviceBuffer<scalar> dPhiOld, dFlux, drAUf, dMask, out, none;
    dPhiOld.copyFrom(phiOld);
    dFlux.copyFrom(fluxUold);
    drAUf.copyFrom(rAUf);
    dMask.copyFrom(ones);
    out.copyFrom(std::vector<scalar>(4, 0.0));

    // Driven through the BOUNDARY sweep (nInternalFaces = 0): it is the elementwise one, and it is what
    // the interface path reuses. The internal sweep runs the identical per-face function.
    deviceDdtCorrFlux(0, none, dPhiOld, none, dFlux, none, drAUf, dMask, rDeltaT, none, out);
    const std::vector<scalar> got = out.host();

    // ---- 1 + 3. the coefficient's limits, and the value ----
    for (int f = 0; f < 4; ++f)
    {
        const scalar phiCorr = phiOld[f] - fluxUold[f];
        const scalar coeff = scalar(1) - std::fmin(std::fabs(phiCorr)/(std::fabs(phiOld[f]) + scalar(1e-15)), scalar(1));
        const scalar want = rAUf[f]*coeff*rDeltaT*phiCorr;
        std::printf("  face %d: phi_old %.1f  phiCorr %+.1f  coeff %.4f  ->  %+.6e\n",
                    f, (double)phiOld[f], (double)phiCorr, (double)coeff, (double)got[f]);
        char nm[48];
        std::snprintf(nm, sizeof(nm), "face %d correction", f);
        near(nm, got[f], want, scalar(1e-14));
    }
    // ...and the limits explicitly, so a refactor that quietly dropped the coefficient is caught by name
    // The clipped faces come back at ~1e-16 rather than exactly 0, and that is OpenFOAM's own arithmetic:
    // the coefficient divides by (|phi| + SMALL), so |phiCorr| == |phi| leaves 1 - |phi|/(|phi|+1e-15)
    // instead of a clean 1 - 1. Asserting an exact zero here would be asserting something OF does not do.
    // 1e-12 is far below the 6e-01 a live correction carries, so it still catches a missing clip.
    near("coeff==1 face gives zero correction (phiCorr==0)", got[0], scalar(0), scalar(1e-18));
    near("coeff==0.75 face", got[1], 2e-3*scalar(0.75)*rDeltaT*scalar(1), scalar(1e-14));
    near("coeff clipped to 0 (|phiCorr|==|phi|)", got[2], scalar(0), scalar(1e-12));
    near("coeff clipped to 0, not negative (|phiCorr|>|phi|)", got[3], scalar(0), scalar(1e-12));
    if (got[3] < scalar(-1e-12))
    { std::printf("  FAIL coefficient went NEGATIVE (%.3e): min(...,1) is missing\n", (double)got[3]); ++failures; }
    // VACUITY GUARD: face 1 must be the one that actually exercises a partial coefficient. Without it
    // every assertion above is satisfied by returning zero.
    if (std::fabs(got[1]) < scalar(1e-6))
    {
        std::printf("  FAIL vacuous: no face produced a non-zero correction, so this test cannot tell a\n"
                    "       working ddtCorr from one that does nothing at all\n");
        ++failures;
    }

    // ---- 2. the boundary mask (fixesValue / cyclicAMI -> coefficient 0) ----
    {
        DeviceBuffer<scalar> masked, zeroMask;
        masked.copyFrom(std::vector<scalar>(4, 0.0));
        zeroMask.copyFrom(std::vector<scalar>{0.0, 0.0, 0.0, 0.0});
        deviceDdtCorrFlux(0, none, dPhiOld, none, dFlux, none, drAUf, zeroMask, rDeltaT, none, masked);
        const std::vector<scalar> m = masked.host();
        for (int f = 0; f < 4; ++f)
            near("masked face contributes nothing", m[f], scalar(0), scalar(1e-18));
        std::printf("  mask 0 on every face: no correction anywhere (OF's fixesValue / cyclicAMI rule)\n");
    }

    // ---- accumulation: the kernel ADDS, because it lands on top of fvc::flux(HbyA) ----
    {
        DeviceBuffer<scalar> acc;
        acc.copyFrom(std::vector<scalar>{10.0, 20.0, 30.0, 40.0});
        deviceDdtCorrFlux(0, none, dPhiOld, none, dFlux, none, drAUf, dMask, rDeltaT, none, acc);
        const std::vector<scalar> a = acc.host();
        const std::vector<scalar> base{10.0, 20.0, 30.0, 40.0};
        for (int f = 0; f < 4; ++f)
            near("accumulates onto phiHbyA rather than replacing it", a[f] - base[f], got[f], scalar(1e-14));
        std::printf("  accumulation: += confirmed on all 4 faces\n");
    }

    std::printf("ddt_corr: %d failures\n", failures);
    return failures ? 1 : 0;
}
