// fvc::ddtCorr for the `backward` ddt scheme -- OF backwardDdtScheme::fvcDdtUfCorr.
//
// 13 of OpenFOAM's 35 pimpleFoam tutorials select `backward`, and brae omitted the flux correction
// entirely for every one of them (a notice, then nothing). The implicit cell ddt was there; the
// pressure/flux coupling was not, which is a different equation on a third of the suite.
//
// OpenFOAM's form, term for term:
//
//     coefft   = 1 + dt/(dt + dt0)
//     coefft00 = dt*dt/(dt0*(dt + dt0))
//     coefft0  = coefft + coefft00
//
//     ddtCorr = fvcDdtPhiCoeff(U_o, Sf&Uf_o) * rDeltaT
//             * [ (coefft0*phi_o - coefft00*phi_oo) - (coefft0*fluxU_o - coefft00*fluxU_oo) ]
//
// THE ASYMMETRY IS THE POINT, and Leg 2 is what pins it. The coupling COEFFICIENT is built from the
// SINGLE old level -- fvcDdtPhiCoeff(U.oldTime(), Sf & Uf.oldTime()) -- while the quantity it multiplies
// is the two-level combination. The coefficient is a limiter (1 - min(|phiCorr|/|phi|, 1)) that switches
// the correction off where the flux and the interpolated velocity disagree strongly; feeding it the
// combined correction instead changes WHERE it switches off, not just what it scales. An implementation
// that computes both from the same array passes every other leg here.
#include "box_mesh.cuh"
#include "device_buffer.cuh"
#include "device_ddt.cuh"
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

// OF ddtScheme::fvcDdtPhiCoeff, the ddtPhiCoeff_ < 0 branch (the only one a v2412 case reaches).
scalar ofCoeff(scalar phiOld, scalar fluxUold)
{
    const scalar phiCorr = phiOld - fluxUold;
    return scalar(1) - std::min(std::fabs(phiCorr)/(std::fabs(phiOld) + scalar(1e-15)), scalar(1));
}
} // namespace

int main()
{
    std::printf("== fvc::ddtCorr, backward scheme ==\n");

    PrimitiveMesh m = boxtest::boxMesh(4, 3, 2);
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const int nIf = (int)m.nInternalFaces();

    // ---- Leg 1: the coefficients themselves, against OF's three lines ------------------------------
    {
        const scalar dt = 0.004, dt0 = 0.003;
        const DdtCoeffs c = ddtCoeffs(DdtScheme::backward, dt, dt0, 0.0, false);
        const scalar coefft   = 1 + dt/(dt + dt0);
        const scalar coefft00 = dt*dt/(dt0*(dt + dt0));
        const scalar coefft0  = coefft + coefft00;
        check(std::fabs(c.coefft   - coefft)   < 1e-14, "coefft   = 1 + dt/(dt + dt0)");
        check(std::fabs(c.coefft00 - coefft00) < 1e-14, "coefft00 = dt^2/(dt0*(dt + dt0))");
        check(std::fabs(c.coefft0  - coefft0)  < 1e-14, "coefft0  = coefft + coefft00");
        check(coefft00 > 0.1, "vacuity guard: coefft00 is not incidentally zero at this dt/dt0");
    }

    // ---- Leg 2: first transient step falls back to Euler -------------------------------------------
    // OF has no oldTime().oldTime() yet, so coefft00 is zero and the scheme IS Euler on step one. brae
    // uses exactly that to decide whether the two-level correction can be formed at all.
    {
        const DdtCoeffs c = ddtCoeffs(DdtScheme::backward, 0.004, 0.0, 0.0, false);
        check(c.coefft00 == scalar(0), "deltaT0 = 0: coefft00 is zero, i.e. Euler on the first step");
    }

    // ---- Leg 3: the kernel applies the SINGLE-level coefficient to the supplied correction ---------
    const scalar dt = 0.004, dt0 = 0.003, rDeltaT = 1.0/dt;
    const DdtCoeffs c = ddtCoeffs(DdtScheme::backward, dt, dt0, 0.0, false);

    std::vector<scalar> hPhiO(nIf), hPhiOO(nIf), hFluxO(nIf), hFluxOO(nIf), hRAUf(nIf, 0.5);
    for (int f = 0; f < nIf; ++f)
    {
        // Deliberately varied so the coefficient's limiter is ACTIVE on some faces and not others: it
        // saturates to 0 wherever |phiCorr| >= |phi|, and a test where it is 1 everywhere would not
        // notice the coefficient being computed from the wrong array.
        hPhiO[f]   = 1.0 + 0.30*f;
        hPhiOO[f]  = 0.8 + 0.25*f;
        hFluxO[f]  = 0.9 + 0.31*f;
        hFluxOO[f] = 0.7 + 0.24*f;
    }
    std::vector<scalar> hEff(nIf);
    for (int f = 0; f < nIf; ++f)
        hEff[f] = (c.coefft0*hPhiO[f] - c.coefft00*hPhiOO[f])
                - (c.coefft0*hFluxO[f] - c.coefft00*hFluxOO[f]);

    DeviceBuffer<scalar> phiO, fluxO, rAUf, eff, out, emptyB, maskB;
    phiO.copyFrom(hPhiO); fluxO.copyFrom(hFluxO); rAUf.copyFrom(hRAUf); eff.copyFrom(hEff);
    out.copyFrom(std::vector<scalar>(nIf, scalar(0)));
    deviceDdtCorrFlux(nIf, phiO, emptyB, fluxO, emptyB, rAUf, emptyB, maskB, rDeltaT, out, emptyB,
                      &eff, nullptr);
    std::vector<scalar> got; out.copyTo(got);

    {
        scalar worst = 0;
        int limited = 0, unlimited = 0;
        for (int f = 0; f < nIf; ++f)
        {
            const scalar k = ofCoeff(hPhiO[f], hFluxO[f]);          // SINGLE old level
            const scalar want = hRAUf[f]*k*rDeltaT*hEff[f];
            worst = std::max(worst, std::fabs(got[f] - want)/std::max(std::fabs(want), scalar(1e-30)));
            if (k < 0.999) ++limited; else ++unlimited;
        }
        check(worst < 1e-13, "backward ddtCorr = coeff(single level) * rAUf * rDeltaT * two-level correction");
        check(limited > 0, "vacuity guard: the coefficient limiter is active on some faces");
    }

    // ---- Leg 4: DISCRIMINATION -- the coefficient must not come from the combined correction --------
    // Recomputing it from `eff` would give a different limiter on exactly the faces where the two-level
    // combination and the single-level phiCorr disagree, which is most of them here.
    {
        scalar maxGap = 0;
        for (int f = 0; f < nIf; ++f)
        {
            const scalar kSingle = ofCoeff(hPhiO[f], hFluxO[f]);
            const scalar kWrong  = scalar(1) - std::min(std::fabs(hEff[f])
                                 / (std::fabs(c.coefft0*hPhiO[f] - c.coefft00*hPhiOO[f]) + scalar(1e-15)),
                                   scalar(1));
            maxGap = std::max(maxGap, std::fabs(kSingle - kWrong));
        }
        check(maxGap > 1e-3,
              "vacuity guard: the two coefficient choices genuinely differ, so leg 3 discriminates");
    }

    // ---- Leg 5: with no correction supplied, the Euler form is unchanged ---------------------------
    // Passing nullptr must reproduce phiCorr = phi_o - fluxU_o exactly -- Euler cases share this kernel.
    {
        DeviceBuffer<scalar> out2;
        out2.copyFrom(std::vector<scalar>(nIf, scalar(0)));
        deviceDdtCorrFlux(nIf, phiO, emptyB, fluxO, emptyB, rAUf, emptyB, maskB, rDeltaT, out2, emptyB,
                          nullptr, nullptr);
        std::vector<scalar> g2; out2.copyTo(g2);
        scalar worst = 0;
        for (int f = 0; f < nIf; ++f)
        {
            const scalar phiCorr = hPhiO[f] - hFluxO[f];
            const scalar want = hRAUf[f]*ofCoeff(hPhiO[f], hFluxO[f])*rDeltaT*phiCorr;
            worst = std::max(worst, std::fabs(g2[f] - want)/std::max(std::fabs(want), scalar(1e-30)));
        }
        check(worst < 1e-13, "negative control: nullptr keeps the Euler form bit-for-bit");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
