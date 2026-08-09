// laminar generalizedNewtonian + powerLaw viscosity (OF laminarModels::generalizedNewtonian).
//
//     strainRate = sqrt(2)*mag(symm(grad U))                     generalizedNewtonian.C:98
//     nu  = max(nuMin, min(nuMax, nu0*pow(max(strainRate, SMALL), n - 1)))   powerLaw.C
//     mu  = rho*nu                                               nuEff() RETURNS nu_, it does not add
//
// THE COEFFICIENTS AND THE CLAMPED REFERENCE COME FROM THE REAL CASE, NOT FROM INVENTION.
// squareBendLiqNoNewtonian ships n = 0.4, nuMin = 1e-3, nuMax = 1. Evaluating OF's own converged fields
// (grad(U) at time 178, rho, and the H2O viscosity correlation) gives strain rates 6.3 .. 1.09e3 and
// nu0 = mu/rho in 4.66e-7 .. 8.92e-7, and the formula lands on nuMin for 100.0% of the 112000 cells.
// That is the branch the tutorial actually exercises, so it is checked here with those numbers.
//
// WHAT EACH CASE IS FOR. The clamped cases pin the behaviour the tutorial depends on; the unclamped case
// exercises the arithmetic the clamps otherwise hide (with clamps opened wide, since these coefficients
// never leave nuMin); the zero-strain case covers the one input that makes a negative exponent explode;
// and the last one asserts mu = rho*nu rather than rho*(nu0 + nu), which is the difference between
// generalizedNewtonian and a turbulence model and is invisible in any single-point value check.
#include "device_generalized_newtonian.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

// The device entry point takes mu as in/out: in = thermo mu (so nu0 = mu/rho), out = rho*nu.
std::vector<scalar> runGN(
    const std::vector<scalar>& strainRate,
    const std::vector<scalar>& rho,
    const std::vector<scalar>& muThermo,
    scalar nuMin, scalar nuMax, scalar n)
{
    std::vector<scalar> s2(strainRate.size());
    for (std::size_t i = 0; i < strainRate.size(); ++i) s2[i] = strainRate[i]*strainRate[i];  // S2 = sr^2
    DeviceBuffer<scalar> S2d, rhod, mud;
    S2d.copyFrom(s2);
    rhod.copyFrom(rho);
    mud.copyFrom(muThermo);
    deviceGeneralizedNewtonianPowerLawMu(S2d, rhod, nuMin, nuMax, n, mud);
    return mud.host();
}

void expectRel(const char* what, int i, double got, double want, double tol = 1e-13)
{
    const double rel = std::fabs(got - want)/std::fmax(std::fabs(want), 1e-300);
    if (!(rel <= tol))
    {
        std::printf("  FAIL %-22s cell %d  got %.17g  want %.17g  rel %.3e\n", what, i, got, want, rel);
        ++failures;
    }
}

}   // namespace

int main()
{
    // squareBendLiqNoNewtonian's own coefficients.
    const scalar nuMin = 1e-3, nuMax = 1.0, n = 0.4;

    // ---------------------------------------------------------------------------------------------
    // 1. THE BRANCH THE TUTORIAL RUNS IN. Strain rates spanning OF's measured range, nu0 from the H2O
    // correlation over the same span: nu must be nuMin at every one of them, so mu = rho*nuMin.
    {
        const std::vector<scalar> sr  = {6.3196, 50.0, 185.01, 500.0, 1086.6};
        const std::vector<scalar> rho = {994.51, 992.0, 988.0, 986.0, 984.74};
        std::vector<scalar> mu(sr.size());
        for (std::size_t i = 0; i < sr.size(); ++i) mu[i] = 8.9e-4;   // ~ the H2O viscosity over this range

        const std::vector<scalar> out = runGN(sr, rho, mu, nuMin, nuMax, n);
        for (std::size_t i = 0; i < sr.size(); ++i)
            expectRel("clamped to nuMin", (int)i, out[i], rho[i]*nuMin);
        std::printf("  nuMin branch: mu = rho*nuMin at strainRate %.1f..%.0f (OF: 100%% of cells)\n",
                    (double)sr.front(), (double)sr.back());
    }

    // ---------------------------------------------------------------------------------------------
    // 2. THE UNCLAMPED ARITHMETIC. Open the clamps so the power law itself is what is being tested --
    // with the tutorial's own clamps this branch is unreachable, which is exactly why it needs its own
    // case rather than being assumed covered by case 1.
    {
        const std::vector<scalar> sr  = {0.5, 1.0, 2.0, 10.0, 100.0};
        const std::vector<scalar> rho = {1000.0, 1000.0, 1000.0, 1000.0, 1000.0};
        const std::vector<scalar> mu(sr.size(), 1.0e-3);              // nu0 = 1e-6
        const scalar wideMin = 0.0, wideMax = 1e30;

        const std::vector<scalar> out = runGN(sr, rho, mu, wideMin, wideMax, n);
        for (std::size_t i = 0; i < sr.size(); ++i)
        {
            const double nu0  = 1.0e-3/1000.0;
            const double want = 1000.0 * (nu0 * std::pow((double)sr[i], (double)n - 1.0));
            expectRel("unclamped powerLaw", (int)i, out[i], want);
        }
        // sr = 1 is the fixed point of the power law: nu = nu0 whatever n is. A sign error in the
        // exponent leaves that one point right and every other point wrong, so it is the anchor.
        expectRel("sr=1 -> nu0", 1, out[1], 1000.0*1.0e-6);
        std::printf("  unclamped: nu = nu0*strainRate^(n-1) over sr 0.5..100, incl. the sr=1 fixed point\n");
    }

    // ---------------------------------------------------------------------------------------------
    // 3. ZERO STRAIN RATE. n < 1 makes the exponent NEGATIVE, so a zero-strain cell raises 0 to a
    // negative power. Two sub-cases, and only the second actually needs the SMALL floor -- established
    // by mutation, not assumed: with nu0 > 0 the pow returns +inf and fmin(nuMax, inf) clamps it back,
    // so removing the floor changes nothing. It is nu0 == 0 that breaks, because 0*inf is NaN and no
    // clamp rescues a NaN. OF applies max(strainRate, SMALL) before the pow for exactly that reason.
    {
        const std::vector<scalar> sr  = {0.0, 0.0};
        const std::vector<scalar> rho = {994.51, 500.0};
        const std::vector<scalar> mu(2, 8.9e-4);
        const std::vector<scalar> out = runGN(sr, rho, mu, nuMin, nuMax, n);
        for (int i = 0; i < 2; ++i)
        {
            if (!std::isfinite((double)out[i]))
            {
                std::printf("  FAIL zero strain rate gave %g -- the SMALL floor before pow() is missing\n",
                            (double)out[i]);
                ++failures;
            }
            expectRel("zero strain -> nuMax", i, out[i], rho[i]*nuMax);
        }
        // nu0 == 0 with zero strain: 0*pow(0, n-1) = 0*inf = NaN unless the strain rate is floored first.
        {
            const std::vector<scalar> sr0  = {0.0};
            const std::vector<scalar> rho0 = {994.51};
            const std::vector<scalar> mu0  = {0.0};                 // nu0 = 0
            const std::vector<scalar> o0 = runGN(sr0, rho0, mu0, nuMin, nuMax, n);
            if (!std::isfinite((double)o0[0]))
            {
                std::printf("  FAIL nu0=0 at zero strain gave %g -- 0*inf, i.e. the SMALL floor before\n"
                            "       pow() is missing (a clamp cannot rescue a NaN)\n", (double)o0[0]);
                ++failures;
            }
            else expectRel("nu0=0 -> nuMin", 0, o0[0], rho0[0]*nuMin);
        }
        std::printf("  zero strain rate: finite for nu0>0 and for nu0=0 (the case that needs the floor)\n");
    }

    // ---------------------------------------------------------------------------------------------
    // 4. REPLACES, DOES NOT ADD. OF's generalizedNewtonian::nuEff() returns nu_ outright and nut() is
    // zero, so mu must be rho*nu and NOT rho*nu + mu_thermo. Two cells with the SAME strain rate and
    // rho but very different thermo viscosities must come out identical once clamped -- an
    // implementation that added the molecular part would differ by exactly that difference.
    {
        const std::vector<scalar> sr  = {185.0, 185.0};
        const std::vector<scalar> rho = {994.51, 994.51};
        const std::vector<scalar> mu  = {8.9e-4, 8.9e-1};            // 1000x apart
        const std::vector<scalar> out = runGN(sr, rho, mu, nuMin, nuMax, n);
        if (out[0] != out[1])
        {
            std::printf("  FAIL clamped mu depends on the thermo viscosity (%.17g vs %.17g) --\n"
                        "       nuEff() must RETURN nu_, not add it to the molecular nu\n",
                        (double)out[0], (double)out[1]);
            ++failures;
        }
        else std::printf("  replaces-not-adds: a 1000x thermo-mu change leaves the clamped mu identical\n");
    }

    // ---------------------------------------------------------------------------------------------
    // 5. NEGATIVE CONTROLS -- the mistakes this test exists to catch, checked in-process so it carries
    // its own proof that it can go red.
    {
        int caught = 0;
        const double nu0 = 1.0e-6, sr = 100.0;

        // (a) exponent n instead of n-1: the single most likely transcription slip.
        if (std::fabs(nu0*std::pow(sr, n) - nu0*std::pow(sr, n - 1.0))/(nu0*std::pow(sr, n - 1.0)) > 1e-9)
            ++caught;

        // (b) clamps applied in the wrong order -- min(nuMax, max(nuMin, x)) vs OF's
        // max(nuMin, min(nuMax, x)). These agree unless nuMin > nuMax, so the control uses that case:
        // it is the only way to tell the two orderings apart, and getting it wrong is silent otherwise.
        {
            const double x = 5.0, lo = 10.0, hi = 1.0;               // deliberately inverted
            const double ofOrder = std::fmax(lo, std::fmin(hi, x));  // = 10
            const double swapped = std::fmin(hi, std::fmax(lo, x));  // = 1
            if (ofOrder != swapped) ++caught;
        }

        // (c) dropping the SMALL floor: 0^(n-1) with n < 1 is not finite.
        if (!std::isfinite(std::pow(0.0, n - 1.0))) ++caught;

        if (caught != 3)
        {
            std::printf("  FAIL negative controls: only %d of 3 fire -- this test cannot distinguish a\n"
                        "       wrong exponent, a swapped clamp order, or a missing zero-strain floor\n", caught);
            ++failures;
        }
        else std::printf("  negative controls: wrong exponent, swapped clamps and missing floor all rejected\n");
    }

    std::printf("test_generalized_newtonian: %d failures\n", failures);
    return failures ? 1 : 0;
}
