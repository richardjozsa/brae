// CrankNicolson ddt: order-of-accuracy validation via a manufactured ODE. Integrate dpsi/dt = -lambda*psi (exact
// psi(T)=exp(-lambda*T)) on a 1-cell "mesh" using the REAL device ddt wrappers -- each step assembles
// (fvm::ddt(psi) + fvm::Sp(lambda,psi)) == 0, i.e. (ddtDiag + lambda*V)*psi_new = ddtSource, with the CrankNicolson
// ddt0 recurrence updated each step. Checks: Euler is 1st order, backward + CrankNicolson are 2nd order (error /4 when
// dt halves), and CrankNicolson is much more accurate than Euler at the same dt. This exercises deviceFvmDdtDiag /
// deviceFvmDdtSource / deviceFvmDdtUpdateDdt0 + the ddtCoeffs() CrankNicolson branch end-to-end.
#include "device_ddt.cuh"
#include "device_buffer.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

// Integrate to time T in nSteps using `scheme`; return psi(T). Backward-Euler bootstrap on the first step (dt0=0).
static double integrate(DdtScheme scheme, double lambda, double T, int nSteps, double oc)
{
    const double dt = T / nSteps;
    DeviceBuffer<scalar> V;    V.copyFrom(std::vector<scalar>{scalar(1)});
    DeviceBuffer<scalar> psiOld, psiOld2, ddt0, diag, source;
    ddt0.copyFrom(std::vector<scalar>{scalar(0)});           // ddt0 starts at zero (OF)
    double psi = 1.0, hOld = 1.0, hOld2 = 0.0, dt0 = 0.0;
    bool ddt0Updated = false;                                // false until the first ddt0 recurrence (OF coef0_ Euler startup)
    for (int step = 0; step < nSteps; ++step)
    {
        hOld2 = hOld; hOld = psi;                            // time advance: rotate the old levels
        const bool haveOld2 = (step >= 1);
        psiOld.copyFrom(std::vector<scalar>{scalar(hOld)});
        if (haveOld2) psiOld2.copyFrom(std::vector<scalar>{scalar(hOld2)}); else psiOld2.resize(0);
        const DdtCoeffs c = ddtCoeffs(scheme, dt, dt0, oc, ddt0Updated);   // dt0=0 on step 0 -> Euler; cnWarm=false on first ddt0 update
        deviceFvmDdtUpdateDdt0(c, psiOld, psiOld2, ddt0);    // CrankNicolson ddt0 recurrence (no-op otherwise)
        if (c.cn && haveOld2) ddt0Updated = true;            // first CN ddt0 update done -> subsequent updates use 1+oc
        diag.copyFrom(std::vector<scalar>{scalar(0)});
        source.copyFrom(std::vector<scalar>{scalar(0)});
        deviceFvmDdtDiag(V, c, 1.0, diag);                  // ddt diagonal
        deviceFvmDdtSource(V, c, 1.0, psiOld, psiOld2, source, &ddt0);   // ddt source (+ ddt0 for CN)
        const double d = diag.host()[0] + lambda * 1.0;     // + fvm::Sp(lambda, psi): implicit -lambda*psi
        psi = source.host()[0] / d;                          // solve the 1-cell implicit system
        dt0 = dt;
    }
    return psi;
}

int main()
{
    int fails = 0;
    auto chk = [&](const char* nm, bool ok, const char* extra) {
        if (!ok) ++fails;
        std::printf("  %-46s %s   %s\n", nm, ok ? "OK" : "FAIL", extra);
    };
    const double lambda = 2.0, T = 1.0, exact = std::exp(-lambda * T);

    auto order = [&](DdtScheme s, double oc) {
        const double e1 = std::fabs(integrate(s, lambda, T, 100,  oc) - exact);
        const double e2 = std::fabs(integrate(s, lambda, T, 200,  oc) - exact);
        return std::log2(e1 / e2);   // observed convergence order (2 -> error /4 when dt halves)
    };
    const double oE = order(DdtScheme::Euler,        1.0);
    const double oB = order(DdtScheme::backward,     1.0);
    const double oC = order(DdtScheme::CrankNicolson,1.0);
    char b[96];
    std::printf("observed convergence orders (exact=exp(-lambda T)=%.8f):\n", exact);
    std::snprintf(b, sizeof b, "order=%.3f", oE); chk("Euler is 1st order (~1.0)",         std::fabs(oE - 1.0) < 0.15, b);
    std::snprintf(b, sizeof b, "order=%.3f", oB); chk("backward is 2nd order (~2.0)",      std::fabs(oB - 2.0) < 0.2,  b);
    std::snprintf(b, sizeof b, "order=%.3f", oC); chk("CrankNicolson is 2nd order (~2.0)", std::fabs(oC - 2.0) < 0.2,  b);

    const double eE = std::fabs(integrate(DdtScheme::Euler,         lambda, T, 100, 1.0) - exact);
    const double eC = std::fabs(integrate(DdtScheme::CrankNicolson, lambda, T, 100, 1.0) - exact);
    std::snprintf(b, sizeof b, "Euler err=%.2e  CN err=%.2e", eE, eC);
    chk("CrankNicolson >> more accurate than Euler", eC < 0.1 * eE, b);
    // first-step bootstrap sanity: 1 step of CN must equal 1 step of Euler (dt0=0 -> Euler), both = 1/(1+lambda*dt)
    const double s1c = integrate(DdtScheme::CrankNicolson, lambda, T, 1, 1.0);
    const double s1e = integrate(DdtScheme::Euler,         lambda, T, 1, 1.0);
    std::snprintf(b, sizeof b, "CN=%.10f Euler=%.10f", s1c, s1e);
    chk("first step bootstraps to Euler (dt0=0)", std::fabs(s1c - s1e) < 1e-12, b);

    std::printf("\n%s: %d failure(s)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
