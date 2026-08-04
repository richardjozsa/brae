// Gate 0 for the rhoSimpleFoam work: the thermophysical subsystem, on the device.
//
// Everything here is closed-form arithmetic, so the tolerance is machine precision, not physics. If this
// drifts to 1e-8 something is wrong with the EOS, not with a discretisation. The table spans 0.5-10 bar
// and 200-1500 K, which brackets the subsonic compressible cases rhoSimpleFoam is being built for.
//
// The ordering matters: T -> he (startup path) then he -> everything (per-iteration path). Running them
// back to back means a round-trip error in either direction shows up as a temperature mismatch.

#include "thermo_types.cuh"
#include "device_thermo.cuh"
#include "equation_of_state.cuh"
#include "thermo_model.cuh"
#include "transport_model.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

void check(
    const char* what,
    scalar got,
    scalar want,
    scalar relTol)
{
    const scalar denom = std::abs(want) > 0.0 ? std::abs(want) : 1.0;
    const scalar rel = std::abs(got - want) / denom;
    if (rel <= relTol) return;
    std::printf("  FAIL %-28s got %.17g want %.17g  rel %.3e\n", what, got, want, rel);
    failures++;
}

}   // namespace

int main()
{
    ThermoCoeffs c;
    c.R = 287.058;
    c.Cp = 1005.0;
    c.Hf = 0.0;
    c.Pr = 0.7;
    c.As = 1.4792e-06;
    c.Ts = 116.0;
    c.sutherland = true;
    c.rhoMin = 1e-06;
    c.rhoMax = 1e+06;
    c.pMin = 1.0;

    const std::vector<scalar> pressures = {0.5e5, 1.0e5, 3.0e5, 10.0e5};
    const std::vector<scalar> temperatures = {200.0, 300.0, 700.0, 1500.0};

    std::vector<scalar> hp;
    std::vector<scalar> hT;
    for (scalar p : pressures)
    {
        for (scalar T : temperatures)
        {
            hp.push_back(p);
            hT.push_back(T);
        }
    }
    const int n = static_cast<int>(hp.size());

    DeviceThermo th;
    th.allocate(n);
    th.T.copyFrom(hT);

    DeviceBuffer<scalar> p(hp);

    // startup path, then the per-iteration path
    deviceThermoHeFromT(th, c);
    deviceThermoUpdate(th, p, c);

    std::vector<scalar> gT(n);
    std::vector<scalar> gRho(n);
    std::vector<scalar> gPsi(n);
    std::vector<scalar> gMu(n);
    std::vector<scalar> gAlpha(n);
    th.T.copyTo(gT);
    th.rho.copyTo(gRho);
    th.psi.copyTo(gPsi);
    th.mu.copyTo(gMu);
    th.alpha.copyTo(gAlpha);

    const scalar tol = 1e-12;
    for (int i = 0; i < n; ++i)
    {
        const scalar P = hp[i];
        const scalar T = hT[i];
        const scalar wantPsi = 1.0 / (c.R * T);
        const scalar wantRho = P / (c.R * T);
        const scalar wantMu = c.As * std::sqrt(T) / (1.0 + c.Ts / T);

        check("T round trip", gT[i], T, tol);
        check("psi", gPsi[i], wantPsi, tol);
        check("rho", gRho[i], wantRho, tol);
        check("mu (Sutherland)", gMu[i], wantMu, tol);
        check("alpha = mu/Pr", gAlpha[i], wantMu / c.Pr, tol);
        check("rho == psi*p", gRho[i], gPsi[i] * P, tol);
    }

    // const-viscosity branch: mu must ignore T entirely
    c.sutherland = false;
    c.mu0 = 1.8e-05;
    deviceThermoUpdate(th, p, c);
    th.mu.copyTo(gMu);
    for (int i = 0; i < n; ++i)
    {
        check("mu (const branch)", gMu[i], c.mu0, tol);
    }

    // rho bounding must clamp, not merely warn
    c.sutherland = true;
    c.rhoMax = 0.5;
    deviceThermoUpdate(th, p, c);
    th.rho.copyTo(gRho);
    for (int i = 0; i < n; ++i)
    {
        if (gRho[i] <= c.rhoMax) continue;
        std::printf("  FAIL rho not clamped: %.17g > rhoMax %.17g\n", gRho[i], c.rhoMax);
        failures++;
    }

    // rho.relax(): rho = rhoPrev + a*(rho - rhoPrev). Checked at a=1 (must be a no-op) and a=0.3
    // (must land exactly on the blend), because a relaxation that silently does nothing still passes
    // every convergence test -- it just converges slower, which looks like a physics problem.
    {
        c.sutherland = true;
        c.rhoMax = 1e+06;
        deviceThermoUpdate(th, p, c);
        std::vector<scalar> rhoNew(n);
        th.rho.copyTo(rhoNew);

        // a = 1: relaxation must not move rho at all
        deviceRhoSeedPrev(th);
        c.relaxRho = 1.0;
        deviceRhoRelax(th, c);
        std::vector<scalar> got(n);
        th.rho.copyTo(got);
        for (int i = 0; i < n; ++i) check("relax a=1 is a no-op", got[i], rhoNew[i], tol);

        // a = 0.3 against a known previous state: rho must land on rhoPrev + a*(rhoNew - rhoPrev)
        const scalar prev = 0.9;
        th.rhoPrev.copyFrom(std::vector<scalar>(n, prev));
        th.rho.copyFrom(rhoNew);
        c.relaxRho = 0.3;
        deviceRhoRelax(th, c);
        th.rho.copyTo(got);
        for (int i = 0; i < n; ++i)
        {
            check("relax a=0.3 blend", got[i], prev + 0.3 * (rhoNew[i] - prev), tol);
        }

        // rhoPrev must have advanced to the blended value, or the next iteration relaxes against a stale state
        std::vector<scalar> gotPrev(n);
        th.rhoPrev.copyTo(gotPrev);
        for (int i = 0; i < n; ++i) check("relax advances rhoPrev", gotPrev[i], got[i], tol);
    }

    // alphat = rho*nut/Prt. Also checks the laminar guard: with no nut field alphat must stay zero,
    // because the energy equation reads alphaEff = alpha + alphat unconditionally and a stray alphat
    // would quietly add turbulent diffusion to a laminar solve.
    {
        c.Prt = 0.85;
        c.rhoMax = 1e+06;
        deviceThermoUpdate(th, p, c);
        std::vector<scalar> rhoNow(n);
        th.rho.copyTo(rhoNow);

        DeviceBuffer<scalar> nut(std::vector<scalar>(n, 1.5e-03));
        deviceAlphat(th, nut, c);
        std::vector<scalar> gotAt(n);
        th.alphat.copyTo(gotAt);
        for (int i = 0; i < n; ++i)
        {
            check("alphat = rho*nut/Prt", gotAt[i], rhoNow[i] * 1.5e-03 / 0.85, tol);
        }

        // laminar: an empty nut must leave alphat untouched at zero
        th.alphat.copyFrom(std::vector<scalar>(n, 0.0));
        DeviceBuffer<scalar> noNut;
        deviceAlphat(th, noNut, c);
        th.alphat.copyTo(gotAt);
        for (int i = 0; i < n; ++i) check("alphat stays 0 when laminar", gotAt[i], 0.0, tol);
    }

    std::printf("thermo_perfect_gas: %d states, %d failures\n", n, failures);
    return failures == 0 ? 0 : 1;
}
