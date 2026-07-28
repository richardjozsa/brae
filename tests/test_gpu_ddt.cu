// Transient ddt (fvm::ddt) value gate: brae's deviceFvmDdt reproduces OpenFOAM-2412 EulerDdtScheme / backwardDdtScheme
// EXACTLY on a fixed mesh. Checks (a) the scheme coefficients vs the OF formula (backwardDdtScheme.C:96-98), (b) the
// device diag/source vs hand-worked OF values (EulerDdtScheme.C:383/391, backwardDdtScheme.C:26/39-43), (c) the
// end-to-end BDF2 time derivative (diag*psi - source)/V == (1.5 psi - 2 psi_old + 0.5 psi_old2)/dt, (d) steadyState is a
// no-op (SIMPLE unaffected). No mesh needed -- ddt is per-cell, so a hand array IS the ground truth.
#include "device_ddt.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

int main()
{
    int fails = 0;
    auto chk = [&](const char* nm, double got, double exp) {
        const bool ok = std::fabs(got - exp) <= 1e-12 * std::fmax(1.0, std::fabs(exp));
        if (!ok) ++fails;
        std::printf("  %-34s got %.10g  exp %.10g  %s\n", nm, got, exp, ok ? "OK" : "FAIL");
    };
    auto dbuf = [](const std::vector<scalar>& h) { DeviceBuffer<scalar> b; b.copyFrom(h); return b; };

    // ---- (a) coefficients vs OF (backwardDdtScheme.C:96-98) --------------------------------------------------------
    std::printf("ddtCoeffs vs OF formula:\n");
    {
        DdtCoeffs e = ddtCoeffs(DdtScheme::Euler, 0.1, 0.1);
        chk("Euler active",   e.active,   1);
        chk("Euler rDeltaT",  e.rDeltaT, 10);
        chk("Euler coefft",   e.coefft,   1);
        chk("Euler coefft0",  e.coefft0,  1);
        chk("Euler coefft00", e.coefft00, 0);

        DdtCoeffs b = ddtCoeffs(DdtScheme::backward, 0.5, 0.5);   // constant dt -> textbook BDF2 (1.5, 2, 0.5)
        chk("backward(dt=dt0) coefft",   b.coefft,   1.5);
        chk("backward(dt=dt0) coefft0",  b.coefft0,  2.0);
        chk("backward(dt=dt0) coefft00", b.coefft00, 0.5);

        DdtCoeffs bv = ddtCoeffs(DdtScheme::backward, 0.1, 0.05);  // variable dt
        chk("backward(var) coefft",   bv.coefft,   1.0 + 0.1/0.15);
        chk("backward(var) coefft00", bv.coefft00, 0.01/(0.05*0.15));
        chk("backward(var) coefft0",  bv.coefft0,  (1.0 + 0.1/0.15) + 0.01/(0.05*0.15));

        DdtCoeffs boot = ddtCoeffs(DdtScheme::backward, 0.1, 0.0); // first step: no 2nd level -> Euler bootstrap
        chk("backward bootstrap coefft",   boot.coefft,   1);
        chk("backward bootstrap coefft00", boot.coefft00, 0);

        DdtCoeffs s = ddtCoeffs(DdtScheme::steadyState, 0.1, 0.1);
        chk("steadyState inactive", s.active, 0);
    }

    // ---- (b) device diag/source vs hand-worked OF values ----------------------------------------------------------
    // Euler, dt=0.5 (rDeltaT=2), rho=1, V=[2,3], Uold=[1,4], accumulating onto a pre-existing (div+visc) matrix.
    std::printf("deviceFvmDdt Euler (accumulates onto assembled matrix):\n");
    {
        auto V = dbuf({2, 3}), Uold = dbuf({1, 4}), Uold2 = dbuf({});
        auto diag = dbuf({10, 10}), src = dbuf({100, 100});
        deviceFvmDdt(V, ddtCoeffs(DdtScheme::Euler, 0.5, 0.5), /*rho*/1.0, Uold, Uold2, diag, src);
        auto d = diag.host(), s = src.host();
        chk("Euler diag[0] 10+2*2",     d[0], 14);   // 10 + rDeltaT*V
        chk("Euler diag[1] 10+2*3",     d[1], 16);
        chk("Euler src[0]  100+2*2*1",  s[0], 104);  // 100 + rDeltaT*V*Uold
        chk("Euler src[1]  100+2*3*4",  s[1], 124);
    }
    // backward, dt=dt0=0.5, rho=1, V=[2,3], Uold=[1,4], Uold2=[0.5,2], onto a zero matrix.
    std::printf("deviceFvmDdt backward:\n");
    {
        auto V = dbuf({2, 3}), Uold = dbuf({1, 4}), Uold2 = dbuf({0.5, 2});
        auto diag = dbuf({0, 0}), src = dbuf({0, 0});
        deviceFvmDdt(V, ddtCoeffs(DdtScheme::backward, 0.5, 0.5), 1.0, Uold, Uold2, diag, src);
        auto d = diag.host(), s = src.host();
        chk("backward diag[0] 1.5*2*2", d[0], 6);
        chk("backward diag[1] 1.5*2*3", d[1], 9);
        chk("backward src[0] 2*2*(2-.25)", s[0], 7);    // rDeltaT*V*(coefft0*Uold - coefft00*Uold2)
        chk("backward src[1] 2*3*(8-1)",   s[1], 42);
    }
    // rho != 1 (constant-density path, fvmDdt(rho,vf)), single cell.
    std::printf("deviceFvmDdt backward rho=2:\n");
    {
        auto V = dbuf({2}), Uold = dbuf({3}), Uold2 = dbuf({1});
        auto diag = dbuf({0}), src = dbuf({0});
        deviceFvmDdt(V, ddtCoeffs(DdtScheme::backward, 0.5, 0.5), 2.0, Uold, Uold2, diag, src);
        chk("rho2 diag 1.5*2*2*2", diag.host()[0], 12);
        chk("rho2 src 8*(6-0.5)",  src.host()[0], 44);
    }

    // ---- (c) end-to-end BDF2 derivative: (diag*psi - source)/V == (1.5 psi - 2 psi_old + 0.5 psi_old2)/dt ----------
    std::printf("BDF2 operator value (diag*psi - source)/V:\n");
    {
        const scalar dt = 0.5, psi = 5, psi0 = 3, psi00 = 1, Vc = 2;
        auto V = dbuf({Vc}), Uold = dbuf({psi0}), Uold2 = dbuf({psi00});
        auto diag = dbuf({0}), src = dbuf({0});
        deviceFvmDdt(V, ddtCoeffs(DdtScheme::backward, dt, dt), 1.0, Uold, Uold2, diag, src);
        const scalar ddtVal = (diag.host()[0] * psi - src.host()[0]) / Vc;
        chk("BDF2 (1.5*5-2*3+0.5*1)/0.5", ddtVal, (1.5 * psi - 2.0 * psi0 + 0.5 * psi00) / dt);
    }

    // ---- (d) steadyState / inactive -> exact no-op (byte-for-byte SIMPLE) ------------------------------------------
    std::printf("steadyState no-op:\n");
    {
        auto V = dbuf({2, 3}), Uold = dbuf({1, 4}), Uold2 = dbuf({});
        auto diag = dbuf({7, 8}), src = dbuf({9, 11});
        deviceFvmDdt(V, ddtCoeffs(DdtScheme::steadyState, 0.5, 0.5), 1.0, Uold, Uold2, diag, src);
        auto d = diag.host(), s = src.host();
        chk("steady diag[0] unchanged", d[0], 7);
        chk("steady diag[1] unchanged", d[1], 8);
        chk("steady src[0] unchanged",  s[0], 9);
        chk("steady src[1] unchanged",  s[1], 11);
    }

    std::printf("\n%s: %d failure(s)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
