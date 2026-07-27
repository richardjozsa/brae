// SA-DDES length-scale gate: deviceSADDESdTilda reproduces the OF SpalartAllmarasDDES delayed length scale
//   dTilda = y - fd*max(0, y - Psi*CDES*Delta),  Delta = cubeRootVol = V^(1/3),
//   fd = 1 - tanh((8 rd)^3),  rd = min((nut+nu)/(|gradU| kappa^2 y^2), 10),  nut = nuTilda*fv1,
//   Psi = sqrt(min(100, (1 - Cb1/(Cw1 kappa^2 fwStar)*fv2)/fv1))  (low-Re correction, ft2=off).
// Checks (a) the device kernel matches a host re-implementation of the exact formula (catches indexing/math bugs), and
// (b) the physical LIMITS by hand: RANS fd->0 => dTilda=y; LES fd->1 => dTilda=min(y, Psi*CDES*Delta); Psi->1 at high
// chi and Psi>1 at low chi (the near-wall/low-Re length-scale enlargement that keeps the model RANS there).
#include "device_kepsilon.cuh"   // deviceSADDESdTilda
#include "spalart_coeffs.cuh"    // SpalartAllmarasCoeffs
#include "device_buffer.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

static double hostPsi(double chi, const SpalartAllmarasCoeffs& co)
{
    const double chi3 = chi*chi*chi, Cv13 = co.Cv1*co.Cv1*co.Cv1;
    const double fv1 = chi3/(chi3 + Cv13);
    const double fv2 = 1.0 - chi/(1.0 + chi*fv1);
    const double K = co.Cb1/(co.Cw1()*co.kappa*co.kappa*co.fwStar);
    return std::sqrt(std::fmax(std::fmin(100.0, (1.0 - K*fv2)/std::fmax(fv1, 1e-300)), 0.0));
}

// host re-implementation of the exact kernel formula (component-major gradU: g9[k] = the 9 tensor components for a cell)
static double hostDTilda(double y, double V, const double g9[9], double nuTilda, double nu, const SpalartAllmarasCoeffs& co)
{
    const double delta = std::cbrt(V);
    double g2 = 0; for (int k = 0; k < 9; ++k) g2 += g9[k]*g9[k];
    const double chi = nuTilda/nu, chi3 = chi*chi*chi, Cv13 = co.Cv1*co.Cv1*co.Cv1;
    const double nut = nuTilda * (chi3/(chi3 + Cv13));
    const double kd2 = std::fmax(co.kappa*co.kappa*y*y, 1e-300);
    const double rd = std::fmin((nut + nu)/(std::fmax(std::sqrt(g2), 1e-300)*kd2), 10.0);
    const double t8 = 8.0*rd, fd = 1.0 - std::tanh(t8*t8*t8);
    return y - fd*std::fmax(0.0, y - hostPsi(chi, co)*co.CDES*delta);
}

int main()
{
    int fails = 0;
    auto chk = [&](const char* nm, double got, double exp, double tol) {
        const bool ok = std::fabs(got - exp) <= tol*std::fmax(1.0, std::fabs(exp));
        if (!ok) ++fails;
        std::printf("  %-40s got %.10g  exp %.10g  %s\n", nm, got, exp, ok ? "OK" : "FAIL");
    };

    const SpalartAllmarasCoeffs co;   // OF defaults: kappa=0.41, Cv1=7.1, CDES=0.65, fwStar=0.424
    const double nu = 1e-5;
    // 5 cells: [0]=RANS(fd~0); [1]=LES high-chi (Psi~1) y>lLES; [2]=LES y<lLES; [3]=intermediate rd;
    // [4]=LES LOW-chi (Psi>1) so lLES = Psi*CDES*Delta is enlarged. V=1 -> Delta=1.
    struct Cell { double y, V, gmag, nuTilda; } cells[] = {
        {0.001, 1.0, 1.0,   0.1},     // small y -> rd huge (>=10) -> fd~0 -> dTilda=y
        {2.0,   1.0, 1e8,   0.1},     // chi=1e4 (Psi~1), fd~1, y>lLES -> dTilda=Psi*CDES*Delta ~ 0.65
        {0.3,   1.0, 1e8,   0.1},     // fd~1 but y<lLES -> dTilda=y=0.3
        {1.0,   1.0, 5.949, 0.1},     // intermediate rd -> fd in (0,1)
        {2.0,   1.0, 1e8,   1e-4},    // chi=10 (Psi>1), fd~1, y>lLES -> dTilda=Psi*CDES*Delta (>0.65)
    };
    const int nC = 5;
    std::vector<scalar> hy(nC), hV(nC), hnt(nC), hgrad(9*nC, 0.0);
    double g9[5][9] = {{0}};
    for (int c = 0; c < nC; ++c) {
        hy[c] = cells[c].y; hV[c] = cells[c].V; hnt[c] = cells[c].nuTilda;
        hgrad[0*nC + c] = cells[c].gmag;   // put the whole magnitude in one component (component-major layout)
        g9[c][0] = cells[c].gmag;
    }
    DeviceBuffer<scalar> y, V, nt, gradU, dTilda;
    y.copyFrom(hy); V.copyFrom(hV); nt.copyFrom(hnt); gradU.copyFrom(hgrad);
    deviceSADDESdTilda(nC, y, V, gradU, nt, nu, co, dTilda);
    const auto dt = dTilda.host();

    std::printf("device kernel vs host formula (exact):\n");
    for (int c = 0; c < nC; ++c) {
        char nm[32]; std::snprintf(nm, sizeof nm, "cell %d dTilda", c);
        chk(nm, dt[c], hostDTilda(cells[c].y, cells[c].V, g9[c], cells[c].nuTilda, nu, co), 1e-12);
    }
    std::printf("physical limits (by hand):\n");
    chk("RANS fd~0 -> dTilda=y",                 dt[0], 0.001, 1e-9);
    chk("LES high-chi -> Psi*CDES*Delta",        dt[1], hostPsi(0.1/nu, co)*co.CDES, 1e-9);
    chk("LES fd~1, y<lLES -> y",                 dt[2], 0.3,   1e-6);
    chk("LES low-chi -> Psi*CDES*Delta (>0.65)", dt[4], hostPsi(1e-4/nu, co)*co.CDES, 1e-9);
    chk("Psi(high chi) ~ 1",                     hostPsi(0.1/nu, co), 1.0, 1e-3);
    chk("Psi(low chi) > 1 (near-wall enlarge)",  (hostPsi(1e-4/nu, co) > 1.0) ? 1.0 : 0.0, 1.0, 0.0);
    chk("all dTilda <= y (limiter never grows)", (dt[0]<=hy[0]&&dt[1]<=hy[1]&&dt[2]<=hy[2]&&dt[3]<=hy[3]&&dt[4]<=hy[4])?1.0:0.0, 1.0, 0.0);

    std::printf("\n%s: %d failure(s)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
