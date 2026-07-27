// Smagorinsky LES sub-grid viscosity: deviceSmagorinskyNut reproduces OF's Smagorinsky<>::k() -> nut exactly.
//   D = symm(gradU); a = Ce/delta; b = (2/3)tr(D); c = 2*Ck*delta*(dev(D)&&D); sqrt(k) = (-b+sqrt(b^2+4ac))/(2a);
//   nut = Ck*delta*sqrt(k),  delta = cbrt(V).
// Checks (a) the device kernel matches a host re-implementation of the quadratic, and (b) the physical limits by hand:
//   zero strain -> nut=0; pure isotropic expansion (dev(D)&&D=0) -> nut=0 (deviatoric subtraction); delta^2 scaling;
//   and the incompressible pure-shear value equals the classic (Cs*delta)^2*|S| with Cs = sqrt(Ck*sqrt(Ck/Ce)) ~ 0.168.
#include "device_smagorinsky.cuh"   // deviceSmagorinskyNut
#include "smagorinsky_coeffs.cuh"   // SmagorinskyCoeffs
#include "device_buffer.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

static double hostSmagNut(const double t[9], double V, const SmagorinskyCoeffs& co)
{
    const double delta = std::cbrt(V);
    const double trD = t[0] + t[4] + t[8];
    double DD = 0.0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            const double sab = 0.5 * (t[i*3+j] + t[j*3+i]);
            DD += sab * sab;
        }
    const double devDD = std::fmax(DD - trD*trD/3.0, 0.0);
    const double a = co.Ce / std::fmax(delta, 1e-300);
    const double b = (2.0/3.0) * trD;
    const double cc = 2.0 * co.Ck * delta * devDD;
    const double sqrtk = (-b + std::sqrt(std::fmax(b*b + 4.0*a*cc, 0.0))) / (2.0*a);
    return co.Ck * delta * std::fmax(sqrtk, 0.0);
}

int main()
{
    int fails = 0;
    auto chk = [&](const char* nm, double got, double exp, double tol) {
        const bool ok = std::fabs(got - exp) <= tol*std::fmax(1.0, std::fabs(exp));
        if (!ok) ++fails;
        std::printf("  %-44s got %.10g  exp %.10g  %s\n", nm, got, exp, ok ? "OK" : "FAIL");
    };

    const SmagorinskyCoeffs co;   // OF defaults Ck=0.094, Ce=1.048
    const int nC = 5;
    // [0]=zero strain; [1]=pure shear (delta=1); [2]=isotropic expansion (dev=0); [3]=general (nonzero divU);
    // [4]=pure shear at delta=2 (V=8) -> nut = 4x cell[1] (delta^2 scaling).
    double g[nC][9] = {
        {0,0,0,  0,0,0,  0,0,0},
        {0,2.0,0, 0,0,0, 0,0,0},
        {0.5,0,0, 0,0.5,0, 0,0,0.5},
        {0.1,0.5,-0.2, 0.3,-0.1,0.4, 0.05,-0.3,0.15},
        {0,2.0,0, 0,0,0, 0,0,0},
    };
    double Vc[nC] = {1.0, 1.0, 1.0, 2.0, 8.0};

    std::vector<scalar> hgradU(9*nC), hV(nC);
    for (int c = 0; c < nC; ++c) { hV[c] = Vc[c]; for (int q = 0; q < 9; ++q) hgradU[q*nC + c] = g[c][q]; }
    DeviceBuffer<scalar> gradU, V, nut;
    gradU.copyFrom(hgradU); V.copyFrom(hV);
    deviceSmagorinskyNut(nC, gradU, V, co, nut);
    const auto nd = nut.host();

    std::printf("device kernel vs host OF quadratic (exact):\n");
    for (int c = 0; c < nC; ++c) {
        char nm[32]; std::snprintf(nm, sizeof nm, "cell %d nut", c);
        chk(nm, nd[c], hostSmagNut(g[c], Vc[c], co), 1e-12);
    }

    std::printf("physical limits (by hand):\n");
    chk("zero strain -> nut=0",              nd[0], 0.0, 1e-14);
    chk("isotropic expansion -> nut=0 (dev)", nd[2], 0.0, 1e-14);
    chk("delta^2 scaling: cell4 = 4*cell1",  nd[4], 4.0*nd[1], 1e-9);
    // pure-shear (incompressible) equals classic (Cs*delta)^2*|S|, |S|=sqrt(2*S:S). cell[1]: S_xy=1 -> S:S=2 -> |S|=2, delta=1.
    const double Sabs = 2.0, Cs = co.Cs();
    chk("pure shear = (Cs*delta)^2*|S|",     nd[1], Cs*Cs*1.0*Sabs, 1e-12);
    chk("recovered Cs ~ 0.168",              std::sqrt(nd[1]/(1.0*Sabs)), 0.1678, 5e-3);

    std::printf("\n%s: %d failure(s)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
