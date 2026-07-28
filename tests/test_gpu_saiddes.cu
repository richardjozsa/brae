// SA-IDDES (SpalartAllmarasIDDES, Shur/Spalart/Strelets/Travin 2008) length scale: deviceSAIDDESdTilda reproduces the
// improved delayed-DES dTilda exactly.
//   dTilda = fdTilde*(1 + fe)*y + (1 - fdTilde)*CDES*Delta,  Delta = min(max(Cw*y, Cw*hmax), hmax),
//   fdTilde = max(1 - fdt, fB),  fdt = 1 - tanh((Cdt1*rd_t)^3),  fB = min(2 exp(-9 alpha^2), 1),  alpha = 0.25 - y/hmax,
//   fe = max(fe1 - 1, 0)*fe2,  fe2 = 1 - max(ft, fl),  ft = tanh((Ct^2 rd_t)^3),  fl = tanh((Cl^2 rd_l)^10),
//   fe1 = 2 exp(-11.09 alpha^2) [alpha>=0] | 2 exp(-9 alpha^2) [alpha<0];  rd_t/rd_l from nut/nu.  (hwn omitted -> hwn=0.)
// Checks (a) the device kernel matches a host re-implementation, and (b) the physical limits by hand: deep near-wall
// (shielded) -> dTilda = y (RANS); far-field -> dTilda = CDES*hmax (LES); the fe band -> dTilda > y (elevated stresses).
#include "device_kepsilon.cuh"    // deviceSAIDDESdTilda
#include "spalart_coeffs.cuh"     // SpalartAllmarasCoeffs
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

static double hostIddes(const double g[9], double y, double hmax, double hwn, double nt, double nu, const SpalartAllmarasCoeffs& co)
{
    double g2 = 0; for (int k = 0; k < 9; ++k) g2 += g[k]*g[k];
    const double magGradU = std::fmax(std::sqrt(g2), 1e-300);
    const double chi = nt/nu, chi3 = chi*chi*chi, Cv13 = co.Cv1*co.Cv1*co.Cv1;
    const double nutc = nt*(chi3/(chi3+Cv13));
    const double kd2 = std::fmax(co.kappa*co.kappa*y*y, 1e-300);
    const double rdt = std::fmin(nutc/(magGradU*kd2), 10.0);
    const double rdl = std::fmin(nu  /(magGradU*kd2), 10.0);
    const double adt = co.Cdt1*rdt; const double fdt = 1.0 - std::tanh(adt*adt*adt);
    const double al = co.Cl*co.Cl*rdl; const double al2=al*al, al4=al2*al2, al8=al4*al4;
    const double fl = std::tanh(al8*al2);
    const double at = co.Ct*co.Ct*rdt; const double ft = std::tanh(at*at*at);
    const double fe2 = 1.0 - std::fmax(ft, fl);
    const double hm = std::fmax(hmax, 1e-300);
    const double alpha = 0.25 - y/hm;
    const double fB = std::fmin(2.0*std::exp(-9.0*alpha*alpha), 1.0);
    const double fdTilde = std::fmax(1.0 - fdt, fB);
    const double fe1 = (alpha >= 0.0) ? 2.0*std::exp(-11.09*alpha*alpha) : 2.0*std::exp(-9.0*alpha*alpha);
    const double fe = std::fmax(fe1 - 1.0, 0.0)*fe2;
    const double delta = std::fmin(std::fmax(std::fmax(co.Cw*y, co.Cw*hm), hwn), hm);
    const double lLES = hostPsi(nt/nu, co)*co.CDES*delta;
    return std::fmax(fdTilde*(1.0 + fe)*y + (1.0 - fdTilde)*lLES, 1e-300);
}

int main()
{
    int fails = 0;
    auto chk = [&](const char* nm, double got, double exp, double tol) {
        const bool ok = std::fabs(got - exp) <= tol*std::fmax(1.0, std::fabs(exp));
        if (!ok) ++fails;
        std::printf("  %-42s got %.10g  exp %.10g  %s\n", nm, got, exp, ok ? "OK" : "FAIL");
    };

    SpalartAllmarasCoeffs co;   // CDES=0.65, Cdt1=20, Cl=5, Ct=1.87, Cw=0.15, kappa=0.41, Cv1=7.1
    const double nu = 1e-5;
    const int nC = 5;
    // [0]=deep near-wall (shielded RANS); [1]=far-field (LES); [2]=fe elevated-stress band (huge shear keeps rd_t moderate);
    // [3]=generic cell (exactness); [4]=hwn is the binding delta term (Cw*y,Cw*hmax < hwn < hmax) in a near-LES cell.
    double g[nC][9] = {
        {0,1000,0,  0,0,0,  0,0,0},
        {0,100,0,   0,0,0,  0,0,0},
        {0,495000,0,0,0,0,  0,0,0},
        {0.1,0.5,-0.2, 0.3,-0.1,0.4, 0.05,-0.3,0.15},
        {0,1e5,0,   0,0,0,  0,0,0},
    };
    double yc[nC]    = {1e-4, 1.0,  2e-4, 5e-3, 1e-3};
    double hmaxc[nC] = {1e-3, 1e-3, 1e-3, 2e-3, 1e-3};
    double hwnc[nC]  = {1e-3, 5e-4, 1e-3, 1e-3, 6e-4};   // cell4: 0.15*y=1.5e-4, 0.15*hmax=1.5e-4 < hwn=6e-4 < hmax=1e-3 -> delta=hwn
    double ntc[nC]   = {1e-2, 1e-4, 1e-3, 1e-3, 1e-4};

    std::vector<scalar> hgradU(9*nC), hy(nC), hhmax(nC), hhwn(nC), hnt(nC);
    for (int c = 0; c < nC; ++c) {
        hy[c]=yc[c]; hhmax[c]=hmaxc[c]; hhwn[c]=hwnc[c]; hnt[c]=ntc[c];
        for (int k = 0; k < 9; ++k) hgradU[k*nC + c] = g[c][k];
    }
    DeviceBuffer<scalar> y, hmax, hwn, gradU, nt, dT;
    y.copyFrom(hy); hmax.copyFrom(hhmax); hwn.copyFrom(hhwn); gradU.copyFrom(hgradU); nt.copyFrom(hnt);
    deviceSAIDDESdTilda(nC, y, hmax, hwn, gradU, nt, nu, co, dT);
    const auto d = dT.host();

    std::printf("device kernel vs host IDDES formula (exact):\n");
    for (int c = 0; c < nC; ++c) {
        char nm[32]; std::snprintf(nm, sizeof nm, "cell %d dTilda", c);
        chk(nm, d[c], hostIddes(g[c], yc[c], hmaxc[c], hwnc[c], ntc[c], nu, co), 1e-12);
    }

    std::printf("physical limits (by hand):\n");
    chk("deep near-wall (shielded) -> dTilda=y", d[0], yc[0], 1e-9);
    chk("far-field -> dTilda=Psi*CDES*hmax (LES)", d[1], hostPsi(ntc[1]/nu, co)*co.CDES*hmaxc[1], 1e-6);
    chk("elevated-stress band -> dTilda>y",      (d[2] > yc[2]) ? 1.0 : 0.0, 1.0, 0.0);
    chk("hwn binds -> dTilda ~ Psi*CDES*hwn (LES)", d[4], hostPsi(ntc[4]/nu, co)*co.CDES*hwnc[4], 3e-2);
    // hwn actually enters: cell4 differs from the hwn-omitted (hwn=0) length scale.
    chk("hwn changes dTilda (vs hwn=0)", (std::fabs(d[4] - hostIddes(g[4], yc[4], hmaxc[4], 0.0, ntc[4], nu, co)) > 1e-6) ? 1.0 : 0.0, 1.0, 0.0);
    chk("all dTilda > 0", (d[0]>0&&d[1]>0&&d[2]>0&&d[3]>0&&d[4]>0)?1.0:0.0, 1.0, 0.0);

    std::printf("\n%s: %d failure(s)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
