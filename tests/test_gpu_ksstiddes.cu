// kOmegaSST-IDDES (Gritskevich/Garbaruk/Schuetze/Menter 2012) length scale: deviceKOmegaSSTIDDESfactor reproduces the
// k-destruction scaling FDES = lRAS/lIDDES exactly.
//   lRAS = sqrt(k)/(betaStar*omega),  lLES = CDES*Delta,  CDES = F1*CDES1 + (1-F1)*CDES2,  Delta = min(max(Cw*y, Cw*hmax), hmax),
//   lIDDES = fdTilde*(1+fe)*lRAS + (1-fdTilde)*lLES,  fdTilde = max(1-fdt, fB),  fdt = 1 - tanh((Cdt1 rd_t)^3),
//   fB = min(2 exp(-9 alpha^2),1), alpha = 0.25 - y/hmax,  fe = max(fe1-1,0)*fe2, fe2 = 1 - max(ft,fl),
//   ft = tanh((Ct^2 rd_t)^3), fl = tanh((Cl^2 rd_l)^10),  rd_t/rd_l from nut/nu over |gradU| = sqrt(0.5(S^2+Omega^2)).
// Checks (a) the device kernel matches a host re-implementation, and (b) the physical limits by hand: deep near-wall
// (shielded) -> FDES=1 (RANS); free shear with lRAS>>lLES -> FDES>1 (LES, extra k destruction); the fe band ->
// lIDDES>lRAS -> FDES<1 (elevated stresses, reduced destruction).
#include "device_komega_sst.cuh"   // deviceKOmegaSSTIDDESfactor
#include "komega_sst_coeffs.cuh"   // KOmegaSSTCoeffs
#include "device_buffer.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

static double hostFDES(const double g[9], double k, double om, double F1, double nut, double y, double hmax, double hwn, double nu,
                       const KOmegaSSTCoeffs& co)
{
    double g2 = 0; for (int q = 0; q < 9; ++q) g2 += g[q]*g[q];
    const double magGradU = std::fmax(std::sqrt(g2), 1e-300);
    const double lRAS = std::sqrt(std::fmax(k, 0.0))/std::fmax(co.betaStar*om, 1e-300);
    const double CDES = F1*co.CDES1 + (1.0 - F1)*co.CDES2;
    const double hm = std::fmax(hmax, 1e-300);
    const double delta = std::fmin(std::fmax(std::fmax(co.Cw*y, co.Cw*hm), hwn), hm);
    const double lLES = CDES*delta;
    const double kd2 = std::fmax(co.kappa*co.kappa*y*y, 1e-300);
    const double rdt = std::fmin(nut/(magGradU*kd2), 10.0);
    const double rdl = std::fmin(nu /(magGradU*kd2), 10.0);
    const double adt = co.Cdt1*rdt; const double fdt = 1.0 - std::tanh(adt*adt*adt);
    const double al = co.Cl*co.Cl*rdl; const double al2=al*al, al4=al2*al2, al8=al4*al4;
    const double fl = std::tanh(al8*al2);
    const double at = co.Ct*co.Ct*rdt; const double ft = std::tanh(at*at*at);
    const double fe2 = 1.0 - std::fmax(ft, fl);
    const double alpha = 0.25 - y/hm;
    const double fB = std::fmin(2.0*std::exp(-9.0*alpha*alpha), 1.0);
    const double fdTilde = std::fmax(1.0 - fdt, fB);
    const double fe1 = (alpha >= 0.0) ? 2.0*std::exp(-11.09*alpha*alpha) : 2.0*std::exp(-9.0*alpha*alpha);
    const double fe = std::fmax(fe1 - 1.0, 0.0)*fe2;
    const double lIDDES = std::fmax(fdTilde*(1.0 + fe)*lRAS + (1.0 - fdTilde)*lLES, 1e-300);
    return lRAS/lIDDES;
}

int main()
{
    int fails = 0;
    auto chk = [&](const char* nm, double got, double exp, double tol) {
        const bool ok = std::fabs(got - exp) <= tol*std::fmax(1.0, std::fabs(exp));
        if (!ok) ++fails;
        std::printf("  %-42s got %.10g  exp %.10g  %s\n", nm, got, exp, ok ? "OK" : "FAIL");
    };

    KOmegaSSTCoeffs co;   // betaStar=0.09, CDES1=0.78, CDES2=0.61, kappa=0.41, Cdt1=20, Cl=5, Ct=1.87, Cw=0.15
    const double nu = 1e-5;
    const int nC = 5;
    // [0]=deep near-wall (shielded RANS); [1]=free shear (LES, lRAS>>lLES); [2]=fe band (elevated stresses); [3]=generic;
    // [4]=hwn is the binding delta term (Cw*y,Cw*hmax < hwn < hmax) in a near-LES cell with small lRAS.
    double g[nC][9] = {
        {0,1000,0,  0,0,0,  0,0,0},
        {0,100,0,   0,0,0,  0,0,0},
        {0,495800,0,0,0,0,  0,0,0},
        {0.1,0.5,-0.2, 0.3,-0.1,0.4, 0.05,-0.3,0.15},
        {0,1e5,0,   0,0,0,  0,0,0},
    };
    double kc[nC]   = {1.0,  1.0,  0.5,  0.8,  1e-6};
    double omc[nC]  = {1000, 1.0,  100,  50,   100};
    double F1c[nC]  = {1.0,  0.0,  1.0,  0.5,  0.0};
    double nutc[nC] = {1e-3, 1e-5, 1e-3, 1e-4, 1e-5};
    double yc[nC]   = {1e-4, 1.0,  2e-4, 5e-3, 1e-3};
    double hmaxc[nC]= {1e-3, 1e-3, 1e-3, 2e-3, 1e-3};
    double hwnc[nC] = {1e-3, 5e-4, 1e-3, 1e-3, 6e-4};   // cell4: 0.15*y=1.5e-4, 0.15*hmax=1.5e-4 < hwn=6e-4 < hmax=1e-3 -> delta=hwn

    std::vector<scalar> hg(9*nC), hk(nC), ho(nC), hF1(nC), hnt(nC), hy(nC), hh(nC), hw(nC);
    for (int c = 0; c < nC; ++c) {
        hk[c]=kc[c]; ho[c]=omc[c]; hF1[c]=F1c[c]; hnt[c]=nutc[c]; hy[c]=yc[c]; hh[c]=hmaxc[c]; hw[c]=hwnc[c];
        for (int q = 0; q < 9; ++q) hg[q*nC + c] = g[c][q];
    }
    DeviceBuffer<scalar> k, om, F1, gradU, nut, y, hmax, hwn, FDES;
    k.copyFrom(hk); om.copyFrom(ho); F1.copyFrom(hF1); gradU.copyFrom(hg); nut.copyFrom(hnt); y.copyFrom(hy); hmax.copyFrom(hh); hwn.copyFrom(hw);
    deviceKOmegaSSTIDDESfactor(nC, k, om, F1, gradU, nut, y, hmax, hwn, nu, co, FDES);
    const auto d = FDES.host();

    std::printf("device kernel vs host IDDES factor (exact):\n");
    for (int c = 0; c < nC; ++c) {
        char nm[32]; std::snprintf(nm, sizeof nm, "cell %d FDES", c);
        chk(nm, d[c], hostFDES(g[c], kc[c], omc[c], F1c[c], nutc[c], yc[c], hmaxc[c], hwnc[c], nu, co), 1e-12);
    }

    std::printf("physical limits (by hand):\n");
    chk("deep near-wall (shielded) -> FDES=1", d[0], 1.0, 1e-12);
    chk("free shear -> FDES>1 (LES)",          (d[1] > 1.0) ? 1.0 : 0.0, 1.0, 0.0);
    chk("elevated-stress band -> FDES<1",      (d[2] < 1.0) ? 1.0 : 0.0, 1.0, 0.0);
    // hwn actually enters: cell4 differs from the hwn-omitted (hwn=0) length scale.
    chk("hwn changes FDES (vs hwn=0)", (std::fabs(d[4] - hostFDES(g[4], kc[4], omc[4], F1c[4], nutc[4], yc[4], hmaxc[4], 0.0, nu, co)) > 1e-3) ? 1.0 : 0.0, 1.0, 0.0);
    chk("all FDES > 0", (d[0]>0&&d[1]>0&&d[2]>0&&d[3]>0&&d[4]>0)?1.0:0.0, 1.0, 0.0);

    std::printf("\n%s: %d failure(s)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
