// C phase ③ gate, kOmegaSST F1/F2/CDkOmega/S2 device kernels vs OF v2412, on a frozen field state with the
// SAME gradient inputs (isolates the pointwise blending/limiter formulas from grad-reduction order → machine
// precision). Reference: validation/of_apps/dumpKOmegaSST writes <case>/komega_sst.dat (per cell: k, omega, y,
// gradU[9], gradk[3], gradOmega[3], S2, CDkOmega, F1, F2). omega derived from a converged k-eps state.
// Run: test_gpu_komega_sst <komega_sst.dat>
#include "device_buffer.cuh"
#include "device_komega_sst.cuh"
#include "komega_sst_coeffs.cuh"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

static void stats(const char* nm, const std::vector<scalar>& cf, const std::vector<scalar>& of,
                  scalar tol, int& fails) {
    scalar maxAbs = 0, maxDen = 0;
    for (std::size_t i = 0; i < cf.size(); ++i) { maxAbs = std::fmax(maxAbs, std::fabs(cf[i]-of[i])); maxDen = std::fmax(maxDen, std::fabs(of[i])); }
    const scalar rel = maxDen > 0 ? maxAbs / maxDen : maxAbs;
    const bool ok = rel <= tol;
    if (!ok) ++fails;
    std::printf("  %-10s max|cf-OF| %.3e  rel %.3e  (tol %.1e)  %s\n", nm, maxAbs, rel, tol, ok ? "OK" : "FAIL");
}

int main(int argc, char** argv) {
    const std::string dat = argc > 1 ? argv[1] : "validation/pitzDaily282/komega_sst.dat";
    std::ifstream in(dat);
    if (!in) { std::printf("cannot open %s\n", dat.c_str()); return 2; }
    int nC = 0; in >> nC;
    if (nC <= 0) { std::printf("bad nCells in %s\n", dat.c_str()); return 2; }

    std::vector<scalar> k(nC), om(nC), y(nC), gU(9*nC);
    std::vector<scalar> gKx(nC), gKy(nC), gKz(nC), gOx(nC), gOy(nC), gOz(nC);
    std::vector<scalar> S2of(nC), CDof(nC), F1of(nC), F2of(nC);
    std::vector<scalar> nutInOf(nC), GbyNu0Of(nC), nutNewOf(nC), PkOf(nC), GbyNuLimOf(nC);
    std::vector<scalar> Vol(nC), divU(nC), reactDiagOf(nC), reactSrcOf(nC), kReactDiagOf(nC), kReactSrcOf(nC);
    for (int c = 0; c < nC; ++c) {
        in >> k[c] >> om[c] >> y[c];
        for (int q = 0; q < 9; ++q) in >> gU[q*nC + c];        // pack tensor into [q*nC+c] (deviceGbyNu layout)
        in >> gKx[c] >> gKy[c] >> gKz[c] >> gOx[c] >> gOy[c] >> gOz[c];
        in >> S2of[c] >> CDof[c] >> F1of[c] >> F2of[c];
        in >> nutInOf[c] >> GbyNu0Of[c] >> nutNewOf[c] >> PkOf[c] >> GbyNuLimOf[c];
        in >> Vol[c] >> divU[c] >> reactDiagOf[c] >> reactSrcOf[c] >> kReactDiagOf[c] >> kReactSrcOf[c];
    }
    if (!in) { std::printf("short read in %s\n", dat.c_str()); return 2; }

    const scalar nu = 1e-5;                                    // pitzDaily; same value the OF dump used
    KOmegaSSTCoeffs co;

    DeviceBuffer<scalar> dGU(gU), dK(k), dOm(om), dY(y);
    DeviceBuffer<scalar> dGKx(gKx), dGKy(gKy), dGKz(gKz), dGOx(gOx), dGOy(gOy), dGOz(gOz);
    DeviceBuffer<scalar> dS2, dCD, dF1, dF2;

    deviceS2(dGU, nC, dS2);
    deviceCDkOmega(dGKx, dGKy, dGKz, dGOx, dGOy, dGOz, dOm, co.alphaOmega2, dCD);
    deviceF1(dK, dOm, dY, dCD, nu, co, dF1);
    deviceF2(dK, dOm, dY, nu, co, dF2);

    const std::vector<scalar> S2cf = dS2.host(), CDcf = dCD.host(), F1cf = dF1.host(), F2cf = dF2.host();

    std::printf("test_gpu_komega_sst (%s, nCells=%d):\n", dat.c_str(), nC);
    int fails = 0;
    stats("S2",       S2cf, S2of, 1e-12, fails);               // pure mul/add -> bit-close
    stats("CDkOmega", CDcf, CDof, 1e-12, fails);               // pure mul/div -> bit-close
    stats("F1",       F1cf, F1of, 1e-10, fails);               // sqrt + tanh (CUDA vs libm ~1 ULP)
    stats("F2",       F2cf, F2of, 1e-10, fails);

    // blend self-check: deviceBlend(F1of, gamma1, gamma2) == F1of*(g1-g2)+g2 (host).
    DeviceBuffer<scalar> dF1of(F1of), dGamma; deviceBlend(dF1of, co.gamma1, co.gamma2, dGamma);
    std::vector<scalar> gExp(nC); for (int c = 0; c < nC; ++c) gExp[c] = F1of[c]*(co.gamma1-co.gamma2)+co.gamma2;
    stats("blend",    dGamma.host(), gExp, 1e-14, fails);

    // C④ limiters, fed OF's F2/S2/GbyNu0 inputs (isolate the limiter formula). G = nut*GbyNu0 for Pk.
    DeviceBuffer<scalar> dF2of(F2of), dS2of(S2of), dGbyNu0(GbyNu0Of);
    std::vector<scalar> Gin(nC); for (int c = 0; c < nC; ++c) Gin[c] = nutInOf[c]*GbyNu0Of[c];
    DeviceBuffer<scalar> dG(Gin), dNut, dPk, dGbyNu;
    deviceNutSST(dK, dOm, dF2of, dS2of, co, dNut);
    devicePk(dG, dK, dOm, co, dPk);
    deviceGbyNuLimit(dGbyNu0, dOm, dF2of, dS2of, co, dGbyNu);
    stats("nut",      dNut.host(),   nutNewOf,   1e-12, fails);    // a1*k/max(...), sqrt+div, bit-close
    stats("Pk",       dPk.host(),    PkOf,       1e-12, fails);    // min(G, c1 b* k w)
    stats("GbyNu",    dGbyNu.host(), GbyNuLimOf, 1e-12, fails);    // min(GbyNu0, ...)

    // C⑤ omega reaction: cf builds gamma/beta (blend) + GbyNu0lim, then the reaction diag/source. Compare vs OF.
    DeviceBuffer<scalar> dGamma2, dBeta; deviceBlend(dF1of, co.gamma1, co.gamma2, dGamma2); deviceBlend(dF1of, co.beta1, co.beta2, dBeta);
    DeviceBuffer<scalar> dVol(Vol), dDivU(divU);
    DeviceBuffer<scalar> dDiag(std::vector<scalar>(nC, 0.0)), dSrc(std::vector<scalar>(nC, 0.0));
    DeviceBuffer<scalar> dF1c(F1of), dCDc(CDof);
    deviceOmegaReaction(dVol, dGamma2, dBeta, dGbyNu, dF1c, dCDc, dOm, dDivU, dDiag, dSrc);   // dGbyNu = GbyNu0lim from above
    stats("omReactD", dDiag.host(), reactDiagOf, 1e-12, fails);   // diag contribution
    stats("omReactS", dSrc.host(),  reactSrcOf,  1e-12, fails);   // source contribution

    // C⑥ k reaction: G = nut*GbyNu0 (raw); Pk inside the kernel. Compare diag/source vs OF.
    DeviceBuffer<scalar> dGraw(Gin), dKDiag(std::vector<scalar>(nC, 0.0)), dKSrc(std::vector<scalar>(nC, 0.0));
    deviceKReactionSST(dVol, dK, dOm, dGraw, dDivU, co, dKDiag, dKSrc);
    stats("kReactD",  dKDiag.host(), kReactDiagOf, 1e-12, fails);
    stats("kReactS",  dKSrc.host(),  kReactSrcOf,  1e-12, fails);

    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
