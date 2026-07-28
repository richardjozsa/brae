// kOmegaSST-DDES DES-factor gate: deviceKOmegaSSTDESfactor reproduces the OF kOmegaSSTDDES enhancement
//   FDES = max( (Lt/(CDES*Delta))*(1 - F2), 1 ),  Lt = sqrt(k)/(betaStar*omega),  Delta = cubeRootVol = V^(1/3),
//   CDES = F1*CDES1 + (1-F1)*CDES2.
// Checks (a) the device kernel matches a host re-implementation, and (b) the physical limits by hand: the DDES shielding
// F2->1 (boundary layer) => FDES=1 (RANS, no extra destruction); F2->0 + Lt>CDES*Delta (free shear) => FDES>1 (LES).
#include "device_komega_sst.cuh"   // deviceKOmegaSSTDESfactor
#include "komega_sst_coeffs.cuh"   // KOmegaSSTCoeffs
#include "device_buffer.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

static double hostFDES(double k, double om, double V, double F1, double F2, const KOmegaSSTCoeffs& co)
{
    const double delta = std::cbrt(V);
    const double Lt = std::sqrt(std::fmax(k, 0.0)) / std::fmax(co.betaStar*om, 1e-300);
    const double CDES = F1*co.CDES1 + (1.0 - F1)*co.CDES2;
    return std::fmax((Lt/std::fmax(CDES*delta, 1e-300))*(1.0 - F2), 1.0);
}

int main()
{
    int fails = 0;
    auto chk = [&](const char* nm, double got, double exp, double tol) {
        const bool ok = std::fabs(got - exp) <= tol*std::fmax(1.0, std::fabs(exp));
        if (!ok) ++fails;
        std::printf("  %-40s got %.10g  exp %.10g  %s\n", nm, got, exp, ok ? "OK" : "FAIL");
    };

    const KOmegaSSTCoeffs co;   // betaStar=0.09, CDES1=0.78, CDES2=0.61
    // 4 cells: [0]=BL shield(F2=1), [1]=free-shear LES(F2=0, large Lt), [2]=free-shear but small Lt(->1), [3]=intermediate.
    struct Cell { double k, om, V, F1, F2; } cells[] = {
        {0.1, 100.0, 1.0, 1.0, 1.0},   // F2=1 -> (1-F2)=0 -> FDES=1
        {1.0, 1.0,   1.0, 1.0, 0.0},   // Lt=1/0.09=11.11, CDES=0.78, Delta=1 -> FDES=11.11/0.78=14.25
        {1e-6,100.0, 1.0, 1.0, 0.0},   // Lt tiny -> FDES=1
        {1.0, 1.0,   1.0, 0.5, 0.3},   // F1 blend + partial shield -> host value
    };
    const int nC = 4;
    std::vector<scalar> hk(nC), hom(nC), hV(nC), hF1(nC), hF2(nC);
    for (int c = 0; c < nC; ++c) { hk[c]=cells[c].k; hom[c]=cells[c].om; hV[c]=cells[c].V; hF1[c]=cells[c].F1; hF2[c]=cells[c].F2; }
    DeviceBuffer<scalar> k, om, V, F1, F2, FDES;
    k.copyFrom(hk); om.copyFrom(hom); V.copyFrom(hV); F1.copyFrom(hF1); F2.copyFrom(hF2);
    deviceKOmegaSSTDESfactor(nC, k, om, V, F1, F2, co, FDES);
    const auto fd = FDES.host();

    std::printf("device kernel vs host formula (exact):\n");
    for (int c = 0; c < nC; ++c) {
        char nm[32]; std::snprintf(nm, sizeof nm, "cell %d FDES", c);
        chk(nm, fd[c], hostFDES(cells[c].k, cells[c].om, cells[c].V, cells[c].F1, cells[c].F2, co), 1e-12);
    }
    std::printf("physical limits (by hand):\n");
    chk("BL shield F2=1 -> FDES=1",           fd[0], 1.0,   1e-12);
    chk("free-shear LES -> 11.11/0.78=14.25", fd[1], 11.11111111/0.78, 1e-6);
    chk("small Lt -> FDES=1 (RANS)",           fd[2], 1.0,   1e-12);
    chk("FDES >= 1 everywhere",                (fd[0]>=1.0&&fd[1]>=1.0&&fd[2]>=1.0&&fd[3]>=1.0)?1.0:0.0, 1.0, 0.0);

    std::printf("\n%s: %d failure(s)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
