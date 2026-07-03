// C phase ② gate, kOmegaSST coefficients: a default-constructed KOmegaSSTCoeffs reproduces the OF v2412
// kOmegaSSTBase.C constructor defaults exactly, and readKOmegaSSTCoeffs parses RAS.kOmegaSSTCoeffs (custom keys
// override, absent keys keep defaults, F3 Switch). Spec = the source line refs in komega_sst_coeffs.cuh.
// Run: test_komega_coeffs [caseDir]   (caseDir/constant/turbulenceProperties = the custom fixture)
#include "komega_sst_coeffs.cuh"
#include "foam_dict.cuh"

#include <cmath>
#include <cstdio>
#include <string>

using namespace brae;

int main(int argc, char** argv) {
    int fails = 0;
    auto chk = [&](const char* nm, double got, double exp) {
        const bool ok = std::fabs(got - exp) <= 1e-12 * std::fmax(1.0, std::fabs(exp));
        if (!ok) ++fails;
        std::printf("  %-22s got %.8g  exp %.8g  %s\n", nm, got, exp, ok ? "OK" : "FAIL");
    };

    // 1. defaults == OF v2412 kOmegaSSTBase.C ctor
    std::printf("defaults vs OF v2412 kOmegaSSTBase.C:\n");
    KOmegaSSTCoeffs d;
    chk("alphaK1", d.alphaK1, 0.85);          chk("alphaK2", d.alphaK2, 1.0);
    chk("alphaOmega1", d.alphaOmega1, 0.5);   chk("alphaOmega2", d.alphaOmega2, 0.856);
    chk("gamma1", d.gamma1, 5.0/9.0);         chk("gamma2", d.gamma2, 0.44);
    chk("beta1", d.beta1, 0.075);             chk("beta2", d.beta2, 0.0828);
    chk("betaStar", d.betaStar, 0.09);
    chk("a1", d.a1, 0.31);                    chk("b1", d.b1, 1.0);   chk("c1", d.c1, 10.0);
    chk("F3", d.F3 ? 1.0 : 0.0, 0.0);
    chk("kappa", d.kappa, 0.41);              chk("E", d.E, 9.8);

    // 2. dict reading: custom keys override, absent keys keep defaults, F3 Switch parses.
    if (argc >= 2) {
        std::printf("from-dict parse (RAS.kOmegaSSTCoeffs):\n");
        const FoamDict tp = readDict(std::string(argv[1]) + "/constant/turbulenceProperties");
        const FoamDict* ras = tp.subDict("RAS");
        KOmegaSSTCoeffs c;
        readKOmegaSSTCoeffs(ras, c);
        chk("a1 (overridden)",   c.a1, 0.35);
        chk("betaStar (overridden)", c.betaStar, 0.085);
        chk("F3 (overridden)",   c.F3 ? 1.0 : 0.0, 1.0);
        chk("alphaK2 (default kept)", c.alphaK2, 1.0);
        chk("gamma2 (default kept)",  c.gamma2, 0.44);
    }

    std::printf("%s\n", fails == 0 ? "PASS" : "FAIL");
    return fails == 0 ? 0 : 1;
}
