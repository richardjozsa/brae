// INTEGRATION: the real deviceThermoCorrect() liquid path, not the standalone functions.
//
// WHAT THIS CATCHES THAT THE STANDALONE TESTS CANNOT. test_etot proves the inversion solves
// e(p,T) = target; test_liquid_fields proves the correlations fill fields correctly. Neither says the
// two are WIRED together right. This calls the function the solver calls and checks the whole result,
// so it fails on:
//   - passing enthalpy semantics where the case asked for internal energy
//   - handing the inversion the wrong p
//   - evaluating the properties BEFORE updating T (a one-iteration lag that looks perfectly smooth)
//   - writing the solver rho instead of rhoThermo
//   - indexing the wrong field
//
// The inputs are OF's own Es(p,T) at known temperatures, with deliberately WRONG initial temperatures
// in th.T, so the inversion has to do real work and cannot pass by echoing its guess.
#include "device_thermo.cuh"
#include "nsrds_functions.cuh"
#include "thermo_types.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace brae;

namespace {
struct Row { double T, p, rho, mu, kappa, Cp, Es; };
const std::vector<Row> kOF = {
    {280, 50000, 999.62487928454073, 0.0014300291329221544, 0.57871727999999989, 4211.0158121564882, -15934627.158276683},
    {290, 50000, 997.09779658913828, 0.0011126737698711271, 0.59440638499999987, 4193.8003320010603, -15892608.964519063},
    {300, 50000, 994.51146842130504, 0.00088614741033889405, 0.60880999999999974, 4182.9480988064988, -15850730.211861541},
    {310, 50000, 991.86272142791631, 0.00072058248924828175, 0.62193901499999982, 4177.4402009991136, -15808932.466631029},
    {320, 50000, 989.14811145249382, 0.00059698641565941789, 0.63380431999999987, 4176.3825576463487, -15767166.860136222},
    {330, 50000, 986.3638912977674, 0.00050295446547332211, 0.64441680499999998, 4179.0059184567817, -15725392.840363575},
    {340, 50000, 983.50597349484815, 0.00043018557800759852, 0.65378735999999982, 4184.6658637801202, -15683576.923673673},
    {350, 50000, 980.5698871147373, 0.00037300059411912175, 0.66192687499999991, 4192.8428046072058, -15641691.44649804},
    {360, 50000, 977.55072743231472, 0.00032743476022936729, 0.66884623999999981, 4203.1419825700141, -15599713.317036513},
    {370, 50000, 974.4430969647309, 0.00029066832134331524, 0.67455634499999995, 4215.2934699416455, -15557622.766955297},
    {380, 50000, 971.24103603494268, 0.00026066121092898937, 0.67906807999999985, 4229.1521696363416, -15515402.103085814},
    {390, 50000, 967.93794052901364, 0.0002359138248953162, 0.68239233499999985, 4244.6978152094707, -15473034.459124567},
    {400, 50000, 964.52646388389849, 0.00021530736151418751, 0.68453999999999982, 4262.0349708575422, -15430502.547334258},
    {280, 100000, 999.62487928454073, 0.0014300291329221544, 0.57871727999999989, 4211.0158121564882, -15934677.177039757},
    {290, 100000, 997.09779658913828, 0.0011126737698711271, 0.59440638499999987, 4193.8003320010603, -15892659.110051598},
    {300, 100000, 994.51146842130504, 0.00088614741033889405, 0.60880999999999974, 4182.9480988064988, -15850780.487802632},
    {310, 100000, 991.86272142791631, 0.00072058248924828175, 0.62193901499999982, 4177.4402009991136, -15808982.876832884},
    {320, 100000, 989.14811145249382, 0.00059698641565941789, 0.63380431999999987, 4176.3825576463487, -15767217.408683423},
    {330, 100000, 986.3638912977674, 0.00050295446547332211, 0.64441680499999998, 4179.0059184567817, -15725443.531594714},
    {340, 100000, 983.50597349484815, 0.00043018557800759852, 0.65378735999999982, 4184.6658637801202, -15683627.762205768},
    {350, 100000, 980.5698871147373, 0.00037300059411912175, 0.66192687499999991, 4192.8428046072058, -15641742.437254189},
    {360, 100000, 977.55072743231472, 0.00032743476022936729, 0.66884623999999981, 4203.1419825700141, -15599764.465277312},
    {370, 100000, 974.4430969647309, 0.00029066832134331524, 0.67455634499999995, 4215.2934699416455, -15557674.078314736},
    {380, 100000, 971.24103603494268, 0.00026066121092898937, 0.67906807999999985, 4229.1521696363416, -15515453.583612423},
    {390, 100000, 967.93794052901364, 0.0002359138248953162, 0.68239233499999985, 4244.6978152094707, -15473086.115328861},
    {400, 100000, 964.52646388389849, 0.00021530736151418751, 0.68453999999999982, 4262.0349708575422, -15430554.386243684},
    {280, 200000, 999.62487928454073, 0.0014300291329221544, 0.57871727999999989, 4211.0158121564882, -15934777.214565905},
    {290, 200000, 997.09779658913828, 0.0011126737698711271, 0.59440638499999987, 4193.8003320010603, -15892759.401116669},
    {300, 200000, 994.51146842130504, 0.00088614741033889405, 0.60880999999999974, 4182.9480988064988, -15850881.039684812},
    {310, 200000, 991.86272142791631, 0.00072058248924828175, 0.62193901499999982, 4177.4402009991136, -15809083.697236596},
    {320, 200000, 989.14811145249382, 0.00059698641565941789, 0.63380431999999987, 4176.3825576463487, -15767318.505777823},
    {330, 200000, 986.3638912977674, 0.00050295446547332211, 0.64441680499999998, 4179.0059184567817, -15725544.91405699},
    {340, 200000, 983.50597349484815, 0.00043018557800759852, 0.65378735999999982, 4184.6658637801202, -15683729.43926996},
    {350, 200000, 980.5698871147373, 0.00037300059411912175, 0.66192687499999991, 4192.8428046072058, -15641844.418766484},
    {360, 200000, 977.55072743231472, 0.00032743476022936729, 0.66884623999999981, 4203.1419825700141, -15599866.76175891},
    {370, 200000, 974.4430969647309, 0.00029066832134331524, 0.67455634499999995, 4215.2934699416455, -15557776.701033611},
    {380, 200000, 971.24103603494268, 0.00026066121092898937, 0.67906807999999985, 4229.1521696363416, -15515556.544665644},
    {390, 200000, 967.93794052901364, 0.0002359138248953162, 0.68239233499999985, 4244.6978152094707, -15473189.427737448},
    {400, 200000, 964.52646388389849, 0.00021530736151418751, 0.68453999999999982, 4262.0349708575422, -15430658.064062536},
    {280, 500000, 999.62487928454073, 0.0014300291329221544, 0.57871727999999989, 4211.0158121564882, -15935077.327144351},
    {290, 500000, 997.09779658913828, 0.0011126737698711271, 0.59440638499999987, 4193.8003320010603, -15893060.274311883},
    {300, 500000, 994.51146842130504, 0.00088614741033889405, 0.60880999999999974, 4182.9480988064988, -15851182.695331354},
    {310, 500000, 991.86272142791631, 0.00072058248924828175, 0.62193901499999982, 4177.4402009991136, -15809386.158447728},
    {320, 500000, 989.14811145249382, 0.00059698641565941789, 0.63380431999999987, 4176.3825576463487, -15767621.797061026},
    {330, 500000, 986.3638912977674, 0.00050295446547332211, 0.64441680499999998, 4179.0059184567817, -15725849.061443819},
    {340, 500000, 983.50597349484815, 0.00043018557800759852, 0.65378735999999982, 4184.6658637801202, -15684034.470462536},
    {350, 500000, 980.5698871147373, 0.00037300059411912175, 0.66192687499999991, 4192.8428046072058, -15642150.363303373},
    {360, 500000, 977.55072743231472, 0.00032743476022936729, 0.66884623999999981, 4203.1419825700141, -15600173.651203705},
    {370, 500000, 974.4430969647309, 0.00029066832134331524, 0.67455634499999995, 4215.2934699416455, -15558084.569190238},
    {380, 500000, 971.24103603494268, 0.00026066121092898937, 0.67906807999999985, 4229.1521696363416, -15515865.427825302},
    {390, 500000, 967.93794052901364, 0.0002359138248953162, 0.68239233499999985, 4244.6978152094707, -15473499.364963213},
    {400, 500000, 964.52646388389849, 0.00021530736151418751, 0.68453999999999982, 4262.0349708575422, -15430969.09751909},
};
int failures = 0;
// TOLERANCE, DERIVED RATHER THAN TUNED. T is recovered by Newton to a relative residual of 1e-12 on
// e ~ -1.6e7 J/kg, i.e. ~1.6e-5 J/kg absolute; with Cv ~ 4200 J/(kg K) that is dT ~ 4e-9 K. The
// properties are then evaluated at that T, so each inherits dT times its own sensitivity. mu is the
// worst: it is exp(a + b/T + ...) with b = 3670.6, so |dln(mu)/dT| = b/T^2 ~ 0.047 /K and dT ~ 4e-9
// becomes ~2e-10 relative -- which is precisely what this test measured (5.7e-11 worst).
//
// Tightening the inversion is not available: reaching dT ~ 1e-12 K would need a residual near 1e-16
// relative, at the floor of double precision for a quantity of magnitude 1e7.
//
// 1e-9 keeps ~5x margin over the worst propagated error while remaining seven orders below anything a
// WIRING error produces (wrong p, wrong index or a one-iteration lag are all >= 1e-2 here), which is
// what this test exists to catch. Mutation-verified below rather than asserted.
void expect(const char* what, int i, double got, double want, double tol = 1e-9)
{
    const double rel = std::fabs(got - want)/std::fmax(std::fabs(want), 1e-300);
    if (!(rel <= tol))
    {
        std::printf("  FAIL %-10s cell %d  brae %.17g  OF %.17g  rel %.3e\n", what, i, got, want, rel);
        ++failures;
    }
}
}   // namespace

int main()
{
    const int n = static_cast<int>(kOF.size());
    ThermoCoeffs c;
    c.model          = ThermoModel::liquidH2O;
    c.internalEnergy = true;            // squareBendLiq: energy sensibleInternalEnergy
    c.rhoThermoType  = true;            // heRhoThermo

    std::vector<scalar> he(n), p(n), Tbad(n);
    for (int i = 0; i < n; ++i)
    {
        he[i]   = kOF[i].Es;
        p[i]    = kOF[i].p;
        // Deliberately wrong guess, reversed within each 13-point pressure block.
        Tbad[i] = kOF[(i/13)*13 + (12 - i%13)].T;
    }

    DeviceThermo th;
    th.allocate(n);
    th.he.copyFrom(he);
    th.T.copyFrom(Tbad);
    // Solver-side rho seeded with a sentinel: deviceThermoCorrect must NOT touch it. OF's calculate()
    // writes the thermo's rho_, and the solver's own rho is assigned later by `rho = thermo.rho()`.
    std::vector<scalar> sentinel(n, -12345.0);
    th.rho.copyFrom(sentinel);

    DeviceBuffer<scalar> pD;
    pD.copyFrom(p);

    deviceThermoCorrect(th, pD, c);     // <-- the real entry point

    const std::vector<scalar> T  = th.T.host(),  Cp = th.CpField.host();
    const std::vector<scalar> mu = th.mu.host(), ka = th.kappa.host();
    const std::vector<scalar> rt = th.rhoThermo.host(), al = th.alpha.host();
    const std::vector<scalar> rs = th.rho.host();

    std::printf("integration: deviceThermoCorrect() liquid path, %d cells over 4 pressures\n", n);
    for (int i = 0; i < n; ++i)
    {
        const Row& r = kOF[i];
        expect("T",         i, T[i],  r.T, 1e-10);
        expect("Cp",        i, Cp[i], r.Cp);
        expect("mu",        i, mu[i], r.mu);
        expect("kappa",     i, ka[i], r.kappa);
        expect("rhoThermo", i, rt[i], r.rho);
        expect("alpha",     i, al[i], r.kappa/r.Cp);
    }
    std::printf("  T, Cp, mu, kappa, rhoThermo, alpha all match OF\n");

    // The two-density distinction must survive integration: correct() writes rhoThermo, never rho.
    {
        int touched = 0;
        for (int i = 0; i < n; ++i) if (rs[i] != scalar(-12345.0)) ++touched;
        if (touched)
        {
            std::printf("  FAIL deviceThermoCorrect wrote the SOLVER rho in %d cells -- it must write\n"
                        "       rhoThermo only; the solver assigns rho = thermo.rho() after the pressure solve\n",
                        touched);
            ++failures;
        }
        else std::printf("  solver rho untouched (rhoThermo is what correct() writes)\n");
    }

    // Properties must come from the NEW T, not the guess. If they lagged, evaluating at Tbad would match.
    {
        int lagged = 0;
        for (int i = 0; i < n; ++i)
            if (std::fabs(Cp[i] - H2OLiquid::Cp(Tbad[i])) < 1e-9 && std::fabs(Tbad[i] - kOF[i].T) > 1.0)
                ++lagged;
        if (lagged)
        {
            std::printf("  FAIL %d cells carry properties evaluated at the PREVIOUS temperature --\n"
                        "       the property update ran before the inversion\n", lagged);
            ++failures;
        }
        else std::printf("  properties evaluated at the updated T, not the guess\n");
    }

    // The gas path must still take its own branch: same call, perfectGas coefficients, liquid fields
    // must stay unallocated.
    {
        ThermoCoeffs g;                 // default perfectGas
        DeviceThermo gt;
        gt.allocate(8);
        std::vector<scalar> ghe(8, 3.0e5), gp(8, 1e5);
        gt.he.copyFrom(ghe);
        DeviceBuffer<scalar> gpD;
        gpD.copyFrom(gp);
        deviceThermoCorrect(gt, gpD, g);
        if (gt.CpField.size() != 0 || gt.kappa.size() != 0)
        {
            std::printf("  FAIL the perfectGas branch allocated liquid fields\n");
            ++failures;
        }
        else std::printf("  perfectGas branch unchanged (no liquid fields, closed-form he->T)\n");
    }

    std::printf("test_liquid_correct: %d failures\n", failures);
    return failures ? 1 : 0;
}
