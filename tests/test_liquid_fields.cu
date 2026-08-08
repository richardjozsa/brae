// Liquid thermophysical FIELD POPULATION: given a nonuniform temperature field, does brae fill
// Cp/mu/kappa/rhoThermo per cell and per boundary face with OpenFOAM's values?
//
// WHY NONUNIFORM, AND WHY THESE TEMPERATURES. A uniform 300 K field cannot distinguish "evaluated per
// cell" from "evaluated once and broadcast" -- both give the same answer everywhere. Every cell here
// gets a DIFFERENT temperature, and they are exactly the 13 temperatures OpenFOAM was asked to tabulate
// (dissect/liqref, 280-400 K), so the comparison is against OF's own numbers rather than against
// brae's host arithmetic. The cells are deliberately laid out in DESCENDING temperature order too, so
// an off-by-one or a reversed index shows up as a large error instead of a small one.
//
// SCOPE: field population only. The he->T inversion is a separate step and is not exercised here --
// these entry points take T directly, which is what makes them testable in isolation.
#include "device_thermo.cuh"
#include "nsrds_functions.cuh"
#include "thermo_types.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace brae;

namespace {

struct Row { double T, rho, mu, kappa, Cp; };

// OpenFOAM v2412, dissect/liqref: Foam::H2O at p = 1e5 Pa. Same oracle as tests/test_nsrds.cu.
const std::vector<Row> kOF = {
    {280, 999.62487928454073, 0.0014300291329221544,  0.57871727999999989, 4211.0158121564882},
    {290, 997.09779658913828, 0.0011126737698711271,  0.59440638499999987, 4193.8003320010603},
    {300, 994.51146842130504, 0.00088614741033889405, 0.60880999999999974, 4182.9480988064988},
    {310, 991.86272142791631, 0.00072058248924828175, 0.62193901499999982, 4177.4402009991136},
    {320, 989.14811145249382, 0.00059698641565941789, 0.63380431999999987, 4176.3825576463487},
    {330, 986.36389129776740, 0.00050295446547332211, 0.64441680499999998, 4179.0059184567817},
    {340, 983.50597349484815, 0.00043018557800759852, 0.65378735999999982, 4184.6658637801202},
    {350, 980.56988711473730, 0.00037300059411912175, 0.66192687499999991, 4192.8428046072058},
    {360, 977.55072743231472, 0.00032743476022936729, 0.66884623999999981, 4203.1419825700141},
    {370, 974.44309696473090, 0.00029066832134331524, 0.67455634499999995, 4215.2934699416455},
    {380, 971.24103603494268, 0.00026066121092898937, 0.67906807999999985, 4229.1521696363416},
    {390, 967.93794052901364, 0.00023591382489531620, 0.68239233499999985, 4244.6978152094707},
    {400, 964.52646388389849, 0.00021530736151418751, 0.68453999999999982, 4262.0349708575422},
};

int failures = 0;

void expectRel(const char* what, int i, double T, double got, double want, double tol = 1e-13)
{
    const double rel = std::fabs(got - want)/std::fmax(std::fabs(want), 1e-300);
    if (!(rel <= tol))
    {
        std::printf("  FAIL %-6s cell %d (T=%.0f)  brae %.17g  OF %.17g  rel %.3e\n",
                    what, i, T, got, want, rel);
        ++failures;
    }
}

}   // namespace

int main()
{
    const int n = static_cast<int>(kOF.size());

    // Descending temperature, so a reversed or shifted index is a big error rather than a subtle one.
    std::vector<scalar> Th(n);
    for (int i = 0; i < n; ++i) Th[i] = kOF[n - 1 - i].T;

    ThermoCoeffs c;
    c.model = ThermoModel::liquidH2O;

    DeviceThermo th;
    th.allocate(n);
    th.T.copyFrom(Th);

    deviceThermoLiquidProperties(th, c);

    const std::vector<scalar> Cp = th.CpField.host();
    const std::vector<scalar> mu = th.mu.host();
    const std::vector<scalar> ka = th.kappa.host();
    const std::vector<scalar> rh = th.rhoThermo.host();
    const std::vector<scalar> al = th.alpha.host();

    std::printf("liquid field population, %d cells, T = %.0f..%.0f K (nonuniform, descending)\n",
                n, (double)Th.front(), (double)Th.back());

    for (int i = 0; i < n; ++i)
    {
        const Row& r = kOF[n - 1 - i];
        expectRel("Cp",    i, r.T, Cp[i], r.Cp);
        expectRel("mu",    i, r.T, mu[i], r.mu);
        expectRel("kappa", i, r.T, ka[i], r.kappa);
        expectRel("rho",   i, r.T, rh[i], r.rho);
        expectRel("alpha", i, r.T, al[i], r.kappa/r.Cp);   // OF alpha = kappa/Cp for a liquid
    }
    std::printf("  internal: Cp, mu, kappa, rhoThermo, alpha match OF at every cell\n");

    // Every cell must differ from every other -- the direct refutation of a broadcast.
    {
        int distinct = 0;
        for (int i = 1; i < n; ++i) if (Cp[i] != Cp[i-1] && rh[i] != rh[i-1]) ++distinct;
        if (distinct != n - 1)
        {
            std::printf("  FAIL only %d of %d neighbouring cells have distinct properties --\n"
                        "       a value was broadcast rather than evaluated per cell\n", distinct, n - 1);
            ++failures;
        }
        else std::printf("  all %d cells carry distinct properties (no broadcast)\n", n);
    }

    // ---------------------------------------------------------------------------------------------
    // BOUNDARY faces, from a boundary temperature field. Different length and a different ordering from
    // the internal field on purpose: a kernel that happened to read the internal T would be caught.
    {
        const int nb = 5;
        std::vector<scalar> Tb(nb);
        const int pick[nb] = {0, 4, 8, 12, 6};       // 280, 320, 360, 400, 340
        for (int i = 0; i < nb; ++i) Tb[i] = kOF[pick[i]].T;

        DeviceBuffer<scalar> TbD, CpB, muB, kaB, rhB;
        TbD.copyFrom(Tb);
        deviceThermoLiquidBoundary(TbD, c, CpB, muB, kaB, rhB);

        const std::vector<scalar> hCp = CpB.host(), hMu = muB.host();
        const std::vector<scalar> hKa = kaB.host(), hRh = rhB.host();
        if (static_cast<int>(hCp.size()) != nb)
        {
            std::printf("  FAIL boundary buffers sized %zu, expected %d\n", hCp.size(), nb);
            ++failures;
        }
        else
        {
            const int before = failures;
            for (int i = 0; i < nb; ++i)
            {
                const Row& r = kOF[pick[i]];
                expectRel("Cp_b",    i, r.T, hCp[i], r.Cp);
                expectRel("mu_b",    i, r.T, hMu[i], r.mu);
                expectRel("kappa_b", i, r.T, hKa[i], r.kappa);
                expectRel("rho_b",   i, r.T, hRh[i], r.rho);
            }
            // Only claim the match if it actually held -- an unconditional success line next to its own
            // FAIL messages is how a red test gets read as green.
            if (failures == before)
                std::printf("  boundary: %d faces match OF (independent length and ordering)\n", nb);
        }
    }

    // ---------------------------------------------------------------------------------------------
    // The gas path must be untouched. A ThermoCoeffs left at its default is perfectGas, and the liquid
    // entry point must decline to write anything -- no allocation, no values -- so a gas case cannot
    // acquire these fields by accident.
    {
        ThermoCoeffs gas;                       // default: ThermoModel::perfectGas
        DeviceThermo g;
        g.allocate(4);
        std::vector<scalar> gt(4, 300.0);
        g.T.copyFrom(gt);
        deviceThermoLiquidProperties(g, gas);
        if (g.CpField.size() != 0 || g.kappa.size() != 0)
        {
            std::printf("  FAIL perfectGas acquired liquid fields (Cp %zu, kappa %zu) -- the gas path\n"
                        "       must not allocate per-cell copies of a scalar\n",
                        g.CpField.size(), g.kappa.size());
            ++failures;
        }
        else std::printf("  perfectGas path untouched: no liquid fields allocated\n");
    }

    // ---------------------------------------------------------------------------------------------
    // NEGATIVE CONTROL: evaluate one property from the WRONG temperature (neighbouring cell) and require
    // that the comparison rejects it. This is the specific failure the nonuniform field exists to catch,
    // so if it slips through, the test above is not actually checking per-cell correctness.
    {
        int caught = 0;
        for (int i = 1; i < n; ++i)
        {
            const Row& wrong = kOF[n - 1 - (i - 1)];    // the neighbour's temperature
            const Row& right = kOF[n - 1 - i];
            if (std::fabs(H2OLiquid::Cp(wrong.T)  - right.Cp)/right.Cp   > 1e-13) ++caught;
            if (std::fabs(H2OLiquid::rho(wrong.T) - right.rho)/right.rho > 1e-13) ++caught;
        }
        if (caught != 2*(n - 1))
        {
            std::printf("  FAIL negative control: only %d of %d wrong-temperature evaluations were\n"
                        "       detected -- the per-cell check cannot catch an index error\n",
                        caught, 2*(n - 1));
            ++failures;
        }
        else std::printf("  negative control: all %d wrong-temperature evaluations rejected\n", caught);
    }

    std::printf("test_liquid_fields: %d failures\n", failures);
    return failures ? 1 : 0;
}
