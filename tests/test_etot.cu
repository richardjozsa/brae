// Standalone e -> T inversion (sensibleInternalEnergy), the form squareBendLiq actually transports.
//
// WHY THIS IS A SEPARATE TEST FROM test_hetot.cu. Same Newton loop, different F and dF/dT. The enthalpy
// test pins the h specialization; this one pins the internal-energy specialization and, critically, its
// PRESSURE dependence -- Es = Hs - p/rho, so an implementation that ignored p would still pass every
// enthalpy assertion.
//
// THE ORACLE IS OPENFOAM, through the exact type squareBendLiq's thermo is built from:
//   thermophysicalPropertiesSelector<liquidProperties>  (basic/rhoThermo/liquidThermo.H)
// so Es and Cv below are computed BY OF, not composed here from parts. 13 temperatures x 4 pressures,
// because a single pressure cannot distinguish e(T) from e(p,T).
//
// TWO FACTS READ FROM OF SOURCE, NOT INFERRED -- both counterintuitive enough to be worth naming:
//   liquidProperties::CpMCv(p,T) { return 0; }   so  Cv == Cp  for a liquid (liquidPropertiesI.H:104)
//   selector::Es(p,T) = Hs(p,T) - p/rho(p,T)                            (…SelectorI.H:152)
// OF then uses Cv as the Newton derivative even though d(Es)/dT also carries p*rho'/rho^2. We reproduce
// that choice; the fixed point is unaffected, and accepting on the residual is what makes that safe.
#include "device_thermo.cuh"
#include "nsrds_functions.cuh"
#include <cstdio>
#include <cstdarg>
#include <cmath>
#include <vector>

using namespace brae;

namespace {

struct Row { double T, p, Es, Cv; };
const std::vector<Row> kOF = {
    {280, 50000, -15934627.158276683, 4211.0158121564882},
    {290, 50000, -15892608.964519063, 4193.8003320010603},
    {300, 50000, -15850730.211861541, 4182.9480988064988},
    {310, 50000, -15808932.466631029, 4177.4402009991136},
    {320, 50000, -15767166.860136222, 4176.3825576463487},
    {330, 50000, -15725392.840363575, 4179.0059184567817},
    {340, 50000, -15683576.923673673, 4184.6658637801202},
    {350, 50000, -15641691.44649804, 4192.8428046072058},
    {360, 50000, -15599713.317036513, 4203.1419825700141},
    {370, 50000, -15557622.766955297, 4215.2934699416455},
    {380, 50000, -15515402.103085814, 4229.1521696363416},
    {390, 50000, -15473034.459124567, 4244.6978152094707},
    {400, 50000, -15430502.547334258, 4262.0349708575422},
    {280, 100000, -15934677.177039757, 4211.0158121564882},
    {290, 100000, -15892659.110051598, 4193.8003320010603},
    {300, 100000, -15850780.487802632, 4182.9480988064988},
    {310, 100000, -15808982.876832884, 4177.4402009991136},
    {320, 100000, -15767217.408683423, 4176.3825576463487},
    {330, 100000, -15725443.531594714, 4179.0059184567817},
    {340, 100000, -15683627.762205768, 4184.6658637801202},
    {350, 100000, -15641742.437254189, 4192.8428046072058},
    {360, 100000, -15599764.465277312, 4203.1419825700141},
    {370, 100000, -15557674.078314736, 4215.2934699416455},
    {380, 100000, -15515453.583612423, 4229.1521696363416},
    {390, 100000, -15473086.115328861, 4244.6978152094707},
    {400, 100000, -15430554.386243684, 4262.0349708575422},
    {280, 200000, -15934777.214565905, 4211.0158121564882},
    {290, 200000, -15892759.401116669, 4193.8003320010603},
    {300, 200000, -15850881.039684812, 4182.9480988064988},
    {310, 200000, -15809083.697236596, 4177.4402009991136},
    {320, 200000, -15767318.505777823, 4176.3825576463487},
    {330, 200000, -15725544.91405699, 4179.0059184567817},
    {340, 200000, -15683729.43926996, 4184.6658637801202},
    {350, 200000, -15641844.418766484, 4192.8428046072058},
    {360, 200000, -15599866.76175891, 4203.1419825700141},
    {370, 200000, -15557776.701033611, 4215.2934699416455},
    {380, 200000, -15515556.544665644, 4229.1521696363416},
    {390, 200000, -15473189.427737448, 4244.6978152094707},
    {400, 200000, -15430658.064062536, 4262.0349708575422},
    {280, 500000, -15935077.327144351, 4211.0158121564882},
    {290, 500000, -15893060.274311883, 4193.8003320010603},
    {300, 500000, -15851182.695331354, 4182.9480988064988},
    {310, 500000, -15809386.158447728, 4177.4402009991136},
    {320, 500000, -15767621.797061026, 4176.3825576463487},
    {330, 500000, -15725849.061443819, 4179.0059184567817},
    {340, 500000, -15684034.470462536, 4184.6658637801202},
    {350, 500000, -15642150.363303373, 4192.8428046072058},
    {360, 500000, -15600173.651203705, 4203.1419825700141},
    {370, 500000, -15558084.569190238, 4215.2934699416455},
    {380, 500000, -15515865.427825302, 4229.1521696363416},
    {390, 500000, -15473499.364963213, 4244.6978152094707},
    {400, 500000, -15430969.09751909, 4262.0349708575422},
};

int failures = 0;

void fail(const char* fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    std::printf("  FAIL ");
    std::vprintf(fmt, ap);
    va_end(ap);
    ++failures;
}

}   // namespace

int main()
{
    const int n = static_cast<int>(kOF.size());
    std::printf("e -> T inversion (sensibleInternalEnergy), %d OF (T,p) points across 4 pressures\n", n);
    const EnergyForm E = EnergyForm::sensibleInternalEnergy;

    // 1. Forward: brae's Es(p,T) and Cv must equal OF's, before any inversion is trusted.
    {
        double worstE = 0, worstC = 0;
        for (const Row& r : kOF)
        {
            const double e  = h2oEnergy(E, r.p, r.T);
            const double cv = h2oCpv(E, r.p, r.T);
            const double re = std::fabs(e - r.Es)/std::fabs(r.Es);
            const double rc = std::fabs(cv - r.Cv)/std::fabs(r.Cv);
            if (re > 1e-13) fail("Es(p=%.0f,T=%.0f) brae %.17g OF %.17g rel %.3e\n", r.p, r.T, e, r.Es, re);
            if (rc > 1e-13) fail("Cv(p=%.0f,T=%.0f) brae %.17g OF %.17g rel %.3e\n", r.p, r.T, cv, r.Cv, rc);
            worstE = std::fmax(worstE, re);
            worstC = std::fmax(worstC, rc);
        }
        std::printf("  forward: Es worst %.2e, Cv worst %.2e vs OF\n", worstE, worstC);
    }

    // 2. Cv == Cp for a liquid, asserted directly against OF's own Cv column.
    {
        for (const Row& r : kOF)
            if (std::fabs(H2OLiquid::Cp(r.T) - r.Cv)/r.Cv > 1e-13)
            { fail("Cp != OF's Cv at T=%.0f -- CpMCv is not 0 as OF states\n", r.T); break; }
        std::printf("  Cv == Cp confirmed against OF at every (p,T) (CpMCv = 0)\n");
    }

    // 3. Inversion with REVERSED guesses, per pressure.
    {
        double worstT = 0, worstR = 0;
        int worstIt = 0;
        for (int i = 0; i < n; ++i)
        {
            const Row& r = kOF[i];
            // reversed within this pressure block: 280 K target started from 400 K and vice versa
            const int blk = i/13, j = i%13;
            const double T0 = kOF[blk*13 + (12 - j)].T;
            const HeToTResult res = h2oEnergyToT(E, r.Es, r.p, T0);
            if (!res.converged)
                fail("p=%.0f T=%.0f from %.0f did not converge (res %.3e)\n", r.p, r.T, T0, res.residual);
            const double dT = std::fabs(res.T - r.T);
            if (dT > 1e-8) fail("p=%.0f T=%.0f from %.0f recovered %.12g (dT %.3e)\n", r.p, r.T, T0, res.T, dT);
            const double rr = std::fabs(h2oEnergy(E, r.p, res.T) - r.Es)/std::fabs(r.Es);
            if (rr > 1e-12) fail("p=%.0f T=%.0f: e(T_recovered) off by %.3e\n", r.p, r.T, rr);
            worstT = std::fmax(worstT, dT);
            worstR = std::fmax(worstR, rr);
            worstIt = res.iterations > worstIt ? res.iterations : worstIt;
        }
        std::printf("  inversion: worst |dT| %.2e K, worst residual %.2e, worst %d iterations\n",
                    worstT, worstR, worstIt);
    }

    // 4. Out-of-range guesses and an unreachable target, same contract as the enthalpy form.
    {
        const double p = 1e5, eT = h2oEnergy(E, p, 300.0);
        for (double bad : {-500.0, 0.0, 5000.0})
        {
            const HeToTResult r = h2oEnergyToT(E, eT, p, bad);
            if (!r.converged || std::fabs(r.T - 300.0) > 1e-8)
                fail("guess %.0f K gave %.10g (converged %d)\n", bad, r.T, (int)r.converged);
        }
        const HeToTResult bad = h2oEnergyToT(E, h2oEnergy(E, p, H2OLiquid::Tt) - 1e7, p, 300.0);
        if (bad.converged) fail("unreachable internal energy reported convergence at %.10g\n", bad.T);
        std::printf("  out-of-range guesses projected; unreachable target failed explicitly\n");
    }

    // 5. GPU vector, with a per-cell pressure field.
    {
        std::vector<scalar> e(n), pf(n), T0(n);
        for (int i = 0; i < n; ++i)
        {
            const int blk = i/13, j = i%13;
            e[i]  = kOF[i].Es;
            pf[i] = kOF[i].p;
            T0[i] = kOF[blk*13 + (12 - j)].T;
        }
        DeviceBuffer<scalar> eD, pD, T0D, TD, resD;
        DeviceBuffer<label> okD;
        eD.copyFrom(e); pD.copyFrom(pf); T0D.copyFrom(T0);
        deviceH2OEnergyToT(E, eD, &pD, T0D, TD, okD, resD);
        const std::vector<scalar> T = TD.host();
        const std::vector<label> ok = okD.host();
        int conv = 0;
        double worstT = 0;
        for (int i = 0; i < n; ++i)
        {
            if (ok[i]) ++conv;
            const double dT = std::fabs(T[i] - kOF[i].T);
            worstT = std::fmax(worstT, dT);
            if (dT > 1e-8) fail("GPU cell %d (p=%.0f): %.12g vs OF %.0f\n", i, kOF[i].p, T[i], kOF[i].T);
        }
        if (conv != n) fail("only %d of %d GPU cells converged\n", conv, n);
        std::printf("  GPU: %d/%d converged across 4 pressures, worst |dT| %.2e K\n", conv, n, worstT);
    }

    // 6. NEGATIVE CONTROLS. The pressure one is the point of this file: it is the only mutation that
    // distinguishes the internal-energy inversion from the enthalpy one.
    {
        int caught = 0;
        // (a) WRONG PRESSURE: invert a 5e5 Pa target at 1e5 Pa.
        {
            int wrong = 0;
            for (int i = 39; i < 52; ++i)      // the 5e5 Pa block
            {
                const HeToTResult r = h2oEnergyToT(E, kOF[i].Es, 1e5, 300.0);
                if (std::fabs(r.T - kOF[i].T) > 1e-8) ++wrong;
            }
            if (wrong == 13) ++caught;
            else std::printf("  note: wrong pressure changed only %d of 13 recovered temperatures\n", wrong);
        }
        // (b) ENTHALPY FORM used for an internal-energy target.
        {
            int wrong = 0;
            for (int i = 39; i < 52; ++i)
            {
                const HeToTResult r = h2oEnergyToT(EnergyForm::sensibleEnthalpy, kOF[i].Es, kOF[i].p, 300.0);
                if (std::fabs(r.T - kOF[i].T) > 1e-8) ++wrong;
            }
            if (wrong == 13) ++caught;
        }
        // (c) WRONG TARGET INDEX.
        {
            int wrong = 0;
            for (int i = 1; i < 13; ++i)
            {
                const HeToTResult r = h2oEnergyToT(E, kOF[0].Es, kOF[0].p, 300.0);
                if (std::fabs(r.T - kOF[i].T) > 1e-8) ++wrong;
            }
            if (wrong == 12) ++caught;
        }
        if (caught != 3)
            fail("negative controls: only %d of 3 detected (pressure / energy form / target index)\n", caught);
        else
            std::printf("  negative controls: wrong pressure, wrong energy form, wrong target all rejected\n");
    }

    std::printf("test_etot: %d failures\n", failures);
    return failures ? 1 : 0;
}
