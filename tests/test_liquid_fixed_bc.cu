// OF's fvPatchField::fixesValue() branch on the LIQUID thermo boundary: when a patch prescribes a
// temperature, that temperature is the input and the energy is the derived quantity -- re-derived from
// the CURRENT boundary pressure on every thermo.correct().
//
// WHY THIS NEEDS ITS OWN TEST RATHER THAN A TUTORIAL COMPARISON. The bug this covers is invisible at
// iteration 1, because at set-up the converted he_b is correct by construction. It only appears once p
// moves away from its initial value, which is exactly the state a single-iteration field comparison
// cannot reach. So the test drives the pressure directly instead of waiting for a solver to do it:
//
//     T_fixed held at one value
//     p_b swept 50 kPa -> 500 kPa
//     assert T_b is STILL exactly T_fixed, and he_b is OF's Es(p_b, T_fixed) at every stop
//
// THE ORACLE IS OPENFOAM. Every Es below is from dissect/liqref, which evaluates
// thermophysicalPropertiesSelector<liquidProperties>::Es(p, T) with OF's own H2O -- not brae's forward
// evaluation, and not hand arithmetic. The pressures are the ones OF was asked to tabulate.
//
// THE PRESSURE DEPENDENCE IS THE WHOLE POINT. Es = h(T) - p/rho(T): the h(T) part is what a
// sensibleEnthalpy case would see and is p-independent, so a test run at a single pressure would pass
// against a completely stale implementation. Sweeping p is what makes the assertion bite -- note the
// four Es values at T=300 differ only in the 8th significant digit, which is why the tolerance below is
// relative to the SPREAD and not to Es itself.
#include "device_thermo.cuh"
#include "nsrds_functions.cuh"
#include "thermo_types.cuh"
#include <cstdio>
#include <cmath>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

struct Ref { double p, Es300, Es350; };

// OpenFOAM v2412, dissect/liqref: Es(p, T) for H2O.
const std::vector<Ref> kOF = {
    { 50000, -15850730.211861541, -15641691.446498040},
    {100000, -15850780.487802632, -15641742.437254189},
    {200000, -15850881.039684812, -15641844.418766484},
    {500000, -15851182.695331354, -15642150.363303373},
};

// A minimal fixedValue boundary: n faces, each backed by its own cell.
DeviceBoundary makeFixedBoundary(int n, double refVal)
{
    DeviceBoundary db;
    db.n = n;
    db.bcType.copyFrom(std::vector<label>(n, 1));            // 1 = fixedValue
    db.faceCell.copyFrom([&]{ std::vector<label> f(n); for (int i = 0; i < n; ++i) f[i] = i; return f; }());
    db.refValue.copyFrom(std::vector<scalar>(n, refVal));
    db.valueFraction.copyFrom(std::vector<scalar>(n, 0.0));
    db.deltaCoeffs.copyFrom(std::vector<scalar>(n, 1.0));
    db.magSf.copyFrom(std::vector<scalar>(n, 1.0));
    return db;
}

ThermoCoeffs liquidCoeffs()
{
    ThermoCoeffs c;
    c.model = ThermoModel::liquidH2O;
    c.internalEnergy = true;       // squareBendLiq: sensibleInternalEnergy, so Es depends on p
    return c;
}

}   // namespace

int main()
{
    const int n = 8;
    const double Tfix[2] = {300.0, 350.0};

    std::printf("liquid fixed-T boundary: T stays exact and he tracks p (OF fixesValue branch)\n");

    for (int t = 0; t < 2; ++t)
    {
        const double Tf = Tfix[t];
        // Deliberately seed he_b at the WRONG pressure -- 50 kPa for the T=300 sweep and 500 kPa for the
        // T=350 one -- so a implementation that keeps the set-up conversion starts out visibly stale and
        // has to be corrected at every stop rather than only at the one that happens to match.
        const double seedEs = (t == 0) ? kOF.front().Es300 : kOF.back().Es350;

        DeviceBoundary dbHe = makeFixedBoundary(n, seedEs);
        DeviceBuffer<scalar> he, Tcell, pCell, TBnd;
        he.copyFrom(std::vector<scalar>(n, seedEs));
        // The cell temperature is the Newton GUESS on non-fixed faces. Set it far away so that if the
        // fixesValue branch were skipped and the code fell through to the inversion, it could not
        // accidentally land on Tf.
        Tcell.copyFrom(std::vector<scalar>(n, 420.0));
        pCell.copyFrom(std::vector<scalar>(n, 0.0));

        DeviceBuffer<label>  fixMask;
        DeviceBuffer<scalar> fixVal;
        fixMask.copyFrom(std::vector<label>(n, 1));
        fixVal.copyFrom(std::vector<scalar>(n, Tf));

        const ThermoCoeffs c = liquidCoeffs();

        for (const Ref& r : kOF)
        {
            const double wantEs = (t == 0) ? r.Es300 : r.Es350;
            DeviceBoundary dbP = makeFixedBoundary(n, r.p);

            deviceThermoTBoundary(dbP, pCell, dbHe, he, c, &Tcell, &fixMask, &fixVal, TBnd);

            const std::vector<scalar> Tb  = TBnd.host();
            const std::vector<scalar> heb = dbHe.refValue.host();

            for (int i = 0; i < n; ++i)
            {
                // T must be EXACTLY the prescribed value. Not "within a tolerance": a prescribed
                // temperature is copied, not solved for, so any deviation at all means the code took
                // the inversion path.
                if (Tb[i] != Tf)
                {
                    std::printf("  FAIL T_b face %d at p=%.0f: got %.15g, prescribed %.15g (drift %.3e K)\n",
                                i, r.p, (double)Tb[i], Tf, (double)Tb[i] - Tf);
                    ++failures;
                }
                const double rel = std::fabs(heb[i] - wantEs)/std::fabs(wantEs);
                if (!(rel <= 1e-15))
                {
                    std::printf("  FAIL he_b face %d at p=%.0f: brae %.15g  OF %.15g  rel %.3e\n",
                                i, r.p, (double)heb[i], wantEs, rel);
                    ++failures;
                }
            }
        }
        std::printf("  T=%.0f K: T_b exact and he_b == OF Es at p = 50/100/200/500 kPa\n", Tf);
    }

    // -----------------------------------------------------------------------------------------------
    // NEGATIVE CONTROL / MUTATION. Drop the fixesValue branch (pass no mask, which is what the code did
    // before this fix) and invert the STALE he_b at the new pressure. The recovered temperature must
    // drift, and by the amount the thermodynamics predicts:
    //
    //     e stale by  dp/rho    ->    dT = dp/(rho*Cp)
    //
    // This is the same arithmetic that identified the bug on squareBendLiq, where p ran 100 -> 150 kPa
    // and the inlet came out at 300.012019 K against a predicted 300.012013 K. If this control ever
    // stops firing, the assertions above are no longer testing anything.
    {
        const double Tf = 300.0, rho = 994.51146842130504, Cp = 4182.9480988064988;
        const double pSeed = 100000.0, pNow = 500000.0;
        const double seedEs = kOF[1].Es300;              // Es(100 kPa, 300 K)

        DeviceBoundary dbHe = makeFixedBoundary(n, seedEs);
        DeviceBoundary dbP  = makeFixedBoundary(n, pNow);
        DeviceBuffer<scalar> he, Tcell, pCell, TBnd;
        he.copyFrom(std::vector<scalar>(n, seedEs));
        Tcell.copyFrom(std::vector<scalar>(n, Tf));      // guess AT the answer, so only staleness moves it
        pCell.copyFrom(std::vector<scalar>(n, 0.0));

        deviceThermoTBoundary(dbP, pCell, dbHe, he, liquidCoeffs(), &Tcell, nullptr, nullptr, TBnd);

        const double got       = TBnd.host()[0];
        const double drift     = got - Tf;
        const double predicted = (pNow - pSeed)/(rho*Cp);

        if (std::fabs(drift - predicted)/std::fabs(predicted) > 1e-3)
        {
            std::printf("  FAIL negative control: dropping the fixesValue branch drifted %.6e K but the\n"
                        "       stale-energy prediction is %.6e K -- the test is not measuring staleness\n",
                        drift, predicted);
            ++failures;
        }
        else if (std::fabs(drift) < 1e-6)
        {
            std::printf("  FAIL negative control produced no drift (%.3e K): a stale he_b is going\n"
                        "       undetected, so the assertions above cannot fail\n", drift);
            ++failures;
        }
        else
        {
            std::printf("  negative control: without the branch, a stale he_b drifts T by %.6f K\n"
                        "                    (dp/(rho*Cp) predicts %.6f K)\n", drift, predicted);
        }
    }

    std::printf("test_liquid_fixed_bc: %d failures\n", failures);
    return failures ? 1 : 0;
}
