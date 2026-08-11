// Standalone h -> T inversion for the liquid path. Nothing in the solver calls this yet.
//
// THE TARGETS COME FROM OPENFOAM. hTarget[i] is OF's own H2O h(T) at a known temperature
// (dissect/liqref), so "did Newton recover the right T" is answered against OF and not against brae's
// own forward evaluation.
//
// THE GUESSES ARE DELIBERATELY BAD, AND REVERSED. hTarget[0] corresponds to 280 K and is started from
// 400 K; hTarget[12] corresponds to 400 K and is started from 280 K. A test where Tguess ~ Ttrue would
// pass with almost any code -- including code that simply returns its own initial guess. Reversing the
// pairing also makes an indexing error large and obvious rather than subtle.
//
// TWO ASSERTIONS PER CASE, and the second is the important one:
//     T_recovered ~ T_OF            (did we land on the right temperature)
//     h(T_recovered) ~ hTarget      (is the inversion self-consistent)
// The first alone can pass while the thermodynamics is inconsistent; the second is what actually says
// the equation h(T) = hTarget was solved.
#include "device_thermo.cuh"
#include "nsrds_functions.cuh"
#include <cstdio>
#include <cstdarg>
#include <cmath>
#include <vector>

using namespace brae;

namespace {

// OpenFOAM v2412, dissect/liqref: T and Foam::H2O::h(1e5, T).
struct Row { double T, h; };
const std::vector<Row> kOF = {
    {280, -15934577.139513608}, {290, -15892558.818986528}, {300, -15850679.935920451},
    {310, -15808882.056429174}, {320, -15767116.311589021}, {330, -15725342.149132438},
    {340, -15683526.085141577}, {350, -15641640.455741892}, {360, -15599662.168795714},
    {370, -15557571.455595860}, {380, -15515350.622559205}, {390, -15472982.802920273},
    {400, -15430450.708424833},
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
    std::printf("h -> T inversion, %d OF-generated targets, guesses reversed (worst case)\n", n);

    // ---------------------------------------------------------------------------------------------
    // 1. Host inversion, reversed guesses.
    {
        double worstT = 0, worstH = 0;
        int worstIter = 0;
        for (int i = 0; i < n; ++i)
        {
            const double hT = kOF[i].h;
            const double T0 = kOF[n - 1 - i].T;          // reversed: 280 K target started from 400 K
            const HeToTResult r = h2oHToT(hT, T0);

            if (!r.converged)
                fail("target T=%.0f from guess %.0f did not converge (residual %.3e, %d iters)\n",
                     kOF[i].T, T0, r.residual, r.iterations);

            const double dT = std::fabs(r.T - kOF[i].T);
            const double rh = std::fabs(H2OLiquid::h(r.T) - hT)/std::fabs(hT);
            if (dT > 1e-8)
                fail("target T=%.0f from guess %.0f recovered %.12g (dT %.3e K)\n",
                     kOF[i].T, T0, r.T, dT);
            if (rh > 1e-12)
                fail("target T=%.0f: h(T_recovered) off by %.3e relative\n", kOF[i].T, rh);

            worstT = std::fmax(worstT, dT);
            worstH = std::fmax(worstH, rh);
            worstIter = r.iterations > worstIter ? r.iterations : worstIter;
        }
        std::printf("  host: worst |dT| %.2e K, worst enthalpy residual %.2e, worst %d iterations\n",
                    worstT, worstH, worstIter);
    }

    // ---------------------------------------------------------------------------------------------
    // 2. Targets AT the valid-range bounds, where projection/clamping hides mistakes.
    {
        const double Tt = H2OLiquid::Tt, Tc = H2OLiquid::Tc;
        struct { const char* what; double Ttrue, T0; } cases[] = {
            {"at Tt, guessed from Tc",   Tt,        Tc},
            {"at Tc, guessed from Tt",   Tc,        Tt},
            {"just inside Tt",           Tt + 1e-3, Tc},
            {"just inside Tc",           Tc - 1e-3, Tt},
        };
        for (const auto& c : cases)
        {
            const double hT = H2OLiquid::h(c.Ttrue);
            const HeToTResult r = h2oHToT(hT, c.T0);
            const double dT = std::fabs(r.T - c.Ttrue);
            if (!r.converged || dT > 1e-6)
                fail("bound case '%s': recovered %.10g for true %.10g (dT %.3e, converged %d, res %.3e)\n",
                     c.what, r.T, c.Ttrue, dT, (int)r.converged, r.residual);
        }
        std::printf("  bounds: Tt=%.2f and Tc=%.2f recovered exactly, from the opposite bound\n", Tt, Tc);
    }

    // ---------------------------------------------------------------------------------------------
    // 3. Out-of-range guesses must be PROJECTED, not evaluated. A caller handing in a temperature from a
    // diverging outer iteration must not put h() outside the correlation's range.
    {
        const double hT = H2OLiquid::h(300.0);
        for (double bad : {-500.0, 0.0, 5000.0})
        {
            const HeToTResult r = h2oHToT(hT, bad);
            if (!r.converged || std::fabs(r.T - 300.0) > 1e-8)
                fail("guess %.0f K (outside [Tt,Tc]) gave %.10g, converged %d\n", bad, r.T, (int)r.converged);
        }
        std::printf("  out-of-range guesses (-500, 0, 5000 K) projected and still recovered 300 K\n");
    }

    // ---------------------------------------------------------------------------------------------
    // 4. Failure must be EXPLICIT. An unreachable enthalpy (far below h(Tt)) cannot be solved; the
    // result must say so rather than quietly return a clamped temperature as if it were an answer.
    {
        const double unreachable = H2OLiquid::h(H2OLiquid::Tt) - 1e7;
        const HeToTResult r = h2oHToT(unreachable, 300.0);
        if (r.converged)
            fail("an unreachable enthalpy reported convergence at T=%.10g\n", r.T);
        else
            std::printf("  unreachable target reported failure explicitly (T clamped to %.2f, residual %.2e)\n",
                        r.T, r.residual);
    }

    // ---------------------------------------------------------------------------------------------
    // 5. GPU vector: 13 targets, 13 reversed guesses, one inversion per cell.
    {
        std::vector<scalar> h(n), T0(n);
        for (int i = 0; i < n; ++i) { h[i] = kOF[i].h; T0[i] = kOF[n - 1 - i].T; }

        DeviceBuffer<scalar> hD, T0D, TD, resD;
        DeviceBuffer<label> okD;
        hD.copyFrom(h);
        T0D.copyFrom(T0);
        deviceH2OHToT(hD, T0D, TD, okD, resD);

        const std::vector<scalar> T = TD.host(), res = resD.host();
        const std::vector<label> ok = okD.host();
        int converged = 0, distinct = 0;
        double worstT = 0;
        for (int i = 0; i < n; ++i)
        {
            if (ok[i]) ++converged;
            const double dT = std::fabs(T[i] - kOF[i].T);
            worstT = std::fmax(worstT, dT);
            if (dT > 1e-8)
                fail("GPU cell %d: recovered %.12g, OF %.0f (dT %.3e)\n", i, T[i], kOF[i].T, dT);
            if (res[i] > 1e-12)
                fail("GPU cell %d: enthalpy residual %.3e\n", i, res[i]);
            if (i > 0 && T[i] != T[i-1]) ++distinct;
        }
        if (converged != n) fail("only %d of %d GPU cells converged\n", converged, n);
        if (distinct != n - 1)
            fail("only %d of %d neighbouring GPU cells differ -- a value was broadcast\n", distinct, n - 1);
        std::printf("  GPU: %d/%d converged, worst |dT| %.2e K, all cells distinct\n", converged, n, worstT);
    }

    // ---------------------------------------------------------------------------------------------
    // 6. NEGATIVE CONTROLS. Both of the failure modes worth guarding, checked in-process so the test
    // carries its own proof that it can go red.
    {
        int caught = 0;

        // (a) WRONG DERIVATIVE: Cp scaled by 1.01.
        //
        // BE PRECISE ABOUT WHAT CATCHES THIS, because it is easy to claim too much. A wrong derivative
        // changes the PATH, not the fixed point: Newton still converges to h(T) = hTarget, just more
        // slowly. So NEITHER stopping criterion detects it -- not |dT|, and not the enthalpy residual
        // either. What detects it is the ITERATION COUNT: the loop below re-derives the same inversion
        // with a perturbed slope and requires it to take strictly more steps than the real
        // implementation. If the real implementation is itself perturbed the two agree, the count no
        // longer increases, and this control fires. Verified by mutation (Cp*1.01 in h2oHToT -> red).
        //
        // The enthalpy residual earns its place against a different failure -- an h() inconsistent with
        // its own Cp -- which is covered directly by the dh/dT == Cp assertion in tests/test_nsrds.cu.
        {
            const double hT = kOF[0].h;                       // 280 K
            double T = 400.0;                                 // reversed guess
            int it = 0;
            for (; it < 50; ++it)
            {
                const double e = H2OLiquid::h(T) - hT;
                if (std::fabs(e)/std::fabs(hT) <= 1e-12) break;
                T = T - e/(H2OLiquid::Cp(T)*1.01);            // deliberately wrong dh/dT
            }
            const double correct = h2oHToT(hT, 400.0).iterations;
            if (it > correct) ++caught;                       // wrong slope => strictly more iterations
            else std::printf("  note: perturbed derivative did not change the iteration count\n");
        }

        // (b) WRONG TARGET/INDEX: invert every cell against hTarget[0] instead of its own.
        {
            int wrong = 0;
            for (int i = 1; i < n; ++i)
            {
                const HeToTResult r = h2oHToT(kOF[0].h, kOF[n - 1 - i].T);
                if (std::fabs(r.T - kOF[i].T) > 1e-8) ++wrong;
            }
            if (wrong == n - 1) ++caught;
        }

        if (caught != 2)
            fail("negative controls: only %d of 2 detected -- the test cannot catch a wrong derivative\n"
                 "       or a wrong target index\n", caught);
        else
            std::printf("  negative controls: wrong derivative and wrong target index both rejected\n");
    }

    std::printf("test_hetot: %d failures\n", failures);
    return failures ? 1 : 0;
}
