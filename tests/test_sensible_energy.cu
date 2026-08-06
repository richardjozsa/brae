// The energy variable brae transports must be OpenFOAM's SENSIBLE energy, measured about Tref.
//
// OF's hConstThermo (hConstThermoI.H) defines
//
//     Hs(T) = Cp*(T - Tref_) + Hsref_        Tref_ defaults to Tstd = 298.15, Hsref_ to 0
//     Ha(T) = Hs(T) + Hc(),  Hc() = Hf       <- Hf belongs to the ABSOLUTE enthalpy only
//
//   sensibleEnthalpy:        he = Hs
//   sensibleInternalEnergy:  he = Es = Hs - p/rho = Hs - R*T
//
// brae used he = Cv*T ("e") and he = Cp*T + Hf ("h"): the Tref offset dropped, and Hf wrongly added to a
// SENSIBLE enthalpy. Measured against OF on aerofoilNACA0012, brae's he was OF's he plus exactly
// Cp*Tstd = 2.996407e+05 J/kg.
//
// WHY AN OFFSET IN he IS NOT COSMETIC -- the point this test exists to protect. The energy equation
//
//     div(phi, he) - laplacian(alphaEff, he) = -div(phi, Ekp)
//
// is NOT invariant under he -> he + C: the shift leaves a spurious source C*div(phi). At convergence
// div(phi) = 0 and it cancels, which is why a slow, well-behaved case still lands on the right answer and
// the defect survived. SIMPLE only REACHES that state through iterates where div(phi) is the continuity
// error, and C ~ 3.0e5 J/kg multiplies it every time.
//
// The negative control is the whole test: assertions that only pin the ROUND TRIP T(he(T)) == T pass just
// as happily with the old formula, because it was self-consistent. What separates the two is the absolute
// value of he, so that is what is checked -- against numbers taken from OF itself.

#include "thermo_model.cuh"
#include "thermo_types.cuh"
#include "foam_constants.cuh"
#include <cmath>
#include <cstdio>

using namespace brae;

static int failures = 0;

static void expectNear(const char* what, scalar got, scalar want, scalar tol)
{
    const scalar err = std::fabs(got - want) / std::max(std::fabs(want), (scalar)1.0);
    if (!(err <= tol))
    {
        std::printf("  FAIL %-46s got %.10e  want %.10e  relerr %.3e (tol %.1e)\n",
                    what, (double)got, (double)want, (double)err, (double)tol);
        ++failures;
    }
    else
    {
        std::printf("  ok   %-46s %.10e\n", what, (double)got);
    }
}

int main()
{
    // aerofoilNACA0012's mixture, the case the defect was found on.
    ThermoCoeffs c;
    c.Cp = 1005.0;
    c.R  = foamRR() / 28.9;          // molWeight 28.9, resolved the way OF resolves RR
    c.Tref = foamTstd();             // hConstThermo's default Tref
    c.Href = 0.0;
    c.Hf   = 0.0;

    const scalar Cv = thermoCv(c);
    std::printf("Cp=%.4f R=%.6f Cv=%.6f Tref=%.5f\n", (double)c.Cp, (double)c.R, (double)Cv, (double)c.Tref);

    // --- sensibleInternalEnergy -------------------------------------------------------------------
    // OF's own value, read out of a dumpPEqn run on aerofoilNACA0012 at T = 298 K: he = -8.588473e+04.
    c.internalEnergy = true;
    expectNear("Es(298) matches OF's measured he", hConstTToHe(298.0, c), -8.588473e+04, 2e-6);
    expectNear("Es(T) == Cv*T - Cp*Tref",          hConstTToHe(298.0, c), Cv*298.0 - c.Cp*c.Tref, 1e-12);
    expectNear("Es(Tref) == -R*Tref",              hConstTToHe(c.Tref, c), -c.R*c.Tref, 1e-12);
    expectNear("T(Es(298)) round trip",            hConstHeToT(hConstTToHe(298.0, c), c), 298.0, 1e-12);

    // The negative control. The old formula was he = Cv*T, which is OF's he PLUS Cp*Tref. If someone
    // reinstates it, the assertions above fail -- this proves they CAN fail rather than being tautologies.
    const scalar oldForm = Cv * 298.0;
    const scalar gap     = oldForm - hConstTToHe(298.0, c);
    expectNear("the old he=Cv*T differs by exactly Cp*Tref", gap, c.Cp*c.Tref, 1e-12);
    if (std::fabs(gap) < 1.0)
    {
        std::printf("  FAIL negative control is inert: the old and new formulas agree\n");
        ++failures;
    }

    // --- sensibleEnthalpy -------------------------------------------------------------------------
    c.internalEnergy = false;
    expectNear("Hs(Tref) == 0",             hConstTToHe(c.Tref, c), 0.0, 1e-12);
    expectNear("Hs(T) == Cp*(T - Tref)",    hConstTToHe(400.0, c), c.Cp*(400.0 - c.Tref), 1e-12);
    expectNear("T(Hs(400)) round trip",     hConstHeToT(hConstTToHe(400.0, c), c), 400.0, 1e-12);

    // Hf must NOT enter the sensible enthalpy: OF puts it in Ha = Hs + Hc(), and EEqn transports Hs.
    // brae added it, which is silent whenever a case leaves Hf at 0 -- i.e. almost always.
    ThermoCoeffs withHf = c;
    withHf.Hf = 2.5e6;                      // a combustion-scale heat of formation
    expectNear("Hf does not shift the sensible enthalpy",
               hConstTToHe(400.0, withHf), hConstTToHe(400.0, c), 1e-12);
    withHf.internalEnergy = true;
    ThermoCoeffs noHf = withHf; noHf.Hf = 0.0;
    expectNear("Hf does not shift the sensible internal energy",
               hConstTToHe(400.0, withHf), hConstTToHe(400.0, noHf), 1e-12);

    // --- an explicit Tref/Href, as hConstThermo.C:40-41 reads them --------------------------------
    ThermoCoeffs shifted = c;
    shifted.Tref = 250.0;
    shifted.Href = 1234.0;
    expectNear("Hs honours an explicit Tref/Href",
               hConstTToHe(400.0, shifted), c.Cp*(400.0 - 250.0) + 1234.0, 1e-12);
    expectNear("T(Hs) round trip with an explicit Tref/Href",
               hConstHeToT(hConstTToHe(400.0, shifted), shifted), 400.0, 1e-12);

    std::printf("%s: %d failures\n", failures ? "FAILED" : "test_sensible_energy", failures);
    return failures ? 1 : 0;
}
