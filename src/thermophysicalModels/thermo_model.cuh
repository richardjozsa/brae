#pragma once
// thermo_model.cuh -- hConst with sensibleEnthalpy (OF "h") or sensibleInternalEnergy (OF "e").
//
// The energy equation solves for he, not T, because he is what is conserved across a boundary where Cp
// changes; T is then recovered from it. With hConst that inversion is exact and closed-form. A janaf
// thermo would make Cp = Cp(T) and turn heToT into a Newton iteration -- which is why every consumer
// goes through these two functions rather than dividing by Cp itself.

#include "cf_types.cuh"
#include "thermo_types.cuh"

namespace brae {

// Cv = Cp - CpMCv, and perfectGas::CpMCv = R (OF HtoEthermo.H + perfectGasI.H). Derived, never stored,
// so it cannot disagree with Cp and R.
BRAE_HD inline scalar thermoCv(const ThermoCoeffs& c)
{
    return c.Cp - c.R;
}

// The energy variable from temperature. OF's hConstThermo measures the SENSIBLE enthalpy about a
// reference point (hConstThermoI.H:Hs):
//
//     Hs(T) = Cp*(T - Tref) + Href        [+ EquationOfState::H, which perfectGas defines as 0]
//     Ha(T) = Hs(T) + Hc(),  Hc() = Hf    <- the heat of formation belongs to the ABSOLUTE enthalpy only
//
//   sensibleEnthalpy:       he = Hs      = Cp*(T - Tref) + Href
//   sensibleInternalEnergy: he = Es = Hs - p/rho = Hs - R*T = Cv*T - Cp*Tref + Href
//
// WHY THE Tref OFFSET IS NOT COSMETIC. he appears in the energy equation as
//
//     div(phi, he) - laplacian(alphaEff, he) = -div(phi, Ekp)
//
// which is NOT invariant under he -> he + C: the shift leaves a spurious source C*div(phi). At
// convergence div(phi) = 0 and the offset cancels, but SIMPLE only reaches that state through iterates
// where div(phi) is the continuity error, and C = Cp*Tstd ~ 3.0e5 J/kg multiplies it. brae used
// he = Cv*T, i.e. OF's he PLUS exactly Cp*Tstd (measured: 2.996407e+05 against Cp*Tstd = 2.996408e+05).
// On a slow well-behaved case the injected source is small and the next iteration repairs it; on
// NACA0012 the first iteration's continuity error made it dominant and the run diverged.
//
// The branch is uniform across every thread in the launch (it is a case-level setting), so it costs
// nothing on the GPU and keeps ONE definition of the inversion rather than two parallel kernels.
BRAE_HD inline scalar hConstTToHe(
    scalar T,
    const ThermoCoeffs& c)
{
    const scalar Hs = c.Cp * (T - c.Tref) + c.Href;
    return c.internalEnergy ? Hs - c.R * T : Hs;
}

// Temperature from the energy variable, the exact inverse of the above. Both branches share a numerator
// because Es and Hs differ only by R*T, which merges into Cv on the internal-energy side:
//     he = Cpv*T - Cp*Tref + Href   with Cpv = Cv (e) or Cp (h).
BRAE_HD inline scalar hConstHeToT(
    scalar he,
    const ThermoCoeffs& c)
{
    return (he - c.Href + c.Cp * c.Tref) / (c.internalEnergy ? thermoCv(c) : c.Cp);
}

// OF heThermo::alphaEff = CpByCpv*(alpha + alphat). CpByCpv is 1 for sensibleEnthalpy and gamma = Cp/Cv
// for sensibleInternalEnergy (OF sensibleInternalEnergy::CpByCpv -> thermo::gamma).
BRAE_HD inline scalar thermoCpByCpv(const ThermoCoeffs& c)
{
    return c.internalEnergy ? c.Cp / thermoCv(c) : scalar(1);
}

} // namespace brae
