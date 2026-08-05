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

// The energy variable from temperature.
//   sensibleEnthalpy:       he = Cp*T + Hf
//   sensibleInternalEnergy: he = Es = Hs - p/rho = Cp*T - R*T = Cv*T   (OF HtoEthermo.H, perfectGas)
// The branch is uniform across every thread in the launch (it is a case-level setting), so it costs
// nothing on the GPU and keeps ONE definition of the inversion rather than two parallel kernels.
BRAE_HD inline scalar hConstTToHe(
    scalar T,
    const ThermoCoeffs& c)
{
    return c.internalEnergy ? thermoCv(c) * T : c.Cp * T + c.Hf;
}

// Temperature from the energy variable, the exact inverse of the above.
BRAE_HD inline scalar hConstHeToT(
    scalar he,
    const ThermoCoeffs& c)
{
    return c.internalEnergy ? he / thermoCv(c) : (he - c.Hf) / c.Cp;
}

// OF heThermo::alphaEff = CpByCpv*(alpha + alphat). CpByCpv is 1 for sensibleEnthalpy and gamma = Cp/Cv
// for sensibleInternalEnergy (OF sensibleInternalEnergy::CpByCpv -> thermo::gamma).
BRAE_HD inline scalar thermoCpByCpv(const ThermoCoeffs& c)
{
    return c.internalEnergy ? c.Cp / thermoCv(c) : scalar(1);
}

} // namespace brae
