#pragma once
// thermo_model.cuh -- hConst with sensibleEnthalpy.
//
// The energy equation solves for he, not T, because he is what is conserved across a boundary where Cp
// changes; T is then recovered from it. With hConst that inversion is exact and closed-form. A janaf
// thermo would make Cp = Cp(T) and turn heToT into a Newton iteration -- which is why every consumer
// goes through these two functions rather than dividing by Cp itself.

#include "cf_types.cuh"
#include "thermo_types.cuh"

namespace brae {

// Sensible enthalpy from temperature. OF: he = Cp*T + Hf for sensibleEnthalpy.
BRAE_HD inline scalar hConstTToHe(
    scalar T,
    const ThermoCoeffs& c)
{
    return c.Cp * T + c.Hf;
}

// Temperature from sensible enthalpy, the exact inverse of the above.
BRAE_HD inline scalar hConstHeToT(
    scalar he,
    const ThermoCoeffs& c)
{
    return (he - c.Hf) / c.Cp;
}

} // namespace brae
