#pragma once
// equation_of_state.cuh -- perfectGas.
//
// Two functions, both device-callable, both taking ThermoCoeffs by value so a kernel can evaluate the
// state without dereferencing anything. psi is the quantity that makes the pressure equation
// compressible: it is d(rho)/d(p) at fixed T, and for a perfect gas that is simply 1/(R T), so
// rho = psi*p. The steady pressure equation carries a psi*p convection term that vanishes as psi -> 0,
// which is exactly how the incompressible limit is recovered.

#include "cf_types.cuh"
#include "thermo_types.cuh"

namespace brae {

// Compressibility d(rho)/d(p)|T for a perfect gas.
BRAE_HD inline scalar perfectGasPsi(
    scalar T,
    const ThermoCoeffs& c)
{
    return 1.0 / (c.R * T);
}

// Density from pressure and temperature. rho = p/(R T) = psi*p.
BRAE_HD inline scalar perfectGasRho(
    scalar p,
    scalar T,
    const ThermoCoeffs& c)
{
    return p / (c.R * T);
}

} // namespace brae
