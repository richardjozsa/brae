#pragma once
// transport_model.cuh -- Sutherland and const viscosity, and the laminar thermal diffusivity.
//
// The branch on c.sutherland is resolved per cell rather than per case. That looks wasteful, but the
// alternative is templating every kernel that touches mu on the transport model, and the branch is
// uniform across the whole grid so it costs one predicted branch per thread, not a divergence.
//
// alpha here is OpenFOAM's laminar thermal diffusivity in kg/(m s) -- mu/Pr, NOT k/(rho Cp). That is the
// form the energy equation wants, because it solves he and multiplies by alphaEff directly.

#include "cf_types.cuh"
#include "thermo_types.cuh"

namespace brae {

// Dynamic viscosity. Sutherland: mu = As*sqrt(T)/(1 + Ts/T); const: mu = mu0.
BRAE_HD inline scalar transportMu(
    scalar T,
    const ThermoCoeffs& c)
{
    if (!c.sutherland) return c.mu0;
    return c.As * sqrt(T) / (1.0 + c.Ts / T);
}

// Laminar thermal diffusivity for the enthalpy equation, alpha = mu/Pr  [kg/(m s)].
BRAE_HD inline scalar transportAlpha(
    scalar mu,
    const ThermoCoeffs& c)
{
    return mu / c.Pr;
}

} // namespace brae
