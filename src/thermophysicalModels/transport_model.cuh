#pragma once
// transport_model.cuh -- Sutherland and const viscosity, and the laminar thermal diffusivity.
//
// The branch on c.sutherland is resolved per cell rather than per case. That looks wasteful, but the
// alternative is templating every kernel that touches mu on the transport model, and the branch is
// uniform across the whole grid so it costs one predicted branch per thread, not a divergence.
//
// alpha here is OpenFOAM's laminar thermal diffusivity in kg/(m s) (OF's alphah), NOT k/(rho Cp). That is
// the form the energy equation wants, because it solves he and multiplies by alphaEff directly.
//
// CRITICAL: the two transport models compute it DIFFERENTLY, and it is not a detail.
//   constTransport:      alphah = mu/Pr                                  (constTransportI.H)
//   sutherlandTransport: alphah = kappa/Cp, kappa = mu*Cv*(1.32 + 1.77*R/Cv)   (sutherlandTransportI.H)
// The second is the modified Eucken relation and involves NO Prandtl number -- a sutherland dict in OF
// carries only As and Ts, never Pr. Using mu/Pr for sutherland is wrong by ~1.4% for air, and because
// the case never supplies Pr the wrong number comes from a default nobody wrote down.

#include "cf_types.cuh"
#include "thermo_types.cuh"
#include "thermo_model.cuh"   // thermoCv

namespace brae {

// Dynamic viscosity. Sutherland: mu = As*sqrt(T)/(1 + Ts/T); const: mu = mu0.
BRAE_HD inline scalar transportMu(
    scalar T,
    const ThermoCoeffs& c)
{
    if (!c.sutherland) return c.mu0;
    return c.As * sqrt(T) / (1.0 + c.Ts / T);
}

// Laminar thermal diffusivity [kg/(m s)], per OF's transport model. See the header note above.
BRAE_HD inline scalar transportAlpha(
    scalar mu,
    const ThermoCoeffs& c)
{
    if (!c.sutherland) return mu / c.Pr;                       // constTransport::alphah
    const scalar cv = thermoCv(c);
    const scalar kappa = mu * cv * (1.32 + 1.77 * c.R / cv);   // sutherlandTransport::kappa (Eucken)
    return kappa / c.Cp;                                       // sutherlandTransport::alphah = kappa/Cp
}

} // namespace brae
