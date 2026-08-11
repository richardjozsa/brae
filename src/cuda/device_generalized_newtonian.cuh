#pragma once
// device_generalized_newtonian.cuh -- OF laminarModels::generalizedNewtonian with the powerLaw viscosity.
//
// WHAT THIS REPLACES, AND WHY IT IS NOT AN ADD-ON. For a turbulent model nuEff = nu + nut, so the model
// ADDS to the molecular viscosity. generalizedNewtonian does not: OF's nut() returns zero and nuEff()
// returns nu_ outright (generalizedNewtonian.C), i.e. the strain-rate-dependent viscosity SUPPLANTS the
// molecular one. Adding instead of replacing would be wrong by whatever the thermo viscosity happens to
// be -- small here, but wrong in the same silent way as running without the model at all.
//
// OF, verbatim:
//     strainRate = sqrt(2)*mag(symm(grad(U)))                       generalizedNewtonian.C:98
//     nu = max(nuMin, min(nuMax, nu0*pow(max(strainRate, SMALL), n - 1)))   powerLaw.C
// with nu0 = the thermo's kinematic viscosity, mu/rho, and the compressible momentum equation then using
// muEff = rho*nuEff = rho*nu.
//
// NOTE `k` IS NOT A COEFFICIENT HERE. squareBendLiqNoNewtonian's dictionary carries `k 0.02`, but
// v2412's powerLaw::read only ever looks up n, nuMin and nuMax -- OpenFOAM ignores k on this model too.
// Reproducing that (rather than inventing a use for it) is the point; brae's dict audit still names it.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"

namespace brae {

// mu = rho * max(nuMin, min(nuMax, (mu/rho)*pow(max(sqrt(S2), SMALL), n-1))), per cell.
//
// S2 is 2*magSqr(symm(gradU)) -- the SAME invariant kOmegaSST already builds (deviceS2), and
// sqrt(S2) is exactly OF's sqrt(2)*mag(symm(grad(U))), so there is one strain definition in the code.
// `mu` is both input (the thermo viscosity, giving nu0 = mu/rho) and output (the effective dynamic
// viscosity the momentum assembly consumes), matching how nuConst_ is used at the call site.
void deviceGeneralizedNewtonianPowerLawMu(
    const DeviceBuffer<scalar>& S2,
    const DeviceBuffer<scalar>& rho,
    scalar nuMin,
    scalar nuMax,
    scalar n,
    DeviceBuffer<scalar>& mu);

}   // namespace brae
