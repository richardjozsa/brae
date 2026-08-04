#pragma once
// device_thermo.cuh -- the one place the equation of state is evaluated.
//
// Everything downstream (momentum, pressure, energy, turbulence) consumes the fields this produces and
// never calls the EOS itself. That is deliberate: when janaf or a real-gas EOS lands, only this file and
// the three model headers change, and no consumer has to be audited.
//
// Call order in the SIMPLE loop mirrors OpenFOAM: solve he, then thermo.correct(), then use rho/psi in
// the pressure equation. deviceThermoUpdate IS thermo.correct().

#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "thermo_types.cuh"
#include "device_boundary.cuh"

namespace brae {

// he,p -> T,rho,psi,mu,alpha. Bounds rho and p, because a diverging pressure iterate drives rho negative
// long before the residuals show anything wrong, and a negative rho poisons the next momentum solve.
void deviceThermoUpdate(
    DeviceThermo& th,
    const DeviceBuffer<scalar>& p,
    const ThermoCoeffs& c);

// T -> he, for initialising the enthalpy field from the T that the case actually ships.
void deviceThermoHeFromT(
    DeviceThermo& th,
    const ThermoCoeffs& c);

// OF's rho.relax(): rho = rhoPrev + relaxRho*(rho - rhoPrev), then rhoPrev = rho.
//
// Called AFTER deviceThermoUpdate, once per outer iteration. SIMPLE derives rho from a pressure field
// that is itself only partly converged, so handing the raw update to the next momentum predictor makes
// the outer loop oscillate -- badly, once psi is large. Seed rhoPrev with the initial rho (see
// deviceRhoSeedPrev) or the first relaxation blends against zeros.
void deviceRhoRelax(
    DeviceThermo& th,
    const ThermoCoeffs& c);

// rhoPrev = rho. Call once after the first thermo update, before the outer loop starts relaxing.
void deviceRhoSeedPrev(DeviceThermo& th);

// Density AT BOUNDARY FACES, from the boundary values of p and he: rho_b = p_b/(R*T_b).
//
// Not the adjacent cell value. OF's rho is thermo.rho(), whose boundaryField is evaluated from the
// boundary p and T, so at a fixed-temperature inlet rho_b differs from the cell behind it -- extrapolating
// instead would put the wrong mass flux through every such patch and still converge, quietly.
//
// Needed because deviceInterpolate covers internal faces only, so phiHbyA's boundary half cannot reuse it.
void deviceThermoRhoBoundary(
    const DeviceBoundary& dbP,
    const DeviceBuffer<scalar>& p,
    const DeviceBoundary& dbHe,
    const DeviceBuffer<scalar>& he,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& rhoBnd);

} // namespace brae
