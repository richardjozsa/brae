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
// rho = rhoPrev + relaxRho*(rho - rhoPrev) on an arbitrary buffer pair -- the BOUNDARY half of the
// solver's rho field. See the .cu for why it must exist.
void deviceRhoRelaxBuffer(
    DeviceBuffer<scalar>& rho,
    DeviceBuffer<scalar>& rhoPrev,
    const ThermoCoeffs& c);

// OF has TWO calculate() functions with DIFFERENT SIGNATURES, not one function with a switch
// (hePsiThermo.C / heRhoThermo.C):
//
//   hePsiThermo::calculate(p, T, he, psi,      mu, alpha, doOldTimes)   <- NO rho parameter
//   heRhoThermo::calculate(p, T, he, psi, rho, mu, alpha, doOldTimes)   <- writes rhoCells too
//
// so brae mirrors them as two functions. An earlier version had a single `deviceThermoUpdate(..., bool
// updateRho)`; that boolean was invented here rather than taken from OF, and picking its value was a
// judgement call that got made wrong -- forcing it false for every thermo type took the transonic
// squareBend from converged-in-136 to NaN. Naming the two OF functions removes the judgement.
void deviceHePsiThermoCalculate(
    DeviceThermo& th,
    const DeviceBuffer<scalar>& p,
    const ThermoCoeffs& c);

void deviceHeRhoThermoCalculate(
    DeviceThermo& th,
    const DeviceBuffer<scalar>& p,
    const ThermoCoeffs& c);

// OF basicThermo::correct() -> calculate(). Dispatches on the thermo type exactly as OF's virtual call
// does, so a call site reads like the solver line it mirrors (EEqn.H's thermo.correct()).
void deviceThermoCorrect(
    DeviceThermo& th,
    const DeviceBuffer<scalar>& p,
    const ThermoCoeffs& c);

// OF thermo.rho(). psiThermo::rho() returns p_*psi_ (recomputed from the CURRENT p); rhoThermo::rho()
// returns the STORED rho_ that calculate() last wrote. rhoSimpleFoam's pEqn.H ends with
// `rho = thermo.rho()`, so which of the two it gets is the whole heRhoThermo-lag question.
void deviceThermoRho(
    DeviceThermo& th,
    const DeviceBuffer<scalar>& p,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& rhoOut);

// Laminar DYNAMIC viscosity AT BOUNDARY FACES, mu_b = Sutherland(T_b) -- OF transport_.mu(patchi).
//
// The compressible momentum boundary needs mu_b for muEff_b = mu_b + rho_b*nut_b, and every OF nut wall
// function needs nu_b = mu_b/rho_b (turbulenceModel::nu(patchi)). A constant scalar viscosity is only
// correct for constant-property incompressible flow.
void deviceThermoMuBoundary(
    const DeviceBoundary& dbHe,
    const DeviceBuffer<scalar>& he,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& muBnd);

// Laminar KINEMATIC viscosity AT BOUNDARY FACES, nu_b = mu_b/rho_b -- OF turbulenceModel::nu(patchi).
//
// This is what every OF wall function reads (nutk/nutUSpalding/nutUBlended/omega/epsilon/kLowRe). Passing a
// single scalar instead is correct only for constant-property incompressible flow; under Sutherland with a
// varying rho it is wrong at every wall face, and wrong in a way that still converges.
void deviceThermoNuBoundary(
    const DeviceBoundary& dbP,
    const DeviceBuffer<scalar>& p,
    const DeviceBoundary& dbHe,
    const DeviceBuffer<scalar>& he,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& nuBnd);

// alphaEff at boundary FACES = mu_b/Pr + rho_b*nut_b/Prt_b -- OF's alphaEff(patchi), the diffusivity its
// laplacian actually uses on a patch. brae previously fell back to the adjacent CELL value there, which
// silently drops the whole alphatWallFunction effect at a fixed-temperature wall.
void deviceAlphaEffBoundary(
    const DeviceBuffer<scalar>& muBnd,
    const DeviceBuffer<scalar>& rhoBnd,
    const DeviceBuffer<scalar>* nutBnd,
    const DeviceBuffer<scalar>* prtBnd,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& alphaEffBnd);

// OF pressureControl::limit(p): clamp p into [pMin, pMax] after the pressure solve. See the .cu for why
// this is load-bearing rather than defensive -- OF itself reports negative p on a stock tutorial.
void deviceLimitPressure(
    DeviceBuffer<scalar>& p,
    scalar pMin,
    scalar pMax);

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

// alphat = mut/Prt = rho*nut/Prt, the turbulent thermal diffusivity [kg/(m s)].
//
// Written into th.alphat, which deviceAlphaEff already sums with alpha -- so the energy equation picks
// this up with no change to it at all. Call AFTER the turbulence model has corrected nut; laminar cases
// leave nut = 0 and this is a no-op that keeps alphat at zero.
void deviceAlphat(
    DeviceThermo& th,
    const DeviceBuffer<scalar>& nut,
    const ThermoCoeffs& c);

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
