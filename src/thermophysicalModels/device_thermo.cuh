#pragma once
// device_thermo.cuh -- the one place the equation of state is evaluated.
//
// Everything downstream (momentum, pressure, energy, turbulence) consumes the fields this produces and
// never calls the EOS itself. That is deliberate: when janaf or a real-gas EOS lands, only this file and
// the three model headers change, and no consumer has to be audited.
//
// Call order in the SIMPLE loop mirrors OpenFOAM: solve he, then thermo.correct(), then use rho/psi in
// the pressure equation. deviceThermoCorrect IS thermo.correct().

#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "thermo_types.cuh"
#include "nsrds_functions.cuh"   // EnergyForm + the liquid inversion
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
    const DeviceBuffer<scalar>& TBnd,
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
    const DeviceBuffer<scalar>& TBnd,
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
    const DeviceBuffer<scalar>& TBnd,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& alphaEffBnd);

// OF pressureControl::limit(p): clamp p into [pMin, pMax] after the pressure solve. See the .cu for why
// this is load-bearing rather than defensive -- OF itself reports negative p on a stock tutorial.
void deviceLimitPressure(
    DeviceBuffer<scalar>& p,
    scalar pMin,
    scalar pMax);

// T -> he, for initialising the enthalpy field from the T that the case actually ships.
// LIQUID (ThermoModel::liquidH2O): evaluate the NSRDS correlations at every cell, filling Cp/mu/kappa
// and the THERMO density rhoThermo (not the solver's rho), plus alpha = kappa/Cp. No-op on the gas path.
// h -> T over a whole field, each cell from its OWN initial guess (so thermo.correct() can later pass
// the previous cell temperature). `ok` is 1 where the enthalpy residual met `tol` within `maxIter`, 0
// otherwise -- reported per cell rather than aborting, because the caller is better placed than the
// kernel to decide whether one bad inversion is fatal. Standalone: nothing in the solver calls this yet.
// The generic inversion: solves F(p,T) = target for T, where F is h (sensibleEnthalpy) or
// h - p/rho (sensibleInternalEnergy). `p` may be null for the enthalpy form, which ignores it.
void deviceH2OEnergyToT(
    EnergyForm form,
    const DeviceBuffer<scalar>& target,
    const DeviceBuffer<scalar>* p,
    const DeviceBuffer<scalar>& Tguess,
    DeviceBuffer<scalar>& T,
    DeviceBuffer<label>& ok,
    DeviceBuffer<scalar>& residual,
    scalar tol = 1e-12,
    int maxIter = 50);

void deviceH2OHToT(
    const DeviceBuffer<scalar>& hTarget,
    const DeviceBuffer<scalar>& Tguess,
    DeviceBuffer<scalar>& T,
    DeviceBuffer<label>& ok,
    DeviceBuffer<scalar>& residual,
    scalar tol = 1e-12,
    int maxIter = 50);

void deviceThermoLiquidProperties(
    DeviceThermo& th,
    const ThermoCoeffs& c);

// The same correlations at boundary faces, from a boundary temperature field. Takes T_b directly so it
// stays independent of the he->T inversion and is testable on a temperature field alone.
void deviceThermoLiquidBoundary(
    const DeviceBuffer<scalar>& Tb,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& CpB,
    DeviceBuffer<scalar>& muB,
    DeviceBuffer<scalar>& kappaB,
    DeviceBuffer<scalar>& rhoB);

// `p` is required for the liquid sensibleInternalEnergy form (e = h(T) - p/rho(T)) and ignored by the
// gas form, which is a pure function of T. Defaulted to null so gas call sites are unchanged.
void deviceThermoHeFromT(
    DeviceThermo& th,
    const ThermoCoeffs& c,
    const DeviceBuffer<scalar>* p = nullptr);

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
    const DeviceBuffer<scalar>& TBnd,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& rhoBnd);

// T AT BOUNDARY FACES, from he_b -- and the argument every one of the four functions above now takes.
//
// OF's heRhoThermo::calculate walks each patch ONCE: it establishes T_b (from he_b, or directly where the
// T patch fixes a value) and then evaluates rho/mu/alphah from (p_b, T_b). brae had instead inverted he_b
// separately inside each property kernel, which was merely redundant on the gas path -- the inversion is
// closed-form and they all agreed -- but on the liquid path put a perfect-gas formula in four places at
// once. Deriving T_b here, once, is both OF's structure and the thing that makes those four kernels
// impossible to get individually wrong.
//
// `Tcell` seeds the liquid Newton from each face's owner cell and may be null (it then starts at 300 K);
// it changes the iteration count only, never the converged answer, which is accepted on the energy
// residual. Ignored entirely by the gas path, whose inversion is exact.
// `TFixMask`/`TFix` carry OF's fixesValue() branch: on a face whose T patch PRESCRIBES a temperature,
// T_b is that value exactly and he_b is re-derived from it at the current p_b (dbHe.refValue is
// rewritten in place). Everywhere else he_b is authoritative and T_b is inverted from it. Pass null for
// both to invert everywhere, which is what the gas path does anyway -- there he is p-independent, so a
// once-converted he_b never goes stale and the round trip is exact.
void deviceThermoTBoundary(
    const DeviceBoundary& dbP,
    const DeviceBuffer<scalar>& p,
    DeviceBoundary& dbHe,
    const DeviceBuffer<scalar>& he,
    const ThermoCoeffs& c,
    const DeviceBuffer<scalar>* Tcell,
    const DeviceBuffer<label>*  TFixMask,
    const DeviceBuffer<scalar>* TFix,
    DeviceBuffer<scalar>& TBnd);

} // namespace brae
