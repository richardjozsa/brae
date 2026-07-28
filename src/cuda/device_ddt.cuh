#pragma once
// Transient time-derivative fvm::ddt(rho, psi) for the PIMPLE / rho-transient solvers -- the ONE numerical module SIMPLE
// (steady) does not have. Adds the IMPLICIT ddt contribution to an already-assembled fvMatrix's diagonal + source, so a
// transient UEqn is  ddt + div(phi,U) + divDevReff == fvOptions  exactly as OpenFOAM assembles it (ddt added BEFORE
// UEqn.relax()). Reproduces OpenFOAM-2412 byte-for-byte on a fixed (non-moving) mesh, where Vsc == Vsc0 == V:
//
//   diag[c]   += coefft *rDeltaT*rho*V[c]
//   source[c] += rDeltaT*rho*V[c]*( coefft0*psi_old[c] - coefft00*psi_old2[c] )
//
// Euler (1st order)   : coefft=1,  coefft0=1,               coefft00=0            -> (rho/dt)(V) , source (rho/dt)(V) psi_old
// backward (2nd order): coefft=1+dt/(dt+dt0), coefft00=dt^2/(dt0(dt+dt0)), coefft0=coefft+coefft00
//
// OF source refs (/home/ghost/cudafoam/src/finiteVolume/finiteVolume/ddtSchemes):
//   EulerDdtScheme.C:381-391   fvmDdt(vf)        diag = rDeltaT*Vsc ; source = rDeltaT*vf.oldTime()*Vsc   (non-moving)
//   EulerDdtScheme.C:416-428   fvmDdt(rho,vf)    diag = rDeltaT*rho*Vsc ; source = rDeltaT*rho*vf.oldTime()*Vsc
//   backwardDdtScheme.C:26     diag = (coefft*rDeltaT)*V
//   backwardDdtScheme.C:39-43  source = rDeltaT*V*(coefft0*oldTime - coefft00*oldTime.oldTime())   (non-moving)
//   backwardDdtScheme.C:96-98  coefft / coefft00 / coefft0
#include "cf_types.cuh"
#include "device_buffer.cuh"

namespace brae {

// fvSchemes ddtSchemes.default. steadyState == SIMPLE (no ddt); Euler/backward/CrankNicolson == transient.
enum class DdtScheme { steadyState = 0, Euler = 1, backward = 2, CrankNicolson = 3 };

// The time-scheme weights, computed ONCE per time step (host side) exactly as OF does. `active == false` (steadyState,
// or before the first advance) makes deviceFvmDdt a no-op, so the same code path serves SIMPLE unchanged.
// CrankNicolson replaces the psi.old2 source term (-coefft00*psi_old2) with an auxiliary stored-old-ddt term
// (+ocCoeff*ddt0), where ddt0 is carried per transported variable and updated each step (OF CrankNicolsonDdtScheme).
struct DdtCoeffs
{
    scalar rDeltaT  = 0;      // 1/deltaT
    scalar coefft   = 1;      // implicit diagonal weight (backward: 1+dt/(dt+dt0); CrankNicolson: 1+ocCoeff)
    scalar coefft0  = 1;      // psi.oldTime()          source weight (CrankNicolson: 1+ocCoeff)
    scalar coefft00 = 0;      // psi.oldTime().oldTime() source weight (backward only)
    scalar ocCoeff  = 0;      // CrankNicolson off-centring: ddt0 source weight + ddt0 recurrence blend (0=Euler, 1=pure CN)
    scalar rDeltaT0 = 0;      // 1/deltaT0, for the ddt0 recurrence (CrankNicolson)
    scalar coefft0dd = 0;     // CrankNicolson ddt0-recurrence coef (OF coef0_): 1 on the FIRST ddt0 update (Euler ddt0), 1+oc after
    bool   cn       = false;  // CrankNicolson: source uses +ocCoeff*ddt0 instead of -coefft00*psi.old2
    bool   active   = false;
};

// Build the coefficients exactly as OF (backwardDdtScheme.C:96-98; CrankNicolsonDdtScheme rDtCoef/offCentre). First
// transient step (deltaT0<=0) falls back to Euler, matching OF (oldTime().oldTime()/ddt0 are uninitialised until two
// levels exist -> pimpleFoam bootstraps with Euler). ocCoeff is the CrankNicolson off-centring (fvSchemes "CrankNicolson <oc>").
// cnWarm=false on the FIRST ddt0 update (second overall step) -> Euler ddt0 (OF coef0_=1); true afterwards (OF coef0_=1+oc).
// This Euler-startup of the ddt0 recurrence is what makes CrankNicolson genuinely 2nd order (else the first ddt0 is 2x too
// large and, at oc=1, oscillates without decaying -> 1st order).
inline DdtCoeffs ddtCoeffs(DdtScheme scheme, scalar deltaT, scalar deltaT0, scalar ocCoeff = scalar(1), bool cnWarm = false)
{
    DdtCoeffs c;
    if (scheme == DdtScheme::steadyState || deltaT <= scalar(0))
        return c;                                   // active stays false -> no-op
    c.active  = true;
    c.rDeltaT = scalar(1) / deltaT;
    if (scheme == DdtScheme::backward && deltaT0 > scalar(0))
    {
        c.coefft   = scalar(1) + deltaT / (deltaT + deltaT0);
        c.coefft00 = deltaT * deltaT / (deltaT0 * (deltaT + deltaT0));
        c.coefft0  = c.coefft + c.coefft00;
    }
    else if (scheme == DdtScheme::CrankNicolson && deltaT0 > scalar(0))
    {
        c.cn        = true;
        c.ocCoeff   = ocCoeff;
        c.rDeltaT0  = scalar(1) / deltaT0;
        c.coefft    = scalar(1) + ocCoeff;          // diag = (1+oc)*rDeltaT*V   (OF coef_)
        c.coefft0   = scalar(1) + ocCoeff;          // source psi.old weight = (1+oc)*rDeltaT
        c.coefft0dd = cnWarm ? (scalar(1) + ocCoeff) : scalar(1);   // ddt0 recurrence coef (OF coef0_): Euler on first update
    }
    // else Euler: coefft=1, coefft0=1, coefft00=0, cn=false (struct defaults). CrankNicolson first step lands here.
    return c;
}

// Add the implicit ddt of ONE scalar component (a velocity component, or a transported scalar) to (diag, source).
// diag/source are the assembled fvMatrix arrays (size == nCells); this ACCUMULATES, matching ddt being one UEqn term.
// psiOld2 may be empty (Euler / bootstrap). rho is the constant density (1 for incompressible pimpleFoam). No-op when
// c.active is false. V is the cell-volume field (mesh V; brae is always a fixed mesh so Vsc == V). Use this when diag +
// source belong to the SAME matrix (a scalar transport, e.g. k/epsilon in PIMPLE).
void deviceFvmDdt(
    const DeviceBuffer<scalar>& V,
    const DdtCoeffs&            c,
    scalar                      rho,
    const DeviceBuffer<scalar>& psiOld,
    const DeviceBuffer<scalar>& psiOld2,
    DeviceBuffer<scalar>&       diag,
    DeviceBuffer<scalar>&       source,
    const DeviceBuffer<scalar>* ddt0 = nullptr);   // CrankNicolson stored old ddt (else nullptr)

// Per-scalar transient-ddt bundle for a transported turbulence scalar (k / epsilon / omega / nuTilda). Threaded into
// deviceSolveScalarTransport so a transient (PIMPLE) turbulence solve folds fvm::ddt(f) into its matrix, while the steady
// (SIMPLE) path passes the default (c.active==false, old==nullptr) -> an exact no-op. `old`/`old2` are that scalar's
// oldTime()[.oldTime()] device fields (owned by the solver, rotated each time step); old2 may be null (Euler/bootstrap).
struct ScalarDdt
{
    DdtCoeffs                   c{};
    const DeviceBuffer<scalar>* old  = nullptr;
    const DeviceBuffer<scalar>* old2 = nullptr;
    const DeviceBuffer<scalar>* ddt0 = nullptr;   // CrankNicolson stored old ddt (that scalar's ddt0_ member); null otherwise
};

// The two halves of deviceFvmDdt, split for the vector momentum predictor where the ddt DIAGONAL is shared across the 3
// components (added once to the assembled diag, BEFORE relax) but the ddt SOURCE is per component (uses that component's
// oldTime). deviceFvmDdt(...) == Diag(...) then Source(...). Both no-op when !c.active.
//   diag[c]   += coefft*rDeltaT*rho*V[c]
void deviceFvmDdtDiag(
    const DeviceBuffer<scalar>& V,
    const DdtCoeffs&            c,
    scalar                      rho,
    DeviceBuffer<scalar>&       diag);
//   backward/Euler : source[c] += rDeltaT*rho*V[c]*( coefft0*psiOld[c] - coefft00*psiOld2[c] )
//   CrankNicolson  : source[c] += rho*V[c]*( coefft0*rDeltaT*psiOld[c] + ocCoeff*ddt0[c] )   (ddt0 non-null when c.cn)
void deviceFvmDdtSource(
    const DeviceBuffer<scalar>& V,
    const DdtCoeffs&            c,
    scalar                      rho,
    const DeviceBuffer<scalar>& psiOld,
    const DeviceBuffer<scalar>& psiOld2,
    DeviceBuffer<scalar>&       source,
    const DeviceBuffer<scalar>* ddt0 = nullptr);   // CrankNicolson stored old ddt (else nullptr)

// CrankNicolson ddt0 recurrence (OF DDt0Field update), run ONCE per time step after the old-time rotation:
//   ddt0[c] = coefft*rDeltaT0*( psiOld[c] - psiOld2[c] ) - ocCoeff*ddt0[c]     (in place; uses the OLD ddt0)
// No-op unless c.cn (steady/Euler/backward carry no ddt0). psiOld2 must hold two old levels (skipped on the first step).
void deviceFvmDdtUpdateDdt0(
    const DdtCoeffs&            c,
    const DeviceBuffer<scalar>& psiOld,
    const DeviceBuffer<scalar>& psiOld2,
    DeviceBuffer<scalar>&       ddt0);

}  // namespace brae
