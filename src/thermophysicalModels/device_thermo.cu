// device_thermo.cu -- kernels behind deviceThermoCorrect / deviceThermoRho.

#include "device_thermo.cuh"
#include "nsrds_functions.cuh"   // liquid property correlations (properties liquid)
#include "equation_of_state.cuh"
#include "thermo_model.cuh"
#include "transport_model.cuh"
#include "device_buffer.cuh"
#include "device_blas.cuh"   // deviceCopy/deviceHadamard, for thermo.rho()
#include "device_boundary.cuh"

namespace brae {

namespace {

constexpr int TPB = 256;

inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

}   // namespace

// One pass over the cells: invert he to get T, then evaluate every T-dependent property from it.
// Fused rather than split per property because each one is a handful of flops on a value already in a
// register -- splitting would re-read T from global memory four times for no benefit.
__global__
void thermoUpdateK(
    int n,
    ThermoCoeffs c,
    const scalar* __restrict__ p,
    const scalar* __restrict__ he,
    scalar* __restrict__ T,
    scalar* __restrict__ rho,
    scalar* __restrict__ psi,
    scalar* __restrict__ mu,
    scalar* __restrict__ alpha)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    const scalar Ti = hConstHeToT(he[i], c);
    T[i] = Ti;

    // Bound p before it reaches the EOS. An unbounded iterate can go negative mid-solve, and
    // rho = p/(R T) would then hand a negative density to the next momentum predictor.
    const scalar pi = fmax(p[i], c.pMin);

    const scalar psii = perfectGasPsi(Ti, c);
    psi[i] = psii;
    // null rho -> this call is OF's thermo.correct(), which leaves the solver's rho field alone.
    if (rho) rho[i] = fmin(fmax(psii * pi, c.rhoMin), c.rhoMax);

    const scalar mui = transportMu(Ti, c);
    mu[i] = mui;
    alpha[i] = transportAlpha(mui, c);
}

// he from T, used once at startup: cases ship a T field, the energy equation wants he.
__global__
void heFromTK(
    int n,
    ThermoCoeffs c,
    const scalar* __restrict__ T,
    scalar* __restrict__ he)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    he[i] = hConstTToHe(T[i], c);
}

// hePsiThermo::calculate -- T, psi, mu, alpha. There is no rho parameter in OF's signature, so the
// kernel is handed a null rho and leaves the solver's density alone.
void deviceHePsiThermoCalculate(
    DeviceThermo& th,
    const DeviceBuffer<scalar>& p,
    const ThermoCoeffs& c)
{
    if (th.n == 0) return;
    thermoUpdateK<<<nBlocks(th.n), TPB>>>(
        th.n, c, p.data(), th.he.data(), th.T.data(),
        nullptr,                       // hePsiThermo::calculate has no rho argument
        th.psi.data(), th.mu.data(), th.alpha.data());
    cudaCheck(cudaGetLastError(), "hePsiThermoCalculate");
}

// heRhoThermo::calculate -- T, psi, RHO, mu, alpha (heRhoThermo.C: rhoCells[celli] = mixture_.rho(p,T)).
void deviceHeRhoThermoCalculate(
    DeviceThermo& th,
    const DeviceBuffer<scalar>& p,
    const ThermoCoeffs& c)
{
    if (th.n == 0) return;
    thermoUpdateK<<<nBlocks(th.n), TPB>>>(
        th.n, c, p.data(), th.he.data(), th.T.data(),
        th.rhoThermo.data(),           // heRhoThermo::calculate writes the THERMO's rho_, not the solver's
        th.psi.data(), th.mu.data(), th.alpha.data());
    cudaCheck(cudaGetLastError(), "heRhoThermoCalculate");
}

// basicThermo::correct() -> the mixture's calculate(). The virtual dispatch OF does at run time.
void deviceThermoCorrect(
    DeviceThermo& th,
    const DeviceBuffer<scalar>& p,
    const ThermoCoeffs& c)
{
    if (c.rhoThermoType) deviceHeRhoThermoCalculate(th, p, c);
    else                 deviceHePsiThermoCalculate(th, p, c);
}

// thermo.rho(): psiThermo returns p_*psi_, rhoThermo returns the stored rho_.
void deviceThermoRho(
    DeviceThermo& th,
    const DeviceBuffer<scalar>& p,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& rhoOut)
{
    if (th.n == 0) return;
    if (c.rhoThermoType) deviceCopy(rhoOut, th.rhoThermo);    // rhoThermo::rho() -> the stored rho_
    else                 deviceHadamard(rhoOut, p, th.psi);   // psiThermo::rho() -> p_*psi_
}

// rho = rhoPrev + a*(rho - rhoPrev), then rhoPrev = rho. Bounds are re-applied because relaxing toward a
// previous value cannot violate them, but relaxing AWAY from a clamped value can.
__global__
void rhoRelaxK(
    int n,
    scalar a,
    scalar rhoMin,
    scalar rhoMax,
    scalar* __restrict__ rho,
    scalar* __restrict__ rhoPrev)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const scalar blended = rhoPrev[i] + a * (rho[i] - rhoPrev[i]);
    const scalar bounded = fmin(fmax(blended, rhoMin), rhoMax);
    rho[i] = bounded;
    rhoPrev[i] = bounded;
}

__global__
void rhoSeedPrevK(
    int n,
    const scalar* __restrict__ rho,
    scalar* __restrict__ rhoPrev)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    rhoPrev[i] = rho[i];
}

// The same relaxation applied to an arbitrary rho buffer, for the BOUNDARY half of the solver's rho
// field. OF has one `rho` volScalarField and `rho.relax()` relaxes its internal AND boundary values
// together; brae stores the two separately, so the boundary one needs the identical treatment or the
// pressure equation's phiHbyA weighting sees an unrelaxed density. Same kernel, deliberately: two
// copies of this arithmetic is two places for them to diverge.
void deviceRhoRelaxBuffer(
    DeviceBuffer<scalar>& rho,
    DeviceBuffer<scalar>& rhoPrev,
    const ThermoCoeffs& c)
{
    const int n = static_cast<int>(rho.size());
    if (n == 0 || rhoPrev.size() != rho.size()) return;
    rhoRelaxK<<<nBlocks(n), TPB>>>(n, c.relaxRho, c.rhoMin, c.rhoMax, rho.data(), rhoPrev.data());
    cudaCheck(cudaGetLastError(), "rhoRelaxBuffer");
}

void deviceRhoRelax(
    DeviceThermo& th,
    const ThermoCoeffs& c)
{
    if (th.n == 0) return;
    rhoRelaxK<<<nBlocks(th.n), TPB>>>(
        th.n,
        c.relaxRho,
        c.rhoMin,
        c.rhoMax,
        th.rho.data(),
        th.rhoPrev.data());
    cudaCheck(cudaGetLastError(), "rhoRelax");
}

void deviceRhoSeedPrev(DeviceThermo& th)
{
    if (th.n == 0) return;
    rhoSeedPrevK<<<nBlocks(th.n), TPB>>>(
        th.n,
        th.rho.data(),
        th.rhoPrev.data());
    cudaCheck(cudaGetLastError(), "rhoSeedPrev");
}

// alphat = rho*nut/Prt. Kept in its own kernel rather than folded into thermoUpdateK because nut is
// owned by the turbulence model and is only valid after correctTurbulence(), which runs at the END of the
// outer iteration -- whereas thermoUpdateK runs in the middle of it.
__global__
void alphatK(
    int n,
    scalar Prt,
    const scalar* __restrict__ rho,
    const scalar* __restrict__ nut,
    scalar* __restrict__ alphat)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    alphat[i] = rho[i] * nut[i] / Prt;
}

void deviceAlphat(
    DeviceThermo& th,
    const DeviceBuffer<scalar>& nut,
    const ThermoCoeffs& c)
{
    if (th.n == 0) return;
    if (static_cast<int>(nut.size()) != th.n) return;   // laminar: no nut field, alphat stays zero
    alphatK<<<nBlocks(th.n), TPB>>>(
        th.n,
        c.Prt,
        th.rho.data(),
        nut.data(),
        th.alphat.data());
    cudaCheck(cudaGetLastError(), "alphat");
}

// rho_b = p_b/(R*T_b), face by face. Same EOS as the cell update, so a boundary and its cell agree
// whenever their p and T do.
__global__
void rhoBoundaryK(
    int n,
    ThermoCoeffs c,
    const scalar* __restrict__ pBnd,
    const scalar* __restrict__ heBnd,
    scalar* __restrict__ rhoBnd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const scalar Tb = hConstHeToT(heBnd[i], c);
    const scalar pb = fmax(pBnd[i], c.pMin);
    rhoBnd[i] = fmin(fmax(pb / (c.R * Tb), c.rhoMin), c.rhoMax);
}

void deviceThermoRhoBoundary(
    const DeviceBoundary& dbP,
    const DeviceBuffer<scalar>& p,
    const DeviceBoundary& dbHe,
    const DeviceBuffer<scalar>& he,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& rhoBnd)
{
    if (dbP.n == 0) return;
    DeviceBuffer<scalar> pBnd;
    DeviceBuffer<scalar> heBnd;
    deviceBCValue(dbP, p, pBnd);
    deviceBCValue(dbHe, he, heBnd);
    rhoBnd.resize(dbP.n);
    rhoBoundaryK<<<nBlocks(dbP.n), TPB>>>(
        dbP.n,
        c,
        pBnd.data(),
        heBnd.data(),
        rhoBnd.data());
    cudaCheck(cudaGetLastError(), "rhoBoundary");
}

// mu_b = Sutherland(T_b), face by face -- OF's transport_.mu(patchi).
//
// Needed because the compressible momentum boundary wants muEff_b = mu_b + rho_b*nut_b, and the wall
// functions want the KINEMATIC nu_b = mu_b/rho_b. Both are per-face: a single scalar viscosity is only
// right for a constant-property incompressible case.
__global__
void muBoundaryK(
    int n,
    ThermoCoeffs c,
    const scalar* __restrict__ heBnd,
    scalar* __restrict__ muBnd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    muBnd[i] = transportMu(hConstHeToT(heBnd[i], c), c);
}

void deviceThermoMuBoundary(
    const DeviceBoundary& dbHe,
    const DeviceBuffer<scalar>& he,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& muBnd)
{
    if (dbHe.n == 0) return;
    DeviceBuffer<scalar> heBnd;
    deviceBCValue(dbHe, he, heBnd);
    muBnd.resize(dbHe.n);
    muBoundaryK<<<nBlocks(dbHe.n), TPB>>>(
        dbHe.n,
        c,
        heBnd.data(),
        muBnd.data());
    cudaCheck(cudaGetLastError(), "muBoundary");
}

// nu_b = mu_b/rho_b, face by face -- OF turbulenceModel::nu(patchi), which is literally
// transport_.mu(patchi)/rho_.boundaryField()[patchi]. Every OF wall function reads this, so brae's wall
// functions need it too: in compressible flow it is a field, not the single number a constant-property
// incompressible case can get away with.
__global__
void nuBoundaryK(
    int n,
    ThermoCoeffs c,
    const scalar* __restrict__ pBnd,
    const scalar* __restrict__ heBnd,
    scalar* __restrict__ nuBnd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const scalar Tb = hConstHeToT(heBnd[i], c);
    const scalar pb = fmax(pBnd[i], c.pMin);
    const scalar rb = fmin(fmax(pb / (c.R * Tb), c.rhoMin), c.rhoMax);
    nuBnd[i] = transportMu(Tb, c) / rb;
}

void deviceThermoNuBoundary(
    const DeviceBoundary& dbP,
    const DeviceBuffer<scalar>& p,
    const DeviceBoundary& dbHe,
    const DeviceBuffer<scalar>& he,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& nuBnd)
{
    if (dbP.n == 0) return;
    DeviceBuffer<scalar> pBnd;
    DeviceBuffer<scalar> heBnd;
    deviceBCValue(dbP, p, pBnd);
    deviceBCValue(dbHe, he, heBnd);
    nuBnd.resize(dbP.n);
    nuBoundaryK<<<nBlocks(dbP.n), TPB>>>(
        dbP.n,
        c,
        pBnd.data(),
        heBnd.data(),
        nuBnd.data());
    cudaCheck(cudaGetLastError(), "nuBoundary");
}

// alphaEff AT BOUNDARY FACES = alpha_b + alphat_b, which is what OF's laplacian(alphaEff, he) uses on a
// patch (heThermo::alphaEff(patchi); CpByCpv = 1 for sensibleEnthalpy).
//
//   alphat_b = rho_b * nut_b / Prt_b       -- OF alphatWallFunction::updateCoeffs is operator==(rhow*tnutw/Prt_)
//   alpha_b  = mu_b / Pr
//
// Prt_b is PER FACE because the two Prt in a compressible case are different numbers: the wall function's
// own (default 0.85) on its patches, and the turbulence model's (default 1.0) everywhere else. nut_b is
// the wall-function nut on a wall face and the extrapolated cell nut elsewhere, matching what the momentum
// boundary already uses -- so momentum and energy see one consistent wall eddy viscosity.
__global__
void alphaEffBoundaryK(
    int n,
    ThermoCoeffs c,
    const scalar* __restrict__ muBnd,
    const scalar* __restrict__ rhoBnd,
    const scalar* __restrict__ nutBnd,
    const scalar* __restrict__ prtBnd,
    scalar* __restrict__ alphaEffBnd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    // transportAlpha, not mu/Pr: under sutherland OF uses the Eucken kappa/Cp and there is no Pr at all.
    const scalar alphaLam = transportAlpha(muBnd[i], c);
    const scalar alphaTur = (nutBnd && prtBnd) ? rhoBnd[i] * nutBnd[i] / prtBnd[i] : scalar(0);
    // CpByCpv, as on the cell path (heThermo::alphaEff) -- gamma for sensibleInternalEnergy, 1 for h.
    alphaEffBnd[i] = thermoCpByCpv(c) * (alphaLam + alphaTur);
}

void deviceAlphaEffBoundary(
    const DeviceBuffer<scalar>& muBnd,
    const DeviceBuffer<scalar>& rhoBnd,
    const DeviceBuffer<scalar>* nutBnd,
    const DeviceBuffer<scalar>* prtBnd,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& alphaEffBnd)
{
    const int n = static_cast<int>(muBnd.size());
    if (n == 0) return;
    alphaEffBnd.resize(n);
    alphaEffBoundaryK<<<nBlocks(n), TPB>>>(
        n,
        c,
        muBnd.data(),
        rhoBnd.data(),
        (nutBnd && nutBnd->size() == muBnd.size()) ? nutBnd->data() : nullptr,
        (prtBnd && prtBnd->size() == muBnd.size()) ? prtBnd->data() : nullptr,
        alphaEffBnd.data());
    cudaCheck(cudaGetLastError(), "alphaEffBoundary");
}

// OF pressureControl::limit(p): clamp p into [pMin, pMax] after the pressure solve, before rho is
// recomputed from it (rhoSimpleFoam/pEqn.H:90).
//
// Not a cosmetic safety net. On the stock aerofoilNACA0012 tutorial (Mach 0.72) OF's own log reports
//   pressureControl: p max 2332401     (23x ambient)
//   pressureControl: p min -241053.81  (NEGATIVE)
// during start-up. A negative p gives a negative rho through the perfect-gas EOS, and the next momentum
// solve is NaN -- which is exactly what brae did on that case before this existed.
__global__
void clampPressureK(
    int n,
    scalar pMin,
    scalar pMax,
    scalar* __restrict__ p)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    p[i] = fmin(fmax(p[i], pMin), pMax);
}

void deviceLimitPressure(
    DeviceBuffer<scalar>& p,
    scalar pMin,
    scalar pMax)
{
    const int n = static_cast<int>(p.size());
    if (n == 0) return;
    clampPressureK<<<nBlocks(n), TPB>>>(n, pMin, pMax, p.data());
    cudaCheck(cudaGetLastError(), "limitPressure");
}

void deviceThermoHeFromT(
    DeviceThermo& th,
    const ThermoCoeffs& c)
{
    if (th.n == 0) return;
    heFromTK<<<nBlocks(th.n), TPB>>>(
        th.n,
        c,
        th.T.data(),
        th.he.data());
    cudaCheck(cudaGetLastError(), "heFromT");
}

// ---------------------------------------------------------------------------------------------------
// LIQUID PROPERTIES. One evaluation point for the whole NSRDS family, so there is exactly one place
// where T maps to (Cp, mu, kappa, rho) and the internal and boundary paths cannot drift apart.
//
// alpha for a liquid is kappa/Cp, NOT the gas path's mu/Pr: the liquid correlations give the thermal
// conductivity directly, so deriving it from a Prandtl number would discard kappa(T) and substitute a
// constant-Pr assumption that is not what the case asked for. (OF: thermophysicalProperties' alpha is
// kappa/Cp for these mixtures; the Pr route belongs to the const/sutherland transport models.)
__global__
void liquidPropsK(
    int n,
    const scalar* __restrict__ T,
    scalar* __restrict__ Cp,
    scalar* __restrict__ mu,
    scalar* __restrict__ kappa,
    scalar* __restrict__ rho,
    scalar* __restrict__ alpha)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const scalar t  = T[i];
    const scalar cp = H2OLiquid::Cp(t);
    const scalar k  = H2OLiquid::kappa(t);
    Cp[i]    = cp;
    mu[i]    = H2OLiquid::mu(t);
    kappa[i] = k;
    rho[i]   = H2OLiquid::rho(t);
    if (alpha) alpha[i] = k/cp;      // OF alpha = kappa/Cp  [kg/(m s)]
}

void deviceThermoLiquidProperties(
    DeviceThermo& th,
    const ThermoCoeffs& c)
{
    if (th.n == 0) return;
    if (c.model != ThermoModel::liquidH2O)
        return;   // the gas path owns its own scalars; silently doing nothing here would hide a miswire
    if (th.CpField.size() != static_cast<std::size_t>(th.n)) th.allocateLiquid();
    liquidPropsK<<<nBlocks(th.n), TPB>>>(
        th.n,
        th.T.data(),
        th.CpField.data(),
        th.mu.data(),
        th.kappa.data(),
        th.rhoThermo.data(),          // the THERMO density; the solver's rho is assigned separately
        th.alpha.size() == static_cast<std::size_t>(th.n) ? th.alpha.data() : nullptr);
    cudaCheck(cudaGetLastError(), "liquidProperties");
}

// The same correlations at boundary FACES, from a boundary temperature field. Takes T_b directly rather
// than deriving it, so this stays independent of the he->T inversion (which is a later step) and can be
// tested on a temperature field alone.
void deviceThermoLiquidBoundary(
    const DeviceBuffer<scalar>& Tb,
    const ThermoCoeffs& c,
    DeviceBuffer<scalar>& CpB,
    DeviceBuffer<scalar>& muB,
    DeviceBuffer<scalar>& kappaB,
    DeviceBuffer<scalar>& rhoB)
{
    const int n = static_cast<int>(Tb.size());
    if (n == 0 || c.model != ThermoModel::liquidH2O) return;
    CpB.resize(n);
    muB.resize(n);
    kappaB.resize(n);
    rhoB.resize(n);
    liquidPropsK<<<nBlocks(n), TPB>>>(
        n,
        Tb.data(),
        CpB.data(),
        muB.data(),
        kappaB.data(),
        rhoB.data(),
        nullptr);                     // no boundary alpha consumer yet; added when the EEqn needs it
    cudaCheck(cudaGetLastError(), "liquidBoundary");
}

} // namespace brae
