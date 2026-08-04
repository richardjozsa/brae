#pragma once
// thermo_types.cuh -- the thermophysical state carried alongside the flow fields, and the constants
// read once from constant/thermophysicalProperties.
//
// Split in two on purpose. ThermoCoeffs is a POD of dictionary constants, small enough to pass into a
// kernel by value, so the equation of state can be evaluated on the device without chasing pointers.
// DeviceThermo owns the per-cell fields the rest of the solver reads: the momentum predictor needs rho
// and mu, the pressure equation needs rho and psi, the energy equation needs alpha and he.
//
// Scope today is perfectGas + hConst + (const | Sutherland), which is why Cp is a constant here rather
// than a field. janaf makes Cp a function of T and would move it into DeviceThermo -- every consumer
// already goes through deviceThermoUpdate, so that change stays local to this subsystem.

#include "cf_types.cuh"
#include "device_buffer.cuh"

namespace brae {

// Constants from constant/thermophysicalProperties. Passed to kernels by value.
struct ThermoCoeffs
{
    scalar R = 287.058;      // specific gas constant, R_universal/molWeight  [J/(kg K)]
    scalar Cp = 1005.0;      // heat capacity at constant pressure (hConst)   [J/(kg K)]
    scalar Hf = 0.0;         // heat of formation; sensibleEnthalpy sets he = Cp*T + Hf
    scalar Pr = 0.7;         // laminar Prandtl number, sets alpha = mu/Pr

    // transport: sutherland uses (As, Ts), const uses mu0 and ignores both
    scalar As = 1.4792e-06;  // Sutherland coefficient  [kg/(m s sqrt(K))]
    scalar Ts = 116.0;       // Sutherland temperature  [K]
    scalar mu0 = 1.8e-05;    // constant dynamic viscosity, used when sutherland is off  [Pa s]

    bool sutherland = true;  // false -> mu = mu0 everywhere

    // Bounds applied after each update. OF calls these rhoMin/rhoMax/pMin in fvSolution; they exist
    // because a diverging pressure iterate can drive rho negative long before the residuals show it.
    scalar rhoMin = 1e-03;
    scalar rhoMax = 1e+03;
    scalar pMin = 1e+03;

    // rho.relax() factor, fvSolution relaxationFactors.fields.rho. 1.0 = no relaxation.
    scalar relaxRho = 1.0;

    // Turbulent Prandtl number, for alphat = rho*nut/Prt. OF's default is 1.0, NOT the 0.85 that is
    // conventional in the literature: EddyDiffusivity.C sets Prt_("Prt", dimless, 1.0, coeffDict()) and
    // only overrides it if the RAS/LES coeffs dict names it. Using 0.85 against an OF case that says
    // nothing gives a ~15% wall heat flux difference that still converges. Laminar cases never touch it
    // (nut = 0 -> alphat = 0). NOTE: the Prt inside a 0/alphat wall-function entry is the WALL
    // function's own, applied at walls only -- it is not this one.
    scalar Prt = 1.0;
};

// Per-cell thermophysical fields, sized nCells. Internal field only -- boundary values live in the
// DeviceBoundary of the field they belong to, exactly as p and U already do.
struct DeviceThermo
{
    int n = 0;                    // cells

    DeviceBuffer<scalar> T;       // temperature                        [K]
    DeviceBuffer<scalar> he;      // sensible enthalpy, the solved var  [J/kg]
    DeviceBuffer<scalar> rho;     // density                            [kg/m3]
    DeviceBuffer<scalar> psi;     // compressibility d(rho)/d(p)|T      [s2/m2]
    DeviceBuffer<scalar> mu;      // laminar dynamic viscosity          [Pa s]
    DeviceBuffer<scalar> alpha;   // laminar thermal diffusivity mu/Pr  [kg/(m s)]

    // Turbulent thermal diffusivity. Zero until the compressible turbulence models land; the energy
    // equation already reads alphaEff = alpha + alphat so that phase does not touch the EEqn.
    DeviceBuffer<scalar> alphat;

    // Density from the previous outer iteration, for rho.relax(). SIMPLE updates rho from a pressure
    // field that is itself only partly converged, so feeding the raw update straight into the next
    // momentum predictor makes the outer loop oscillate. OF relaxes rho for exactly this reason.
    DeviceBuffer<scalar> rhoPrev;

    void allocate(int nCells)
    {
        n = nCells;
        T.resize(n);
        he.resize(n);
        rho.resize(n);
        psi.resize(n);
        mu.resize(n);
        alpha.resize(n);
        alphat.resize(n);
        rhoPrev.resize(n);
    }
};

} // namespace brae
