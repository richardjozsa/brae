#pragma once
// device_energy.cuh -- the sensible-enthalpy equation.
//
// This is a thin wrapper over deviceSolveScalarTransport, not a new solver. That scaffold already does
// div(phi,f) + laplacian(D,f) with the deferred limited/linearUpwind/non-orth corrections, relaxation,
// cyclic/AMI coupling and the linear solve; its own header says it is meant to be shared by
// "energy/species/compressible transport later". All this file adds is the diffusivity the energy
// equation needs -- alphaEff = alpha + alphat -- and the choice of what goes in the source term.
//
// Phase 1 scope is the frozen-field steady form:
//
//     div(phi, he) - laplacian(alphaEff, he) = 0
//
// The kinetic-energy and pressure-work terms OpenFOAM's rhoSimpleFoam carries in its EEqn are deliberately
// absent: with the velocity field held fixed there is nothing to exchange energy with, and adding them
// before the momentum coupling exists would make Gate 1 untestable against scalarTransportFoam.

#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_ami.cuh"
#include "device_cyclic.cuh"
#include "thermo_types.cuh"

namespace brae {

// alphaEff = alpha + alphat, the diffusivity of the he equation. alphat is zero until the compressible
// turbulence models land, so this is a laminar solve today and needs no change when they do.
void deviceAlphaEff(
    const DeviceThermo& th,
    DeviceBuffer<scalar>& alphaEff);

// he boundary values from the T boundary. OpenFOAM's fixedEnergy/gradientEnergy/mixedEnergy are not
// independent conditions -- they are the temperature conditions expressed in enthalpy, which is why a
// case specifies 0/T and never 0/he. So brae does the same: the existing BC machinery builds the
// boundary from 0/T (fixedValue, zeroGradient, inletOutlet, cyclic, ... all work unchanged), and this
// converts only the values.
//
// With hConst the conversion is linear, he = Cp*T + Hf, so a zero-gradient T patch is exactly a
// zero-gradient he patch and no bcType has to change. janaf would break that linearity and force a
// per-face inversion here -- another reason the thermo lives behind hConstTToHe rather than inline.
//
// dbHe must already have the structure of dbT (same face count and bcType); only refValue is written.
void deviceEnergyBoundaryFromT(
    const DeviceBoundary& dbT,
    const ThermoCoeffs& c,
    DeviceBoundary& dbHe);

// Solves he in place. Steady, frozen velocity: phi is whatever flux the caller supplies and is not
// recomputed here.
void deviceSolveEnergy(
    const DeviceMesh& dm,
    const DeviceBoundary& dbHe,
    DeviceThermo& th,
    const DeviceBuffer<scalar>& phiInt,
    const DeviceBuffer<scalar>& phiBnd,
    const DeviceBuffer<scalar>& divU,
    bool limited,
    bool linearUpwind,
    bool nonOrth,
    scalar twoByk,
    scalar relax,
    scalar tol,
    scalar relTol,
    int checkEvery,
    bool useGS,
    DeviceAMI* ami = nullptr,
    DeviceCyclic* cyc = nullptr);

} // namespace brae
