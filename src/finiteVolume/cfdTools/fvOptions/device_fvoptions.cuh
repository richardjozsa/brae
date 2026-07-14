#pragma once
// Device DarcyForchheimer porosity (explicitPorositySource), evaluated each iteration from the current U.
// OF (incompressible) porosityModels::DarcyForchheimer::apply, mu=nu_laminar, rho=1:
//   Cd = nu*diag(d) + |U|*diag(0.5 f);  isoCd = tr(Cd)
//   diag[c]            += V*isoCd                       (implicit, isotropic part)
//   relaxSrc[comp][c]  -= V*((Cd - I*isoCd).U)[comp]    (explicit, anisotropic remainder)
// Diagonal d/f only (identity coordinateSystem). Cells are a cellZone (unique) -> no atomics needed.
#include "cf_types.cuh"
#include "device_buffer.cuh"

namespace brae {

struct DevicePorosity {
    bool                active = false;
    DeviceBuffer<label> cells;          // porous cellZone cells
    vector              d{0,0,0}, f{0,0,0};   // adjusted Darcy d + Forchheimer f
};

// diag[c] += V*isoCd for the porous cells.  Call once (mDiag) before rAU.
void deviceFvoPorosityDiag(const DevicePorosity& por, scalar nu, const DeviceBuffer<scalar>& V,
                           const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                           DeviceBuffer<scalar>& diag);
// relaxSrc[c] += V*(isoCd - c_comp)*U_comp for the porous cells (= the explicit -V*((Cd-I*isoCd).U)[comp]).
void deviceFvoPorositySource(const DevicePorosity& por, int comp, scalar nu, const DeviceBuffer<scalar>& V,
                             const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                             DeviceBuffer<scalar>& src);

// limitVelocity: clamp |U| <= max on the given cells (OF fv::limitVelocity::correct, U *= sqrt(max^2/|U|^2) where it
// exceeds max, preserving direction).
void deviceFvoLimitVelocity(const DeviceBuffer<label>& cells, scalar maxU,
                            DeviceBuffer<scalar>& Ux, DeviceBuffer<scalar>& Uy, DeviceBuffer<scalar>& Uz);

// velocityDampingConstraint (OF fv::velocityDampingConstraint::addDamping, called from constrain(eqn) after relax):
// for cells where |U| > UMax, diag[c] += C*V[c]^(2/3)*(|U|-UMax) (implicit-only diagonal sink; all 3 components share
// the matrix diagonal). U is lagged (eqn.psi()). Add to the RELAXED diagonal so it feeds the predictor AND rAU/HbyA.
void deviceFvoVelocityDamping(const DeviceBuffer<label>& cells, scalar UMax, scalar C,
                              const DeviceBuffer<scalar>& V, const DeviceBuffer<scalar>& Ux,
                              const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                              DeviceBuffer<scalar>& diag);

} // namespace brae
