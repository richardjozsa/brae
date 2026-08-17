#pragma once
// DEShybrid: the per-face blend of a low-dissipation and an upwind-biased convection scheme.
// See deshybrid_coeffs.cuh for the sensor OF defines and how it reduces to a deferred correction here.
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "deshybrid_coeffs.cuh"

namespace brae {

// sigma per CELL from the DES sensor. gradU is the OF-convention 9-component tensor (gradU[q*nC + c],
// q = 3i + j); the filter width is cbrt(V) (cubeRootVol), so the CELL VOLUMES come in directly.
void deviceDesHybridSigma(int nC, const DeviceBuffer<scalar>& gradU, const DeviceBuffer<scalar>& V,
                          const DeviceBuffer<scalar>& nut, scalar nu, const DesHybridCoeffs& co,
                          DeviceBuffer<scalar>& sigma, const DeviceBuffer<scalar>* delta = nullptr);

// The blended deferred correction for one component: corr = div(phi * [(1-bf)*linCorr + bf*luCorr]),
// bf = interpolate(sigma). gx/gy/gz are the gradient linearUpwind names, `field` the component solved for.
void deviceDesHybridCorr(const DeviceMesh& dm, const DeviceBuffer<scalar>& phiInt,
                         const DeviceBuffer<scalar>& sigma, const DeviceBuffer<scalar>& field,
                         const DeviceBuffer<scalar>& gx, const DeviceBuffer<scalar>& gy,
                         const DeviceBuffer<scalar>& gz, DeviceBuffer<scalar>& corr);

} // namespace brae
