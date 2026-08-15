#pragma once
// Smagorinsky LES sub-grid viscosity: the ALGEBRAIC nut = f(gradU, V) (no transport equation). Declares the public
// wrapper used by the pure-LES path in DeviceSimpleSolver (validate() + correctTurbulence()). The caller supplies the
// full 9-component cell gradU (deviceGradU) and the cell volumes; the kernel evaluates OF's Smagorinsky<>::k() ->
// nut = Ck*delta*sqrt(k_sgs) exactly (see smagorinsky_coeffs.cuh for the reduction to the classic Cs form).
#include "device_buffer.cuh"
#include "smagorinsky_coeffs.cuh"

namespace brae {

// nut = Ck*delta*sqrt(k_sgs), delta = cbrt(V), k_sgs from the local production=dissipation balance (OF Smagorinsky).
// gradU is the OF-convention 9-component tensor per cell (gradU[q*nC + c], q = i*3+j), as produced by deviceGradU.
void deviceSmagorinskyNut(int nC, const DeviceBuffer<scalar>& gradU, const DeviceBuffer<scalar>& V,
                          const SmagorinskyCoeffs& co, DeviceBuffer<scalar>& nut);

// WALE (OF LESModels::WALE::correctNut): the other ALGEBRAIC sub-grid nut, same inputs as Smagorinsky.
void deviceWaleNut(int nC, const DeviceBuffer<scalar>& gradU, const DeviceBuffer<scalar>& V,
                   const WaleCoeffs& co, DeviceBuffer<scalar>& nut);

} // namespace brae
