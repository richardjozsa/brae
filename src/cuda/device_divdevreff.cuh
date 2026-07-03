#pragma once
// cf GPU offload (#2): the explicit divDevReff stress source on device,
//   source += V * fvc::div( nuEff * dev2(T(grad U)) ).
// Steps: grad U (tensor, = 3 component gaussGrads) -> sigma = nuEff*dev2(transpose(gradU)) (cell) ;
// boundary gradient via snGrad correction (gradU_b = gradC + n (x) (snGrad - n & gradC), snGrad =
// (U_b - U_cell)*deltaCoeffs) -> sigma_b ; tensor divergence (Sf & sigma_face) gathered per cell.
// Tensors are packed component-major: comp (i,j) at index (i*3+j)*n + cell.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_cyclic.cuh"
#include "device_ami.cuh"

namespace brae {

// nuCell (nC), nuBnd (nBndFaces), the effective viscosity at cells / boundary faces (nu for laminar).
// cyc (optional): a periodic interface, its faces contribute to gradU AND to the tensor divergence, so the
// boundary-column stress stays consistent with the interior (else x-invariance drifts on periodic meshes).
void deviceDivDevReff(const DeviceMesh& dm, const DeviceVectorBoundary& dbU,
                      const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                      const DeviceBuffer<scalar>& nuCell, const DeviceBuffer<scalar>& nuBnd,
                      DeviceBuffer<scalar>& srcX, DeviceBuffer<scalar>& srcY, DeviceBuffer<scalar>& srcZ,
                      const DeviceCyclic* cyc = nullptr, const DeviceAMI* ami = nullptr);

} // namespace brae
