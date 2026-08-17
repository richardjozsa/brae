#pragma once
// Maxwell viscoelastic laminar model -- OF laminarModels::Maxwell (TurbulenceModels/laminar/Maxwell).
//
// A polymer solution does not respond to strain instantly: the stress relaxes over a time lambda. OF
// carries that as a transported SYMMETRIC TENSOR sigma (6 components) with its own equation, and feeds
// it back into the momentum equation as an extra divergence:
//
//   stress transport   ddt(sigma) + div(phi, sigma) + Sp(1/lambda, sigma) == P
//   with               P = twoSymm(sigma & gradU) - nuM/lambda * twoSymm(gradU)
//
//   momentum           divDevRhoReff = div(nuM*grad(U)) + div(sigma)
//                                    - div(nu*dev2(T(grad(U)))) - laplacian(nu + nuM, U)
//
// Note what is NOT there: no nut anywhere. This is a LAMINAR model -- the extra viscosity nuM is a
// material property of the fluid, not a turbulence closure, and the implicit laplacian carries
// nu0 = nu + nuM while the dev2 transpose term keeps the MOLECULAR nu alone. Running it as a Newtonian
// fluid with nu is a different fluid: on pimpleFoam/laminar/planarPoiseuille (nuM 1, lambda 5) the
// startup is an elastic oscillation that a Newtonian solve does not have at all.
//
// The six components are solved as six scalar transports sharing one convection matrix, coupled only
// through the explicit P -- the same shape as OF's fvSymmTensorMatrix, which is also segregated.
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"

namespace brae {

// OF component order for a symmTensor, which is also its dictionary order.
enum SymmIdx { SXX = 0, SXY = 1, SXZ = 2, SYY = 3, SYZ = 4, SZZ = 5 };

// P = twoSymm(sigma & gradU) - nuM*rLambda*twoSymm(gradU), per cell, six components.
//
//   (sigma & gradU)_ij = sigma_ik gradU_kj          [gradU packed q = 3i + j = d(U_j)/d(x_i)]
//   twoSymm(T)         = T + T^T
//
// so P is symmetric by construction and only its six independent components are formed.
void deviceMaxwellP(int nC,
                    const DeviceBuffer<scalar>* sigma,   // sigma[6], cell fields
                    const DeviceBuffer<scalar>& gradU,   // packed 9 x nC
                    scalar nuM, scalar rLambda,
                    DeviceBuffer<scalar>* P);            // P[6], out

// The sigma equation's reaction half: diag += V/lambda (the fvm::Sp relaxation sink) and
// source += V*P. Signature matches the Reaction functor deviceSolveScalarTransport expects.
void deviceMaxwellReaction(const DeviceBuffer<scalar>& V, scalar rLambda,
                           const DeviceBuffer<scalar>& Pc,
                           DeviceBuffer<scalar>& diag, DeviceBuffer<scalar>& source);

// fvc::div of a symmetric tensor field: out_j = (1/V) sum_f Sf_i T_ij, Gauss with linear
// interpolation (`div(sigma) Gauss linear` in both Maxwell tutorials). Boundary faces use the field's
// own boundary values, so a fixedValue sigma inlet enters the momentum equation as OF's does.
void deviceDivSymmTensor(const DeviceMesh& dm,
                         const DeviceBuffer<scalar>* sigCell,   // [6]
                         const DeviceBuffer<scalar>* sigBnd,    // [6], boundary faces
                         DeviceBuffer<scalar>& outX, DeviceBuffer<scalar>& outY, DeviceBuffer<scalar>& outZ);

// fvc::div(nuM*grad(U)) -- the OTHER new momentum term. grad(U) is the cell gradient, and its BOUNDARY
// value is OF's gaussGrad-corrected one (normal component replaced by the patch snGrad), not the bare
// extrapolation: at a no-slip wall that difference IS the wall shear.
void deviceDivNuMGradU(const DeviceMesh& dm, const DeviceVectorBoundary& dbU,
                       const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                       const DeviceBuffer<scalar>& gradU,   // packed 9 x nC (the case's grad(U) scheme)
                       scalar nuM,
                       DeviceBuffer<scalar>& outX, DeviceBuffer<scalar>& outY, DeviceBuffer<scalar>& outZ);

// magSqr(sigma) per cell -- the scalar the vanAlbada limiter is built from. OF's limited schemes for a
// non-scalar type take limitFuncs::magSqr, so ONE limiter per face serves all six components.
void deviceSymmMagSqr(int nC, const DeviceBuffer<scalar>* sigma, DeviceBuffer<scalar>& out);

} // namespace brae
