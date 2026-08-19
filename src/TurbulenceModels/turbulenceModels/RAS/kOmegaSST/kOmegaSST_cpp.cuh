#pragma once
// _cpp REFERENCE -- host transcription of OpenFOAM's kOmegaSST.
//
// provenance:
//   openfoam:
//     file: src/TurbulenceModels/turbulenceModels/RAS/kOmegaSST/kOmegaSSTBase.C
//     symbols: kOmegaSSTBase::correct / F1 / F2 / F23 / GbyNu0 / GbyNu / Pk / epsilonByk / correctNut
//   brae:
//     reference: src/TurbulenceModels/turbulenceModels/RAS/kOmegaSST/kOmegaSST_cpp.cu
//     cuda:      src/cuda/device_komega_sst.cu   (deviceKOmegaSSTCorrect)
//     tests:     tests/test_komegasst_cpp.cu, tests/komegasst_vs_openfoam.sh
//
// WHY A HOST REFERENCE AT ALL, when a validated CUDA kOmegaSST already exists: the same reason every other
// component of this port has one. The device model is a single fused entry point; a disagreement with
// OpenFOAM in it is one number covering production, the two blending functions, the eddy-viscosity
// limiter, two transport equations and two wall functions. The reference is stage-addressable, so a
// disagreement names the stage. kEpsilon was wired into the rebuilt solver WITHOUT one, which is a gap in
// this port and not a precedent to follow.
//
// EVERY EXPRESSION BELOW IS TRANSCRIBED, NOT RE-DERIVED. In OpenFOAM's own notation:
//
//   S2       = 2*magSqr(symm(gradU))                                      (kOmegaSSTBase.C:137)
//   GbyNu0   = gradU && devTwoSymm(gradU)                                 (:168-181)
//   G        = nut*GbyNu0                                                 (:525)
//   CDkOmega = (2*alphaOmega2)*(grad(k) & grad(omega))/omega              (:530)
//   F1       = tanh(pow4(arg1)),
//              arg1 = min(min(max((1/betaStar)*sqrt(k)/(omega*y),
//                                 500*nu/(sqr(y)*omega)),
//                             (4*alphaOmega2)*k/(max(CDkOmega,1e-10)*sqr(y))), 10)
//   F2       = tanh(sqr(arg2)),
//              arg2 = min(max((2/betaStar)*sqrt(k)/(omega*y), 500*nu/(sqr(y)*omega)), 100)
//   F23      = F2 (times F3 when the F3 switch is on; F3 is not ported and is refused)
//   blend    = F1*(psi1 - psi2) + psi2                                    (kOmegaSSTBase.H:blend)
//   GbyNu    = min(GbyNu0, (c1/a1)*betaStar*omega*max(a1*omega, b1*F23*sqrt(S2)))
//   Pk(G)    = min(G, (c1*betaStar)*k*omega)
//   epsilonByk = betaStar*omega                                           (:157-165)
//   nut      = a1*k/max(a1*omega, b1*F23*sqrt(S2))                        (:117-126)
//   DkEff    = alphaK(F1)*nut + nu,  DomegaEff = alphaOmega(F1)*nut + nu
//
// y IS THE CELL WALL DISTANCE, at every cell -- wallDist::New(mesh).y() -- NOT the near-wall face distance
// the wall functions use. OpenFOAM writes both as `y` and they are different fields; F1/F2 need the former.
//
// REFUSED, not ignored: the F3 near-wall switch, decayControl (with kInf/omegaInf), MRF and fvOptions.
// ofscan lists 17 keys on kOmegaSSTBase; the three this reference does not implement are exactly those.
#include "cf_types.cuh"
#include "komega_sst_coeffs.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "ldu_matrix.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"
#include <vector>

namespace brae {
namespace cpu {
namespace kOmegaSST {

// S2 = 2*magSqr(symm(gradU)).
std::vector<scalar> S2(const std::vector<tensor>& gradU);

// GbyNu0 = gradU && devTwoSymm(gradU). devTwoSymm(t) = (t + t^T) - (2/3)*tr(t)*I.
std::vector<scalar> GbyNu0(const std::vector<tensor>& gradU);

// CDkOmega = (2*alphaOmega2)*(grad(k) & grad(omega))/omega.
std::vector<scalar> CDkOmega(const std::vector<vector>& gradK,
                             const std::vector<vector>& gradOmega,
                             const std::vector<scalar>& omega,
                             const KOmegaSSTCoeffs& co);

// The two blending functions. `y` is the CELL wall distance; `nu` the laminar kinematic viscosity.
std::vector<scalar> F1(const std::vector<scalar>& k, const std::vector<scalar>& omega,
                       const std::vector<scalar>& y, const std::vector<scalar>& CDkOmega,
                       scalar nu, const KOmegaSSTCoeffs& co);
std::vector<scalar> F2(const std::vector<scalar>& k, const std::vector<scalar>& omega,
                       const std::vector<scalar>& y, scalar nu, const KOmegaSSTCoeffs& co);

// nut = a1*k/max(a1*omega, b1*F23*sqrt(S2)).
std::vector<scalar> correctNut(const std::vector<scalar>& k, const std::vector<scalar>& omega,
                               const std::vector<scalar>& F23, const std::vector<scalar>& S2,
                               const KOmegaSSTCoeffs& co);

// The initial residuals of the two assembled transport equations, in OpenFOAM's normalisation, WITHOUT
// solving anything. This is the honest oracle for "does our discretisation agree with OpenFOAM": at
// OpenFOAM's converged state both must be small, and that is a statement about the equations rather than
// about how far a solve happens to move a field. (The obvious test -- run one correct() and check the
// field does not move -- is NOT valid here: OpenFOAM itself stops on a residual plateau, not at an exact
// fixed point, so solving from its state to 1e-12 moves the field by however much that plateau is worth.)
struct SSTResiduals { scalar omega = 0, k = 0; };

// One kOmegaSST::correct(): the whole model, updating k, omega and nut in place.
//
// Refuses (throws) rather than silently approximating: F3, decayControl, and a case with no wall.
void correct(
    const GeometricField<vector>&  U,
    GeometricField<scalar>&        k,
    GeometricField<scalar>&        omega,
    GeometricField<scalar>&        nutField,
    const SurfaceScalarField&      phi,
    const std::vector<scalar>&     y,          // CELL wall distance
    scalar                         nu,
    const PrimitiveMesh&           m,
    const FvGeometry&              g,
    const std::vector<FvPatch>&    patches,
    scalar                         relaxOmega,
    scalar                         relaxK,
    scalar                         tol,
    scalar                         relTol,
    int                            maxIter,
    const KOmegaSSTCoeffs&         co = {},
    SSTResiduals*                  res = nullptr,    // the two INITIAL residuals, in OF's normalisation
    // div(phi,k) and div(phi,omega). The SST tutorials ask for `bounded Gauss limitedLinear 1` on both,
    // which is a DIFFERENT matrix from upwind -- limitedLinear supplies the convection weights, and
    // `bounded` subtracts Sp(fvc::div(phi), var). pitzDaily's kEpsilon asks for plain `Gauss upwind`,
    // so the defaults keep that and every existing call site is unchanged.
    bool                           bounded = false,
    bool                           limitedLinear = false,
    scalar                         limiterCoeff = 1.0);

} // namespace kOmegaSST
} // namespace cpu
} // namespace brae
