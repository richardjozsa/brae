#pragma once
// _cpp REFERENCE -- host transcription of OpenFOAM's TVD-limited and blended convection schemes.
//
// provenance:
//   openfoam:
//     limitedSurfaceInterpolationScheme.C::weights   weights = limiter*CDweights + (1-limiter)*pos0(phi)
//     limitedLinear.H                                limiter = clamp(twoByk*r, 0, 1),  twoByk = 2/max(k,SMALL)
//     NVDTVD.H::r        (scalar)                    r = 2*(gradcf/gradf) - 1
//     NVDVTVDV.H::r      (the `V` variants)          gradf = gradfV & gradfV,  gradcf = gradfV & (d & gradc)
//     LUST.H                                         weights = 0.75*linear + 0.25*upwind
//                                                    correction = 0.25*linearUpwind::correction
//     linearUpwindV.C                                the vector-LIMITED linearUpwind correction
//   brae:
//     reference: src/finiteVolume/interpolation/surfaceInterpolation/limitedSchemes/limitedSchemes_cpp.cu
//     cuda:      src/cuda/device_deferred_correction.cu (deviceDivLimitedCoeffs, deviceLinearUpwindVCorr,
//                                                        deviceLinearCorr)
//     tests:     tests/test_limitedschemes_cpp.cu
//
// EVERY ONE OF THESE SCHEMES IS A WEIGHT CHANGE, A DEFERRED CORRECTION, OR BOTH, and which of the three
// it is decides how it is ported:
//
//   upwind          weights = pos0(phi)                       no correction
//   linearUpwind    weights = pos0(phi)  (derives from upwind) correction from grad(vf)
//   limitedLinear   weights = limiter*CD + (1-limiter)*pos0    no correction
//   LUST            weights = 0.75*CD + 0.25*pos0             correction = 0.25*linearUpwind's
//
// Getting that classification wrong is silent: a scheme ported as "upwind plus a correction" when it is
// really a weight change still converges, and to a plausible answer.
//
// A THIRD BEHAVIOUR, and the easiest of the three to miss: `limitedLinear` APPLIED TO A VECTOR is not
// per-component and is not the V form either. LimitedScheme.H's family macro instantiates it as
// NVDTVD + limitFuncs::magSqr, so the limiter is built from the SCALAR field magSqr(U) and its gradient,
// and only `limitedLinearV` uses NVDVTVDV. So on U:
//     limitedLinear   -> limitedLinearWeights (below) with vf = magSqr(U), gradVf = grad(magSqr(U))
//     limitedLinearV  -> limitedLinearVWeights
// Three plausible readings, one right; the other two converge to plausible wrong answers.
//
// `d` IS THE OWNER-TO-NEIGHBOUR VECTOR, C[nei] - C[own], and the r ratio branches on faceFlux to pick
// which cell's gradient it uses -- so r is NOT symmetric in the two cells and the sign of phi matters.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include <vector>

namespace brae {
namespace cpu {
namespace limitedSchemes {

// pos0(phi): the upwind weights. 1 where the flux leaves the owner, 0 where it enters.
std::vector<scalar> upwindWeights(const std::vector<scalar>& phi);

// LUST: 0.75*linear + 0.25*upwind. LUST also carries 0.25 of linearUpwind's deferred correction, which is
// the caller's job -- LUST.H overrides BOTH weights() and correction() and a port that took only one of
// them would be a different scheme.
std::vector<scalar> lustWeights(const std::vector<scalar>& phi, const FvGeometry& g);

// limitedLinear, scalar form: weights = limiter*CD + (1-limiter)*pos0(phi).
std::vector<scalar> limitedLinearWeights(
    const std::vector<scalar>&        phi,
    const GeometricField<scalar>&     vf,
    const std::vector<vector>&        gradVf,
    scalar                            k,        // the scheme coefficient; `limitedLinear 1` -> k = 1
    const PrimitiveMesh&              m,
    const FvGeometry&                 g);

// limitedLinearV: the same limiter, but r is formed on the VECTOR difference (NVDVTVDV), so all three
// components share one limiter per face instead of being limited independently.
std::vector<scalar> limitedLinearVWeights(
    const std::vector<scalar>&        phi,
    const GeometricField<vector>&     vf,
    const std::vector<tensor>&        gradVf,
    scalar                            k,
    const PrimitiveMesh&              m,
    const FvGeometry&                 g);

} // namespace limitedSchemes
} // namespace cpu
} // namespace brae
