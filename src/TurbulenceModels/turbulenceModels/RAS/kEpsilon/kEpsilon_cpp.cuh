#pragma once
// _cpp REFERENCE -- host transcription of OpenFOAM's kEpsilon.
//
// provenance:
//   openfoam:
//     file: src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon.C
//     symbols: kEpsilon::correct / correctNut
//     wall:    epsilonWallFunctionFvPatchScalarField.C (calculate, createAveragingWeights,
//              manipulateMatrix), nutkWallFunctionFvPatchScalarField.C
//     keys (ofscan schema kEpsilon): Cmu 0.09, C1 1.44, C2 1.92, C3 0, sigmak 1.0, sigmaEps 1.3
//   brae:
//     reference: src/TurbulenceModels/turbulenceModels/RAS/kEpsilon/kEpsilon_cpp.cu
//     cuda:      src/cuda/device_kepsilon.cu (deviceKEpsilonCorrect)
//     tests:     tests/test_kepsilon_cpp.cu
//
// WHY THIS EXISTS AT ALL, this late. kEpsilon is the one turbulence model in this port that was wired
// straight to CUDA without a host reference, and it is the one that turned out to be wrong: on simpleCar
// brae converges 50% away from OpenFOAM, and freezing nut at OpenFOAM's values makes that disappear
// (2.8e-02), which puts the defect in this model and nowhere else. A fused device correct() can only be
// compared as one number; this reference exists so a disagreement names the term.
//
// OpenFOAM's correct(), verbatim in structure:
//
//   GbyNu = gradU && devTwoSymm(gradU)
//   G     = nut*GbyNu
//   epsEqn:  div(phi,eps) - laplacian(DepsilonEff, eps)
//         == C1*GbyNu*Cmu*k  -  SuSp(((2/3)C1 - C3)*divU, eps)  -  Sp(C2*eps/k, eps)
//   kEqn:    div(phi,k)   - laplacian(DkEff, k)
//         == G              -  SuSp((2/3)*divU, k)              -  Sp(eps/k, k)
//   correctNut: nut = Cmu*k^2/eps
//
// The epsilon PRODUCTION is C1*Cmu*k*GbyNu, not C1*G/k -- algebraically the same only when nut is exactly
// Cmu*k^2/eps, which it is not at the wall, where nut comes from the wall function instead.
//
// THE WALL TREATMENT is where this model is easiest to get wrong, and OpenFOAM's own definition is
// narrower than "a wall":
//   * the averaging weight is 1/(number of adjacent faces that are on an epsilonWallFunction PATCH FIELD)
//     -- createAveragingWeights counts BC types, NOT patch types, so a `wall` patch carrying a different
//     epsilon condition must not contribute;
//   * epsilon at those cells is the STEPWISE blender's log branch, Cmu^0.75 k^1.5/(kappa y), with the
//     viscous branch reached only under lowReCorrection (default off);
//   * G at those cells is REPLACED, not added to: (nutw + nuw)*magGradUw*Cmu^0.25*sqrt(k)/(kappa*y);
//   * the epsilon equation's near-wall rows are then fixed by setValues (manipulateMatrix).
#include "cf_types.cuh"
#include "kepsilon_coeffs.cuh"
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
namespace kEpsilonRef {

// The initial residuals of the two transport equations, in OpenFOAM's normalisation, and the wall-cell
// count the averaging weights were built from. The count is reported because it is the quantity that
// silently differs when BC types and patch types disagree.
struct KEResiduals
{
    scalar epsilon = 0;
    scalar k = 0;
    label  wallCells = 0;
    // Per-cell |b - A.psi| for the epsilon equation, so the residual can be located rather than only
    // measured. A residual concentrated at the wall means the wall treatment; at the inlet or outlet, a
    // boundary condition; spread through the interior, the operator.
    std::vector<scalar> epsCellResidual;
};

// GbyNu = gradU && devTwoSymm(gradU).
std::vector<scalar> GbyNu(const std::vector<tensor>& gradU);

// nut = Cmu*k^2/epsilon.
std::vector<scalar> correctNut(
    const std::vector<scalar>& k,
    const std::vector<scalar>& epsilon,
    const KEpsilonCoeffs& co);

// One kEpsilon::correct(). Updates k, epsilon and nut in place.
void correct(
    const GeometricField<vector>& U,
    GeometricField<scalar>& k,
    GeometricField<scalar>& epsilon,
    GeometricField<scalar>& nutField,
    const SurfaceScalarField& phi,
    scalar nu,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    scalar relaxEps,
    scalar relaxK,
    scalar tol,
    scalar relTol,
    int maxIter,
    const KEpsilonCoeffs& co = {},
    KEResiduals* res = nullptr,
    bool bounded = false,    // `bounded Gauss <scheme>` on div(phi,k) and div(phi,epsilon)
    int  dropTerm = 0);      // DIAGNOSTIC: epsilon eqn 1 production, 2 divU SuSp, 3 destruction,
                             // 4 diffusion; k eqn 5 production G, 6 divU SuSp, 7 destruction eps/k,
                             // 8 diffusion nut/sigmak
                             // (epsilonWallFunction's lowReCorrection rides on KEpsilonCoeffs::epsLowRe)

} // namespace kEpsilonRef
} // namespace cpu
} // namespace brae
