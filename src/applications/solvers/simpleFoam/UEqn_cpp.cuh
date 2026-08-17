#pragma once
// _cpp REFERENCE -- host transcription of simpleFoam's momentum predictor.
//
// provenance:
//   openfoam:
//     file: applications/solvers/incompressible/simpleFoam/UEqn.H:1-24
//   brae:
//     reference: src/applications/solvers/simpleFoam/UEqn_cpp.cu
//     cuda:      (pending)
//     tests:     tests/test_ueqn_cpp.cu
//
// OpenFOAM, verbatim:
//
//     MRF.correctBoundaryVelocity(U);
//
//     tmp<fvVectorMatrix> tUEqn
//     (
//         fvm::div(phi, U)
//       + MRF.DDt(U)
//       + turbulence->divDevReff(U)
//      ==
//         fvOptions(U)
//     );
//     fvVectorMatrix& UEqn = tUEqn.ref();
//
//     UEqn.relax();
//
//     fvOptions.constrain(UEqn);
//
//     if (simple.momentumPredictor())
//     {
//         solve(UEqn == -fvc::grad(p));
//
//         fvOptions.correct(U);
//     }
//
// WHAT THIS REFERENCE COVERS, and what it refuses.
//
// Covered: fvm::div(phi,U), turbulence->divDevReff(U) (both halves), UEqn.relax(), and the -fvc::grad(p)
// right-hand side. That is the whole of UEqn.H for a case with no MRF and no fvOptions -- which is
// pitzDaily, airFoil2D and the rest of the incompressible validation set.
//
// REFUSED, not ignored: MRF and fvOptions. Both appear in UEqn.H and both change the answer. brae has
// already shipped a solver that silently ignored MRFProperties on the compressible path and produced a
// converged, plausible, wrong result; assembleUEqn therefore takes explicit flags and the caller is
// expected to refuse rather than proceed. A component that is out of scope must say so at run time, not
// be absent from the code and therefore from the reader's attention.
//
// ORDER MATTERS and is OpenFOAM's, not a rearrangement:
//   1. assemble div + divDevReff        (divDevReff contributes to BOTH the matrix and the source)
//   2. relax                            (fvMatrix::relax is asymmetric -- see fv_matrix_ops.cuh)
//   3. add the pressure gradient        (an explicit source, added AFTER relaxation)
// Relaxing after adding grad(p) would relax the pressure gradient too, which OpenFOAM does not do:
// `solve(UEqn == -fvc::grad(p))` builds a new equation from the ALREADY-RELAXED UEqn.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "ldu_matrix.cuh"
#include "fvc.cuh"
#include <vector>

namespace brae {
namespace cpu {

// Everything UEqn.H needs that is not the mesh. Grouped so the call site reads like the OpenFOAM text.
struct MomentumInput
{
    const std::vector<scalar>*              phi = nullptr;       // internal face flux
    const std::vector<std::vector<scalar>>* phiBnd = nullptr;    // boundary face flux
    const std::vector<scalar>*              nuEff = nullptr;     // cells
    const std::vector<std::vector<scalar>>* nuEffBnd = nullptr;  // [patch][face]
    scalar relaxU = 1.0;                                         // relaxationFactors/equations U
    bool   hasMRF = false;                                       // present in the case -> must refuse
    bool   hasFvOptions = false;                                 // present in the case -> must refuse
};

// Step 1+2 of UEqn.H: fvm::div(phi,U) + turbulence->divDevReff(U), then UEqn.relax().
//
// Returns the relaxed momentum matrix WITHOUT the pressure gradient, because that is the object pEqn.H
// needs: rAU = 1/UEqn.A() and UEqn.H() are both taken from the relaxed matrix before -grad(p) is applied.
FvVectorMatrix assembleUEqn(
    const GeometricField<vector>& U,
    const MomentumInput&          in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches);

// The convection + implicit-stress core ONLY: fvm::div(phi,U) - fvm::laplacian(nuEff,U), with no explicit
// dev2 term and no relaxation.
//
// This exists because OpenFOAM can be made to dump exactly this and nothing else
// (validation/matrixDumpAsym/momentum.dat), which makes it a directly checkable oracle. Splitting the
// assembly here means a disagreement can be attributed to convection/diffusion or to the explicit stress,
// instead of being one number covering both.
FvVectorMatrix momentumCore(
    const GeometricField<vector>& U,
    const MomentumInput&          in,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches);

// Step 3: the right-hand side of `solve(UEqn == -fvc::grad(p))`.
// Added to source as -grad(p)*V, extensive, after relaxation.
void addPressureGradient(
    FvVectorMatrix&               UEqn,
    const GeometricField<scalar>& p,
    const PrimitiveMesh&          m,
    const FvGeometry&             g,
    const std::vector<FvPatch>&   patches);

} // namespace cpu
} // namespace brae
