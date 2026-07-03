#pragma once
// brae::fvc, explicit finite-volume operators (mirrors OpenFOAM finiteVolume/fvc).
// Gauss linear gradient of a volScalarField -> volVectorField (internal field).
//   grad[c] = (1/V)[ sum_internal Sf*p_f (+own/-nei) + sum_boundary Sf_bf*p_bf ]
//   p_f = w*p_own + (1-w)*p_nei  (Gauss linear interpolation)
// The field's boundary must be evaluated first (p.evaluateBoundary()).
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include <vector>

namespace brae {

// A surfaceScalarField (face flux): internal faces + per-patch boundary faces.
struct SurfaceScalarField {
    std::vector<scalar>              internal;   // nInternalFaces
    std::vector<std::vector<scalar>> boundary;   // [patch][face]
};

namespace fvc {

std::vector<vector> gaussGrad(const GeometricField<scalar>& p,
                              const PrimitiveMesh& m,
                              const FvGeometry& g,
                              const std::vector<FvPatch>& patches);

// Gauss gradient of a volVectorField -> volTensorField (grad(U)_ij = sum Sf_i U_j / V).
std::vector<tensor> gaussGrad(const GeometricField<vector>& U,
                              const PrimitiveMesh& m,
                              const FvGeometry& g,
                              const std::vector<FvPatch>& patches);

// flux(U) = interpolate(U) & Sf  (linear interpolation, face-normal flux).
SurfaceScalarField flux(const GeometricField<vector>& U,
                        const PrimitiveMesh& m, const FvGeometry& g,
                        const std::vector<FvPatch>& patches);

// interpolate a volScalarField (cell array) to faces: linear internal; boundary = cell value
// (zeroGradient/extrapolated, as for rAU = 1/A()).
SurfaceScalarField interpolate(const std::vector<scalar>& vol,
                               const PrimitiveMesh& m, const FvGeometry& g,
                               const std::vector<FvPatch>& patches);

// div(phi) = surfaceIntegrate(phi) = (1/V)[ sum_internal phi (+own/-nei) + sum_boundary phi ].
std::vector<scalar> div(const SurfaceScalarField& phi,
                        const PrimitiveMesh& m, const FvGeometry& g,
                        const std::vector<FvPatch>& patches);

// Boundary face gradient tensors of U (per patch), built from the cell gradient and corrected so
// the wall-normal component equals snGrad(U) (OpenFOAM gaussGrad::correctBoundaryConditions):
//   gradU_b = gradU_cell + n (x) ( snGrad(U) - n & gradU_cell ),  n = Sf/|Sf|.
std::vector<std::vector<tensor>> gradUBoundary(const GeometricField<vector>& U,
                                               const std::vector<tensor>& gradUcell,
                                               const PrimitiveMesh& m, const FvGeometry& g,
                                               const std::vector<FvPatch>& patches);

// fvc::div of a volTensorField (cell tensors + per-patch boundary face tensors) -> volVectorField:
//   div(T)[c] = (1/V)[ sum_internal (Sf & T_f)(+own/-nei) + sum_boundary (Sf & T_b) ],  linear interp.
std::vector<vector> div(const std::vector<tensor>& tCell,
                        const std::vector<std::vector<tensor>>& tBnd,
                        const PrimitiveMesh& m, const FvGeometry& g,
                        const std::vector<FvPatch>& patches);

} // namespace fvc
} // namespace brae
