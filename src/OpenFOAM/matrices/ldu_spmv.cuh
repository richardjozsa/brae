#pragma once
// brae::Amul, sparse matrix-vector product for the lduMatrix (mirrors OpenFOAM
// lduMatrix::Amul). Raw product over the internal addressing (no boundary/interface):
//   Apsi[c] = diag[c]*psi[c]
//   Apsi[neighbour[f]] += lower[f]*psi[owner[f]]      (lower-triangular)
//   Apsi[owner[f]]     += upper[f]*psi[neighbour[f]]  (upper-triangular)
// Boundary diag/source (internalCoeffs/boundaryCoeffs) and coupled interfaces are applied by
// the solver, not here. owner/neighbour are lowerAddr/upperAddr; the GPU phase will add the
// atomic-free ownerStart/losort traversal.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "ldu_matrix.cuh"
#include <vector>

namespace brae {

std::vector<scalar> Amul(
    const FvScalarMatrix& M,
    const std::vector<scalar>& psi,
    const PrimitiveMesh& m);

} // namespace brae
