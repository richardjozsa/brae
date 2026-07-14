#pragma once
// brae::pbicgstab, preconditioned BiCGStab with DILU preconditioner, transcribed from
// OpenFOAM lduMatrix PBiCGStab + DILUPreconditioner. For asymmetric matrices (convection /
// segregated momentum). Completes the matrix like fvMatrix::solve, solves A*psi = b in place.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_patch.cuh"
#include "ldu_matrix.cuh"
#include "pcg.cuh"   // SolverPerformance
#include <vector>

namespace brae {

SolverPerformance pbicgstab(
    const FvScalarMatrix& M,
    std::vector<scalar>& psi,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    scalar tolerance,
    scalar relTol,
    int maxIter,
    int minIter = 0);

} // namespace brae
