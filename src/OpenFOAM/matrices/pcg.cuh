#pragma once
// brae::pcg, preconditioned conjugate gradient with DIC preconditioner, transcribed from
// OpenFOAM lduMatrix PCG + DICPreconditioner + lduMatrix::solver::normFactor. Completes the
// matrix (boundary diag/source) like fvMatrix::solve, then solves A*psi = b in place.
// Symmetric matrices only (DIC). Serial (global reductions are local sums here).
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_patch.cuh"
#include "ldu_matrix.cuh"
#include <vector>

namespace brae {

struct SolverPerformance
{
    scalar initialResidual = 0.0;
    scalar finalResidual   = 0.0;
    int    nIterations      = 0;
};

SolverPerformance pcg(
    const FvScalarMatrix& M,
    std::vector<scalar>& psi,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    scalar tolerance,
    scalar relTol,
    int maxIter,
    int minIter = 0);

} // namespace brae
