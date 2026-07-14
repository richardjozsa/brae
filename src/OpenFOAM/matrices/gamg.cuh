#pragma once
// brae::gamg, geometric agglomeration multigrid, transcribed from OpenFOAM GAMGSolver:
// faceAreaPair agglomeration (pairGAMGAgglomerate) + matrix coarsening + GaussSeidel smoother
// + V-cycle (nPreSweeps/nPostSweeps/nFinestSweeps) with correction scaling, coarsest level
// solved directly. Symmetric matrices, serial. Same SolverPerformance contract as PCG.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "ldu_matrix.cuh"
#include "pcg.cuh"   // SolverPerformance
#include <vector>

namespace brae {

SolverPerformance gamg(
    const FvScalarMatrix& M,
    std::vector<scalar>& psi,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    scalar tolerance,
    scalar relTol,
    int maxIter,
    int nCellsInCoarsest = 10,
    int nPreSweeps = 0,
    int nPostSweeps = 2,
    int nFinestSweeps = 2);

} // namespace brae
