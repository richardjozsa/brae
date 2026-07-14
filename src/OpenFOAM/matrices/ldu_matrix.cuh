#pragma once
// brae::FvMatrix<T>, an assembled fvMatrix: scalar lduMatrix coefficients (diag/upper/lower)
// shared across components, plus the Type (scalar/vector) source and per-patch boundary
// coupling coefficients. Mirrors OpenFOAM fvMatrix<Type> (block-diagonal-scalar segregated).
#include "cf_types.cuh"
#include <vector>

namespace brae {

template <typename T>
struct FvMatrix
{
    std::vector<scalar> diag;     // nCells (raw, before boundary diag)
    std::vector<scalar> upper;    // nInternalFaces
    std::vector<scalar> lower;    // nInternalFaces
    std::vector<T>      source;   // nCells (raw, before boundary source)

    std::vector<std::vector<T>> internalCoeffs;  // [patch][face] -> diag[faceCell] (per component)
    std::vector<std::vector<T>> boundaryCoeffs;  // [patch][face] -> source[faceCell]
};

using FvScalarMatrix = FvMatrix<scalar>;
using FvVectorMatrix = FvMatrix<vector>;

// In-place matrix combination (e.g. fvm::div(phi,U) - fvm::laplacian(nuEff,U)). Same addressing.
template <typename T>
inline void addEqual(FvMatrix<T>& a, const FvMatrix<T>& b, scalar s)
{
    for (std::size_t i = 0; i < a.diag.size();   ++i) a.diag[i]   += s * b.diag[i];
    for (std::size_t i = 0; i < a.upper.size();  ++i) a.upper[i]  += s * b.upper[i];
    for (std::size_t i = 0; i < a.lower.size();  ++i) a.lower[i]  += s * b.lower[i];
    for (std::size_t i = 0; i < a.source.size(); ++i) a.source[i] += s * b.source[i];
    for (std::size_t p = 0; p < a.internalCoeffs.size(); ++p)
        for (std::size_t i = 0; i < a.internalCoeffs[p].size(); ++i)
        {
            a.internalCoeffs[p][i] += s * b.internalCoeffs[p][i];
            a.boundaryCoeffs[p][i] += s * b.boundaryCoeffs[p][i];
        }
}

} // namespace brae
