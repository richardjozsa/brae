#pragma once
// brae::matrixA / matrixH, the fvMatrix momentum operations used by the SIMPLE p-U coupling.
//   A() = D/V,  D = diag + cmptAv(boundary internalCoeffs)          (volScalarField)
//   H() = ( (cmptAv(ic)-ic_cmpt)*psi_cmpt  + lduMatrix::H(psi) + source + boundarySource ) / V
// Mirrors OpenFOAM fvMatrix::A()/D()/H() + lduMatrix::H().
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "ldu_matrix.cuh"
#include "geometric_field.cuh"
#include "fvc.cuh"   // SurfaceScalarField
#include <cmath>
#include <vector>

namespace brae {

// pEqn.flux(): conservative face flux of a solved matrix. Mirrors fvMatrix::flux() (orthogonal,
// no faceFluxCorrection): internal = faceH(p) = upper*p[nei] - lower*p[own]; boundary =
// internalCoeffs*p[faceCell] - boundaryCoeffs.
inline SurfaceScalarField matrixFlux(
    const FvScalarMatrix& M,
    const GeometricField<scalar>& p,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches)
{
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    SurfaceScalarField flux;
    flux.internal.resize(nIf);
    for (label f = 0; f < nIf; ++f)
        flux.internal[f] = M.upper[f] * p.internal[nei[f]] - M.lower[f] * p.internal[own[f]];
    flux.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        flux.boundary[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
            flux.boundary[pi][i] = M.internalCoeffs[pi][i] * p.internal[patches[pi].faceCells[i]]
                                 - M.boundaryCoeffs[pi][i];
    }
    return flux;
}

// fvMatrix::setValues: fix psi at the given cells to the given values (epsilonWallFunction
// near-wall constraint). Mirrors fvMatrix<Type>::setValuesFromList exactly:
//   - internal faces of a constrained cell: move the coupling to the neighbour source, zero
//     upper/lower (asymmetric -> both);
//   - BOUNDARY faces of a constrained cell: zero internalCoeffs/boundaryCoeffs (else a boundary
//     coeff, e.g. an outlet div(phi) flux on a wall/outlet corner cell, gets folded into the
//     diagonal at solve time and pulls the pinned cell off its value);
//   - then, in a SECOND loop, psi[c]=value and source[c]=value*diag[c] (so adjacent constrained
//     cells do not corrupt each other's source).
inline void setValues(
    FvScalarMatrix& M,
    std::vector<scalar>& psi,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    const std::vector<label>& cells,
    const std::vector<scalar>& values)
{
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    std::vector<std::vector<std::pair<label, bool>>> adj(m.nCells());      // cell -> (internal face, isOwner)
    for (label f = 0; f < nIf; ++f)
    {
        adj[own[f]].push_back({f, true});
        adj[nei[f]].push_back({f, false});
    }
    std::vector<std::vector<std::pair<label, label>>> badj(m.nCells());    // cell -> (patch, patchFace)
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i)
            badj[patches[pi].faceCells[i]].push_back({(label)pi, i});

    for (std::size_t i = 0; i < cells.size(); ++i)
    {
        const label c = cells[i];
        const scalar v = values[i];
        for (const auto& fe : adj[c])
        {
            const label f = fe.first;
            if (fe.second) M.source[nei[f]] -= M.lower[f] * v;   // c is owner
            else           M.source[own[f]] -= M.upper[f] * v;   // c is neighbour
            M.upper[f] = 0.0;
            M.lower[f] = 0.0;
        }
        for (const auto& be : badj[c])                           // zero this cell's boundary coeffs
        {
            M.internalCoeffs[be.first][be.second] = 0.0;
            M.boundaryCoeffs[be.first][be.second] = 0.0;
        }
    }
    for (std::size_t i = 0; i < cells.size(); ++i)               // set source AFTER, per OF
    {
        const label c = cells[i];
        psi[c]      = values[i];
        M.source[c] = values[i] * M.diag[c];
    }
}

// fvMatrix::relax(alpha): diagonal-dominance + under-relaxation of the matrix (modifies diag and
// source). No coupled interfaces here.
template <typename T>
void relaxMatrix(
    FvMatrix<T>& M,
    const GeometricField<T>& psi,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches,
    scalar alpha)
{
    if (alpha <= 0.0) return;
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();

    const std::vector<scalar> D0 = M.diag;
    std::vector<scalar> sumOff(nC, 0.0);
    for (label f = 0; f < nIf; ++f)
    {
        sumOff[own[f]] += std::fabs(M.upper[f]);
        sumOff[nei[f]] += std::fabs(M.lower[f]);
    }

    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i)
            M.diag[patches[pi].faceCells[i]] += cmptMagMax(M.internalCoeffs[pi][i]);
    for (label c = 0; c < nC; ++c)
        M.diag[c] = std::fmax(std::fabs(M.diag[c]), sumOff[c]) / alpha;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i)
            M.diag[patches[pi].faceCells[i]] -= cmptMin(M.internalCoeffs[pi][i]);
    for (label c = 0; c < nC; ++c)
        M.source[c] += (M.diag[c] - D0[c]) * psi.internal[c];
}

template <typename T>
std::vector<scalar> matrixA(
    const FvMatrix<T>& M,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const label nC = m.nCells();
    std::vector<scalar> D = M.diag;
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i)
            D[patches[pi].faceCells[i]] += cmptAv(M.internalCoeffs[pi][i]);
    std::vector<scalar> A(nC);
    for (label c = 0; c < nC; ++c)
        A[c] = D[c] / g.V()[c];
    return A;
}

inline std::vector<vector> matrixH(
    const FvVectorMatrix& M,
    const GeometricField<vector>& U,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    std::vector<vector> H(nC, vector{0, 0, 0});

    // Boundary-diagonal component term ( (cmptAv(ic)-ic_cmpt)*psi_cmpt ); 0 for uniform BCs.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const label c = patches[pi].faceCells[i];
            const vector ic = M.internalCoeffs[pi][i];
            const scalar av = cmptAv(ic);
            H[c].x += (av - ic.x) * U.internal[c].x;
            H[c].y += (av - ic.y) * U.internal[c].y;
            H[c].z += (av - ic.z) * U.internal[c].z;
        }
    // lduMatrix::H(psi): negated off-diagonal multiply.
    for (label f = 0; f < nIf; ++f)
    {
        H[nei[f]] = H[nei[f]] - M.lower[f] * U.internal[own[f]];
        H[own[f]] = H[own[f]] - M.upper[f] * U.internal[nei[f]];
    }
    for (label c = 0; c < nC; ++c)
        H[c] += M.source[c];                       // source
    for (std::size_t pi = 0; pi < patches.size(); ++pi)                       // boundarySource
        for (label i = 0; i < patches[pi].size; ++i)
            H[patches[pi].faceCells[i]] += M.boundaryCoeffs[pi][i];
    for (label c = 0; c < nC; ++c)
        H[c] = H[c] / g.V()[c];
    return H;
}

} // namespace brae
