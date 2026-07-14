#pragma once
// brae::fvm, implicit finite-volume operators (templated on field type T = scalar/vector).
// Matrix coefficients (diag/upper/lower) are scalar; source and BC coupling coefficients are T.
//   laplacian(gamma, vf): orthogonal Gauss laplacian
//     upper=lower=deltaCoeffs*gamma*magSf; diag=negSumDiag;
//     internalCoeffs = gamma*magSf * gradientInternalCoeffs;  boundaryCoeffs = -gamma*magSf * gradientBoundaryCoeffs
//   div(phi, vf): Gauss upwind convection
//     w=pos0(phi); lower=-w*phi; upper=lower+phi; diag=negSumDiag;
//     internalCoeffs = phi_pf * valueInternalCoeffs;  boundaryCoeffs = -phi_pf * valueBoundaryCoeffs
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "ldu_matrix.cuh"
#include "fvc.cuh"   // SurfaceScalarField
#include <vector>

namespace brae {
namespace fvm {

// laplacian with a face-varying diffusivity gammaf (e.g. interpolate(rAU)) for the pEqn.
template <typename T>
FvMatrix<T> laplacian(
    const SurfaceScalarField& gammaf,
    const GeometricField<T>& vf,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const label nC  = m.nCells();
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& dc    = g.deltaCoeffs();
    const std::vector<scalar>& magSf = g.magSf();

    FvMatrix<T> M;
    M.diag.assign(nC, 0.0);
    M.source.assign(nC, T{});
    M.upper.resize(nIf);
    M.lower.resize(nIf);
    for (label f = 0; f < nIf; ++f)
    {
        const scalar coeff = dc[f] * gammaf.internal[f] * magSf[f];
        M.upper[f] = coeff;
        M.lower[f] = coeff;
        M.diag[own[f]] -= coeff;
        M.diag[nei[f]] -= coeff;
    }
    M.internalCoeffs.resize(patches.size());
    M.boundaryCoeffs.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& fp = patches[pi];
        const std::vector<T> gIC = vf.boundary[pi]->gradientInternalCoeffs();
        const std::vector<T> gBC = vf.boundary[pi]->gradientBoundaryCoeffs();
        M.internalCoeffs[pi].resize(fp.size);
        M.boundaryCoeffs[pi].resize(fp.size);
        for (label i = 0; i < fp.size; ++i)
        {
            const scalar pGamma = gammaf.boundary[pi][i] * magSf[fp.start + i];
            M.internalCoeffs[pi][i] =  pGamma * gIC[i];
            M.boundaryCoeffs[pi][i] = (-pGamma) * gBC[i];
        }
    }
    return M;
}

template <typename T>
FvMatrix<T> laplacian(
    const GeometricField<T>& vf,
    scalar gamma,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const label nC  = m.nCells();
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& dc    = g.deltaCoeffs();
    const std::vector<scalar>& magSf = g.magSf();

    FvMatrix<T> M;
    M.diag.assign(nC, 0.0);
    M.source.assign(nC, T{});
    M.upper.resize(nIf);
    M.lower.resize(nIf);

    for (label f = 0; f < nIf; ++f)
    {
        const scalar coeff = dc[f] * gamma * magSf[f];
        M.upper[f] = coeff;
        M.lower[f] = coeff;
        M.diag[own[f]] -= coeff;
        M.diag[nei[f]] -= coeff;
    }

    M.internalCoeffs.resize(patches.size());
    M.boundaryCoeffs.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& fp = patches[pi];
        const std::vector<T> gIC = vf.boundary[pi]->gradientInternalCoeffs();
        const std::vector<T> gBC = vf.boundary[pi]->gradientBoundaryCoeffs();
        M.internalCoeffs[pi].resize(fp.size);
        M.boundaryCoeffs[pi].resize(fp.size);
        for (label i = 0; i < fp.size; ++i)
        {
            const scalar pGamma = gamma * magSf[fp.start + i];
            M.internalCoeffs[pi][i] =  pGamma * gIC[i];
            M.boundaryCoeffs[pi][i] = (-pGamma) * gBC[i];
        }
    }
    return M;
}

template <typename T>
FvMatrix<T> div(
    const std::vector<scalar>& phiInternal,
    const std::vector<std::vector<scalar>>& phiBoundary,
    const GeometricField<T>& vf,
    const PrimitiveMesh& m,
    const std::vector<FvPatch>& patches)
{
    const label nC  = m.nCells();
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();

    FvMatrix<T> M;
    M.diag.assign(nC, 0.0);
    M.source.assign(nC, T{});
    M.upper.resize(nIf);
    M.lower.resize(nIf);

    for (label f = 0; f < nIf; ++f)
    {
        const scalar phi = phiInternal[f];
        const scalar w   = (phi >= 0.0) ? 1.0 : 0.0;
        M.lower[f] = -w * phi;
        M.upper[f] = M.lower[f] + phi;
        M.diag[own[f]] -= M.lower[f];
        M.diag[nei[f]] -= M.upper[f];
    }

    M.internalCoeffs.resize(patches.size());
    M.boundaryCoeffs.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& fp = patches[pi];
        const std::vector<T> vIC = vf.boundary[pi]->valueInternalCoeffs();
        const std::vector<T> vBC = vf.boundary[pi]->valueBoundaryCoeffs();
        M.internalCoeffs[pi].resize(fp.size);
        M.boundaryCoeffs[pi].resize(fp.size);
        for (label i = 0; i < fp.size; ++i)
        {
            const scalar pf = phiBoundary[pi][i];
            M.internalCoeffs[pi][i] =  pf * vIC[i];
            M.boundaryCoeffs[pi][i] = (-pf) * vBC[i];
        }
    }
    return M;
}

} // namespace fvm
} // namespace brae
