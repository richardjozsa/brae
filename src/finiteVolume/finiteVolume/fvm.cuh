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
#include <type_traits>
#include <vector>

namespace brae {
namespace fvm {

// corrVec & grad(vf), the contraction OpenFOAM writes as `nonOrthCorrectionVectors() & interpolate(grad)`.
// scalar field -> grad is a vector -> result scalar; vector field -> grad is a tensor -> result vector.
inline scalar dotCorr(const vector& c, const vector& gradS)
{
    return c.x*gradS.x + c.y*gradS.y + c.z*gradS.z;
}
inline vector dotCorr(const vector& c, const tensor& gradV)
{
    // grad(U)_ij = d(U_j)/d(x_i) in OpenFOAM's convention, so the contraction is over the FIRST index.
    return { c.x*gradV.xx + c.y*gradV.yx + c.z*gradV.zx,
             c.x*gradV.xy + c.y*gradV.yy + c.z*gradV.zy,
             c.x*gradV.xz + c.y*gradV.yz + c.z*gradV.zz };
}

// laplacian with a face-varying diffusivity gammaf (e.g. interpolate(rAU)) for the pEqn.
//
// `corrected` selects OpenFOAM's NON-ORTHOGONAL correction, and it changes TWO things, not one
// (gaussLaplacianScheme.C, correctedSnGrad.H:108-119):
//
//   implicit:  the face coefficient uses nonOrthDeltaCoeffs = 1/max(n.delta, 0.05|delta|)
//              instead of deltaCoeffs -- correctedSnGrad::deltaCoeffs() returns mesh.nonOrthDeltaCoeffs()
//   explicit:  source -= V * fvc::div( gamma*magSf * (corrVecs & interpolate(grad(vf))) )
//              with corrVecs = n - delta*nonOrthDeltaCoeffs, ZERO on boundary faces
//              (basicFvGeometryScheme.C:266 and makeNonOrthCorrectionVectors)
//
// Only the implicit half lives here; the explicit source is laplacianNonOrthSource() below, kept separate
// because it is a DEFERRED correction -- it uses the current vf and so must be rebuilt every iteration,
// while the matrix coefficients are geometry. Splitting them also lets each be compared against OpenFOAM
// on its own rather than as one number.
template <typename T>
FvMatrix<T> laplacian(
    const SurfaceScalarField& gammaf,
    const GeometricField<T>& vf,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    bool corrected)
{
    const label nC  = m.nCells();
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& dc    = corrected ? g.nonOrthDeltaCoeffs() : g.deltaCoeffs();
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

// The EXPLICIT non-orthogonal correction, as an extensive per-cell source contribution:
//     V * fvc::div( gamma*magSf * (corrVecs & interpolate(grad(vf))) )
// which OpenFOAM SUBTRACTS from the laplacian's source (gaussLaplacianScheme.C). Returned rather than
// applied so the caller supplies the sign for its own equation, and so it can be compared on its own.
//
// Boundary faces contribute nothing: OpenFOAM sets the correction vectors to zero there
// (makeNonOrthCorrectionVectors). The 1/V of fvc::div and the V of the extensive source cancel, so no
// volume factor appears below.
//
// G is the gradient's element type (vector for a scalar field, tensor for a vector field), taken as a
// second template parameter -- deriving it with std::conditional puts it in a non-deduced context and
// breaks overload resolution for every other laplacian call site.
template <typename T, typename G>
std::vector<T> laplacianNonOrthSource(
    const SurfaceScalarField& gammaf,
    const GeometricField<T>& vf,
    const std::vector<G>& gradVf,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    (void)vf; (void)patches;
    const label nC  = m.nCells();
    const label nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& w  = g.weights();
    const std::vector<scalar>& magSf = g.magSf();
    const std::vector<vector>& cv = g.nonOrthCorrectionVectors();

    std::vector<T> src(nC, T{});
    for (label f = 0; f < nIf; ++f)
    {
        const G gf = w[f] * gradVf[own[f]] + (1.0 - w[f]) * gradVf[nei[f]];
        const T corr = (gammaf.internal[f] * magSf[f]) * dotCorr(cv[f], gf);
        src[own[f]] += corr;
        src[nei[f]] -= corr;
    }
    return src;
}

// Orthogonal (OpenFOAM `orthogonal`/`uncorrected`): the original 5-argument form, behaviour unchanged.
template <typename T>
FvMatrix<T> laplacian(
    const SurfaceScalarField& gammaf,
    const GeometricField<T>& vf,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    return laplacian<T>(gammaf, vf, m, g, patches, false);
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
