#pragma once
// cf k-epsilon turbulence: production and eddy viscosity, transcribed from OpenFOAM kEpsilon.
//   GbyNu = gradU && devTwoSymm(gradU);  devTwoSymm(g) = (g + g^T) - (2/3)*tr(g)*I
//   G     = nut * GbyNu                       (turbulence production)
//   nut   = Cmu * k^2 / epsilon
// Constants are OF defaults: Cmu 0.09, C1 1.44, C2 1.92, sigmak 1.0, sigmaEps 1.3.
#include "cf_types.cuh"
#include "kepsilon_coeffs.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "ldu_matrix.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"
#include "nut_wall_function.cuh"
#include "near_wall_dist.cuh"
#include "pbicgstab.cuh"
#include <cmath>
#include <vector>

namespace brae {
namespace kepsilon {

constexpr scalar Cmu     = 0.09;
constexpr scalar C1      = 1.44;
constexpr scalar C2      = 1.92;
constexpr scalar sigmaK  = 1.0;
constexpr scalar sigmaEps = 1.3;

inline std::vector<scalar> GbyNu(const std::vector<tensor>& gradU)
{
    std::vector<scalar> g(gradU.size());
    for (std::size_t c = 0; c < gradU.size(); ++c)
    {
        const tensor t  = gradU[c];
        const scalar t23 = (2.0 / 3.0) * tr(t);
        tensor dts = t + transpose(t);   // twoSymm
        dts.xx -= t23;                   // devTwoSymm
        dts.yy -= t23;
        dts.zz -= t23;
        g[c] = doubleDot(t, dts);
    }
    return g;
}

inline std::vector<scalar> nut(
    const std::vector<scalar>& k,
    const std::vector<scalar>& epsilon,
    const KEpsilonCoeffs& co = {})
{
    std::vector<scalar> nu(k.size());
    for (std::size_t c = 0; c < k.size(); ++c)
        nu[c] = co.Cmu * k[c] * k[c] / epsilon[c];
    return nu;
}

inline std::vector<scalar> production(
    const std::vector<scalar>& nutf,
    const std::vector<scalar>& gByNu)
{
    std::vector<scalar> G(nutf.size());
    for (std::size_t c = 0; c < nutf.size(); ++c)
        G[c] = nutf[c] * gByNu[c];
    return G;
}

inline std::vector<scalar> DkEff(
    const std::vector<scalar>& nutf,
    scalar nu,
    const KEpsilonCoeffs& co = {})
{
    std::vector<scalar> d(nutf.size());
    for (std::size_t c = 0; c < nutf.size(); ++c)
        d[c] = nutf[c] / co.sigmaK + nu;
    return d;
}

inline std::vector<scalar> DepsilonEff(
    const std::vector<scalar>& nutf,
    scalar nu,
    const KEpsilonCoeffs& co = {})
{
    std::vector<scalar> d(nutf.size());
    for (std::size_t c = 0; c < nutf.size(); ++c)
        d[c] = nutf[c] / co.sigmaEps + nu;
    return d;
}

// kEqn = div(phi,k) - laplacian(DkEff,k) + Sp(eps/k, k) == G.
inline FvScalarMatrix kEqn(
    const GeometricField<scalar>& k,
    const std::vector<scalar>& eps,
    const std::vector<scalar>& G,
    const std::vector<scalar>& nutf,
    const SurfaceScalarField& phi,
    scalar nu,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    const KEpsilonCoeffs& co = {})
{
    const SurfaceScalarField Dkf = fvc::interpolate(DkEff(nutf, nu, co), m, g, patches);
    FvScalarMatrix M = fvm::div(phi.internal, phi.boundary, k, m, patches);
    addEqual(M, fvm::laplacian(Dkf, k, m, g, patches), -1.0);
    for (label c = 0; c < m.nCells(); ++c)
    {
        M.diag[c]   += (eps[c] / k.internal[c]) * g.V()[c];   // + Sp(eps/k, k)
        M.source[c] += G[c] * g.V()[c];                       // == G
    }
    return M;
}

// epsEqn = div(phi,eps) - laplacian(DepsilonEff,eps) + Sp(C2*eps/k, eps) == C1*Cmu*k*GbyNu.
inline FvScalarMatrix epsilonEqn(
    const GeometricField<scalar>& eps,
    const std::vector<scalar>& k,
    const std::vector<scalar>& GbyNuF,
    const std::vector<scalar>& nutf,
    const SurfaceScalarField& phi,
    scalar nu,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    const KEpsilonCoeffs& co = {})
{
    const SurfaceScalarField Def = fvc::interpolate(DepsilonEff(nutf, nu, co), m, g, patches);
    FvScalarMatrix M = fvm::div(phi.internal, phi.boundary, eps, m, patches);
    addEqual(M, fvm::laplacian(Def, eps, m, g, patches), -1.0);
    for (label c = 0; c < m.nCells(); ++c)
    {
        M.diag[c]   += (co.C2 * eps.internal[c] / k[c]) * g.V()[c];     // + Sp(C2*eps/k, eps)
        M.source[c] += (co.C1 * co.Cmu * k[c] * GbyNuF[c]) * g.V()[c];  // == C1*Cmu*k*GbyNu
    }
    return M;
}

// nut = Cmu*k^2/eps in the interior; nutkWallFunction value on wall patches.
inline std::vector<scalar> correctNut(
    const GeometricField<scalar>& k,
    const GeometricField<scalar>& eps,
    scalar /*nu*/,
    const KEpsilonCoeffs& co = {})
{
    return nut(k.internal, eps.internal, co);
}

// Full kEpsilon::correct(): update nut, assemble + solve epsEqn (with epsilonWallFunction
// near-wall constraint + G override) then kEqn, bounding k/eps. Mirrors OpenFOAM kEpsilon::correct.
inline void correct(
    const GeometricField<vector>& U,
    GeometricField<scalar>& k,
    GeometricField<scalar>& eps,
    GeometricField<scalar>& nutField,
    const SurfaceScalarField& phi,
    scalar nu,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches,
    scalar relaxEps,
    scalar relaxK,
    scalar tol,
    scalar relTol,
    int maxIter,
    const KEpsilonCoeffs& co = {})
{
    const label nC = m.nCells();
    const scalar kappa = co.kappa;
    const scalar Cmu25 = std::pow(co.Cmu, 0.25), Cmu75 = std::pow(co.Cmu, 0.75);
    std::vector<scalar>& nutF = nutField.internal;

    // production from the current nut (= last correctNut, from the previous outer iteration's k/eps).
    const std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
    const std::vector<scalar> gByNu = GbyNu(gradU);
    std::vector<scalar> G = production(nutF, gByNu);

    // divU = fvc::div(phi) for the SuSp compressibility-correction terms (static mesh: absolute=phi).
    const std::vector<scalar> divU = fvc::div(phi, m, g, patches);
    const scalar C3 = co.C3;

    // epsilonWallFunction: near-wall eps + G override (cornerWeights = 1/nWallFaces).
    std::vector<label> nw(nC, 0);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        if (patches[pi].type == "wall")
            for (label i = 0; i < patches[pi].size; ++i)
                ++nw[patches[pi].faceCells[i]];

    const std::vector<std::vector<scalar>> yWall = nearWallDist(m, g, patches);  // OF turbulence.y()
    std::vector<scalar> eps0(nC, 0.0), G0(nC, 0.0);   // weighted accumulators
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type != "wall") continue;
        const FvPatch& wp = patches[pi];
        const std::vector<scalar>& y = yWall[pi];
        const std::vector<scalar> nutw = nutkWallFunction(wp, y, k.internal, nu, co.Cmu, co.kappa, co.E);
        const std::vector<vector>& Uw = U.boundary[pi]->value();
        for (label i = 0; i < wp.size; ++i)
        {
            const label c = wp.faceCells[i];
            const scalar w = 1.0 / nw[c], kc = k.internal[c];
            const scalar magGradUw = mag((Uw[i] - U.internal[c]) * wp.deltaCoeffs[i]);
            eps0[c] += w * Cmu75 * std::pow(kc, 1.5) / (kappa * y[i]);
            G0[c]   += w * (nutw[i] + nu) * magGradUw * Cmu25 * std::sqrt(kc) / (kappa * y[i]);
        }
    }
    std::vector<label> wallCells;
    std::vector<scalar> epsVals;
    for (label c = 0; c < nC; ++c)
        if (nw[c] > 0)
        {
            G[c] = G0[c];                // near-wall production replaced by the wall-function value
            eps.internal[c] = eps0[c];   // near-wall epsilon constraint value
            wallCells.push_back(c);
            epsVals.push_back(eps0[c]);
        }

    // epsilon equation. + fvm::SuSp(((2/3)C1 - C3)*divU, eps): implicit where the coeff > 0,
    // explicit (old eps) where < 0. (Near-wall rows are overwritten by setValues below.)
    FvScalarMatrix epsEqn = epsilonEqn(eps, k.internal, gByNu, nutF, phi, nu, m, g, patches, co);
    for (label c = 0; c < nC; ++c)
    {
        const scalar sp = ((2.0 / 3.0) * co.C1 - C3) * divU[c];
        epsEqn.diag[c]   += g.V()[c] * std::fmax(sp, 0.0);
        epsEqn.source[c] -= g.V()[c] * std::fmin(sp, 0.0) * eps.internal[c];
    }
    relaxMatrix(epsEqn, eps, m, patches, relaxEps);
    setValues(epsEqn, eps.internal, m, patches, wallCells, epsVals);
    pbicgstab(epsEqn, eps.internal, m, patches, tol, relTol, maxIter);
    for (label c = 0; c < nC; ++c)
        eps.internal[c] = std::fmax(eps.internal[c], 1e-15);
    eps.evaluateBoundary();

    // k equation (uses the wall-overridden G). + fvm::SuSp((2/3)*divU, k).
    FvScalarMatrix kEqnM = kEqn(k, eps.internal, G, nutF, phi, nu, m, g, patches, co);
    for (label c = 0; c < nC; ++c)
    {
        const scalar sp = (2.0 / 3.0) * divU[c];
        kEqnM.diag[c]   += g.V()[c] * std::fmax(sp, 0.0);
        kEqnM.source[c] -= g.V()[c] * std::fmin(sp, 0.0) * k.internal[c];
    }
    relaxMatrix(kEqnM, k, m, patches, relaxK);
    pbicgstab(kEqnM, k.internal, m, patches, tol, relTol, maxIter);
    for (label c = 0; c < nC; ++c)
        k.internal[c] = std::fmax(k.internal[c], 1e-15);
    k.evaluateBoundary();

    // correctNut: nut = Cmu k^2/eps; walls -> nutkWallFunction(y, k_new); calculated/empty kept.
    nutF = nut(k.internal, eps.internal, co);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type != "wall") continue;
        nutField.boundary[pi]->setValue(nutkWallFunction(patches[pi], yWall[pi], k.internal, nu, co.Cmu, co.kappa, co.E));
    }
}

} // namespace kepsilon
} // namespace brae
