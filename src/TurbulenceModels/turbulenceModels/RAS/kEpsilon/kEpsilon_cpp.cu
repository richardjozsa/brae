// _cpp REFERENCE implementation -- see kEpsilon_cpp.cuh for the OpenFOAM provenance and the wall note.
#include "kEpsilon_cpp.cuh"
#include "nut_wall_function.cuh"
#include "near_wall_dist.cuh"
#include "pbicgstab.cuh"
#include <cmath>
#include <algorithm>
#include <vector>

namespace brae {
namespace cpu {
namespace kEpsilonRef {

namespace {

// DkEff / DepsilonEff as OpenFOAM builds them: `nut_/sigma + nu()` is a volScalarField, so its BOUNDARY
// value comes from nut's OWN boundary, not from the adjacent cell. fvm::laplacian then takes that
// boundary value for the patch coefficients.
//
// brae's fvc::interpolate gives a surface field whose boundary is the CELL value, which is right for an
// extrapolatedCalculated field like rAU but wrong here: at pitzDaily's inlet OpenFOAM's nut_b is
// Cmu*k_b^2/eps_b = 8.5e-04 from the inlet's fixed k and epsilon, while the adjacent cell's nut is
// several times that. It is the same distinction the momentum path already makes with nuEffBnd -- the
// turbulence path simply never made it.
SurfaceScalarField effectiveDiffusivity(
    const std::vector<scalar>& nutCell,
    const std::vector<scalar>& nutBndFlatPerPatch,
    const GeometricField<scalar>& nutField,
    scalar sigma,
    scalar nu,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    (void)nutBndFlatPerPatch;
    std::vector<scalar> D(nutCell.size());
    for (std::size_t c = 0; c < nutCell.size(); ++c)
    {
        D[c] = nutCell[c] / sigma + nu;
    }
    SurfaceScalarField sf = fvc::interpolate(D, m, g, patches);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar>& nb = nutField.boundary[pi]->value();
        for (label i = 0; i < patches[pi].size; ++i)
        {
            sf.boundary[pi][i] = nb[i] / sigma + nu;
        }
    }
    return sf;
}

} // namespace

std::vector<scalar> GbyNu(const std::vector<tensor>& gradU)
{
    std::vector<scalar> out(gradU.size());
    for (std::size_t c = 0; c < gradU.size(); ++c)
    {
        const tensor& t = gradU[c];
        const scalar tr = t.xx + t.yy + t.zz;

        // devTwoSymm(t) = (t + t^T) - (2/3)*tr(t)*I
        const scalar d[9] =
        {
            2*t.xx - (2.0/3.0)*tr,
            t.xy + t.yx,
            t.xz + t.zx,
            t.yx + t.xy,
            2*t.yy - (2.0/3.0)*tr,
            t.yz + t.zy,
            t.zx + t.xz,
            t.zy + t.yz,
            2*t.zz - (2.0/3.0)*tr
        };
        const scalar gg[9] =
        {
            t.xx, t.xy, t.xz,
            t.yx, t.yy, t.yz,
            t.zx, t.zy, t.zz
        };

        scalar s = 0;
        for (int q = 0; q < 9; ++q)
        {
            s += gg[q] * d[q];
        }
        out[c] = s;
    }
    return out;
}


std::vector<scalar> correctNut(
    const std::vector<scalar>& k,
    const std::vector<scalar>& epsilon,
    const KEpsilonCoeffs& co)
{
    std::vector<scalar> out(k.size());
    for (std::size_t c = 0; c < k.size(); ++c)
    {
        out[c] = co.Cmu * k[c] * k[c] / epsilon[c];
    }
    return out;
}


void correct(
    const GeometricField<vector>& U,
    GeometricField<scalar>& k,
    GeometricField<scalar>& epsilon,
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
    const KEpsilonCoeffs& co,
    KEResiduals* res,
    bool bounded,
    int dropTerm)
{
    const label nC = m.nCells();
    const scalar Cmu25 = std::pow(co.Cmu, 0.25);
    const scalar Cmu75 = std::pow(co.Cmu, 0.75);
    std::vector<scalar>& nutF = nutField.internal;

    const std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
    const std::vector<scalar> gByNu = GbyNu(gradU);
    const std::vector<scalar> divU = fvc::div(phi, m, g, patches);

    std::vector<scalar> G(nC);
    for (label c = 0; c < nC; ++c)
    {
        G[c] = nutF[c] * gByNu[c];
    }

    // createAveragingWeights: count the adjacent faces that carry an epsilonWallFunction PATCH FIELD.
    // brae's boundary factory maps epsilonWallFunction to zeroGradient and applies the constraint here,
    // so the discriminator available at this level is the patch's own wall-ness together with epsilon's
    // BC category; patch type alone would also count a `wall` carrying a plain fixedValue epsilon.
    std::vector<label> nw(nC, 0);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!epsilon.boundary[pi]->isEpsilonWallFunction()) continue;
        for (label i = 0; i < patches[pi].size; ++i)
        {
            ++nw[patches[pi].faceCells[i]];
        }
    }

    const std::vector<std::vector<scalar>> yWall = nearWallDist(m, g, patches);
    std::vector<scalar> eps0(nC, 0.0);
    std::vector<scalar> G0(nC, 0.0);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (!epsilon.boundary[pi]->isEpsilonWallFunction()) continue;

        const FvPatch& wp = patches[pi];
        const std::vector<scalar>& yw = yWall[pi];
        const std::vector<scalar> nutw =
            nutkWallFunction(wp, yw, k.internal, nu, co.Cmu, co.kappa, co.E);
        const std::vector<vector>& Uw = U.boundary[pi]->value();

        for (label i = 0; i < wp.size; ++i)
        {
            const label c = wp.faceCells[i];
            const scalar w = 1.0 / static_cast<scalar>(nw[c]);
            const scalar kc = k.internal[c];
            const scalar magGradUw = mag((Uw[i] - U.internal[c]) * wp.deltaCoeffs[i]);

            eps0[c] += w * Cmu75 * std::pow(kc, 1.5) / (co.kappa * yw[i]);
            G0[c]   += w * (nutw[i] + nu) * magGradUw * Cmu25 * std::sqrt(kc) / (co.kappa * yw[i]);
        }
    }

    std::vector<label> wallCells;
    std::vector<scalar> epsVals;
    for (label c = 0; c < nC; ++c)
    {
        if (nw[c] == 0) continue;
        G[c] = G0[c];
        epsilon.internal[c] = eps0[c];
        wallCells.push_back(c);
        epsVals.push_back(eps0[c]);
    }
    if (res) res->wallCells = static_cast<label>(wallCells.size());

    // epsilon equation
    {
        std::vector<scalar> nutForD = nutF;
        if (dropTerm == 4)
        {
            std::fill(nutForD.begin(), nutForD.end(), 0.0);
        }
        const SurfaceScalarField Df =
            effectiveDiffusivity(nutForD, {}, nutField, co.sigmaEps, nu, m, g, patches);

        FvScalarMatrix M = fvm::div(phi.internal, phi.boundary, epsilon, m, patches);
        addEqual(M, fvm::laplacian(Df, epsilon, m, g, patches), -1.0);

        for (label c = 0; c < nC; ++c)
        {
            const scalar V = g.V()[c];

            // == C1*GbyNu*Cmu*k
            if (dropTerm != 1) M.source[c] += co.C1 * gByNu[c] * co.Cmu * k.internal[c] * V;

            // - SuSp(((2/3)*C1 - C3)*divU, epsilon)
            const scalar sp = (dropTerm == 2) ? 0.0 : ((2.0/3.0) * co.C1 - co.C3) * divU[c];
            M.diag[c]   += V * std::fmax(sp, 0.0);
            M.source[c] -= V * std::fmin(sp, 0.0) * epsilon.internal[c];

            // - Sp(C2*epsilon/k, epsilon)
            if (dropTerm != 3) M.diag[c] += co.C2 * epsilon.internal[c] / k.internal[c] * V;

            // `bounded`: - Sp(div(phi), epsilon). Vanishes where phi is conservative, so it cannot move
            // a converged state -- which is exactly why it needs its own measurement rather than being
            // assumed harmless.
            if (bounded) M.diag[c] -= divU[c] * V;
        }

        relaxMatrix(M, epsilon, m, patches, relaxEps);
        setValues(M, epsilon.internal, m, patches, wallCells, epsVals);
        if (res)
        {
            // |b - A.psi| per cell, on the SAME assembled matrix the solve is about to use, before the
            // solve changes psi. Boundary coefficients are folded in exactly as fvMatrix::solve does.
            std::vector<scalar> diagC = M.diag;
            std::vector<scalar> b = M.source;
            for (std::size_t pi = 0; pi < patches.size(); ++pi)
            {
                for (label i = 0; i < patches[pi].size; ++i)
                {
                    const label c = patches[pi].faceCells[i];
                    diagC[c] += M.internalCoeffs[pi][i];
                    b[c]     += M.boundaryCoeffs[pi][i];
                }
            }
            std::vector<scalar> r(nC);
            for (label c = 0; c < nC; ++c)
            {
                r[c] = b[c] - diagC[c] * epsilon.internal[c];
            }
            const std::vector<label>& own = m.owner();
            const std::vector<label>& nei = m.neighbour();
            for (label f = 0; f < m.nInternalFaces(); ++f)
            {
                r[own[f]] -= M.upper[f] * epsilon.internal[nei[f]];
                r[nei[f]] -= M.lower[f] * epsilon.internal[own[f]];
            }
            for (label c = 0; c < nC; ++c)
            {
                r[c] = std::fabs(r[c]);
            }
            res->epsCellResidual = r;
        }
        const SolverPerformance p = pbicgstab(M, epsilon.internal, m, patches, tol, relTol, maxIter);
        if (res) res->epsilon = p.initialResidual;

        for (label c = 0; c < nC; ++c)
        {
            epsilon.internal[c] = std::fmax(epsilon.internal[c], 1e-15);
        }
        epsilon.evaluateBoundary();
    }

    // k equation
    {
        std::vector<scalar> nutForD = nutF;
        if (dropTerm == 8)
        {
            std::fill(nutForD.begin(), nutForD.end(), 0.0);
        }
        const SurfaceScalarField Df =
            effectiveDiffusivity(nutForD, {}, nutField, co.sigmaK, nu, m, g, patches);

        FvScalarMatrix M = fvm::div(phi.internal, phi.boundary, k, m, patches);
        addEqual(M, fvm::laplacian(Df, k, m, g, patches), -1.0);

        for (label c = 0; c < nC; ++c)
        {
            const scalar V = g.V()[c];

            // == G   (the wall-overridden value at wall cells)
            if (dropTerm != 5) M.source[c] += G[c] * V;

            // - SuSp((2/3)*divU, k)
            const scalar sp = (dropTerm == 6) ? 0.0 : (2.0/3.0) * divU[c];
            M.diag[c]   += V * std::fmax(sp, 0.0);
            M.source[c] -= V * std::fmin(sp, 0.0) * k.internal[c];

            // - Sp(epsilon/k, k)
            if (dropTerm != 7) M.diag[c] += epsilon.internal[c] / k.internal[c] * V;

            if (bounded) M.diag[c] -= divU[c] * V;
        }

        relaxMatrix(M, k, m, patches, relaxK);
        const SolverPerformance p = pbicgstab(M, k.internal, m, patches, tol, relTol, maxIter);
        if (res) res->k = p.initialResidual;

        for (label c = 0; c < nC; ++c)
        {
            k.internal[c] = std::fmax(k.internal[c], 1e-15);
        }
        k.evaluateBoundary();
    }

    // correctNut.
    //
    // OpenFOAM writes this as `nut_ = Cmu*sqr(k_)/epsilon_`, and that is a GeometricField assignment: it
    // sets the BOUNDARY field as well, from k and epsilon's own boundary values. A patch whose nut is
    // `calculated` therefore gets Cmu*k_b^2/eps_b every iteration -- it does NOT keep its initial value.
    // Only a patch carrying a nut wall function has that overwritten afterwards.
    //
    // brae used to set the internal field and the wall patches only, leaving everything else at whatever
    // was read from disk. On pitzDaily the walls carry nutkWallFunction so the gap never showed. On
    // simpleCar a trailing ".*" entry makes nut `calculated` on the walls, and brae held it at the
    // initial uniform 0 while OpenFOAM had 0.6 to 2.5 there -- a wall viscosity wrong by everything,
    // which is what a 50% different flow looks like.
    nutF = correctNut(k.internal, epsilon.internal, co);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (epsilon.boundary[pi]->isEpsilonWallFunction())
        {
            nutField.boundary[pi]->setValue(
                nutkWallFunction(patches[pi], yWall[pi], k.internal, nu, co.Cmu, co.kappa, co.E));
            continue;
        }

        const std::vector<scalar>& kb = k.boundary[pi]->value();
        const std::vector<scalar>& eb = epsilon.boundary[pi]->value();
        std::vector<scalar> nb(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
        {
            nb[i] = co.Cmu * kb[i] * kb[i] / eb[i];
        }
        nutField.boundary[pi]->setValue(nb);
    }
}

} // namespace kEpsilonRef
} // namespace cpu
} // namespace brae
