// _cpp REFERENCE implementation -- see kOmegaSST_cpp.cuh for the OpenFOAM provenance and refusals.
#include "kOmegaSST_cpp.cuh"
#include "nut_wall_function.cuh"
#include "near_wall_dist.cuh"
#include "pbicgstab.cuh"
#include "limitedSchemes_cpp.cuh"
#include "bound_cpp.cuh"
#include "fvc.cuh"
#include <cmath>
#include <stdexcept>
#include <cstdio>
#include <cstdlib>

namespace brae {
namespace cpu {
namespace kOmegaSST {

namespace {

namespace ls = limitedSchemes;

// gaussConvectionScheme with the SCHEME's weights. `limitedLinear <coeff>` limits on the transported
// scalar itself (NVDTVD on vf, not on magSqr as the vector form does), so vf and its gradient are the
// field being solved for.
inline scalar blend(scalar F1, scalar psi1, scalar psi2) { return F1 * (psi1 - psi2) + psi2; }

// DkEff/DomegaEff are volScalarFields in OpenFOAM -- alphaK(F1)*nut_ + nu() -- so their BOUNDARY comes
// from the operands' boundary fields, not from interpolating the cell value out to the face. nut's own
// boundary is what matters most: on a `calculated` patch OF carries the evaluated eddy viscosity, which
// at an inlet differs from the adjacent cell by more than 10x. Interpolating the cell value there is the
// same defect that put 90% of the kEpsilon epsilon residual on pitzDaily's inlet.
//
// F1 is likewise a volScalarField with its own boundary; this uses the OWNER CELL's F1 for the blend and
// nut's true boundary value, which is the part that measurably moves the residual.
SurfaceScalarField effectiveDiffusivity(
    const std::vector<scalar>&       DCell,
    const GeometricField<scalar>&    nutField,
    const std::vector<scalar>&       f1,
    scalar                           alpha1,
    scalar                           alpha2,
    scalar                           nu,
    const PrimitiveMesh&             m,
    const FvGeometry&                g,
    const std::vector<FvPatch>&      patches)
{
    static const bool dbg = std::getenv("BRAE_SST_DIFF_DEBUG") != nullptr;
    SurfaceScalarField sf = fvc::interpolate(DCell, m, g, patches);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const std::vector<scalar>& nb = nutField.boundary[pi]->value();
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const label c = patches[pi].faceCells[i];
            const scalar was = sf.boundary[pi][i];
            sf.boundary[pi][i] = blend(f1[c], alpha1, alpha2) * nb[i] + nu;
            if (dbg && i == 5)
                std::printf("      [Deff] patch %-12s face %d: interp %.6e -> nut_b %.6e (nut_b=%.3e)\n",
                            patches[pi].name.c_str(), i, was, sf.boundary[pi][i], nb[i]);
        }
    }
    return sf;
}

FvScalarMatrix divWithScheme(
    const SurfaceScalarField&        phi,
    const GeometricField<scalar>&    vf,
    bool                             limitedLinear,
    scalar                           limiterCoeff,
    const PrimitiveMesh&             m,
    const FvGeometry&                g,
    const std::vector<FvPatch>&      patches)
{
    if (!limitedLinear)
    {
        return fvm::div(phi.internal, phi.boundary, vf, m, patches);
    }

    std::vector<std::vector<scalar>> vfb(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        vfb[pi] = vf.boundary[pi]->value();
    }
    const std::vector<vector> gradVf = fvc::gaussGrad(vf.internal, vfb, m, g, patches);
    return fvm::div(phi.internal, phi.boundary, vf,
                    ls::limitedLinearWeights(phi.internal, vf, gradVf, limiterCoeff, m, g),
                    m, patches);
}



} // namespace


std::vector<scalar> S2(const std::vector<tensor>& gradU)
{
    // 2*magSqr(symm(gradU)). symm(t)_ij = 0.5*(t_ij + t_ji); magSqr sums the squares of all 9 components.
    std::vector<scalar> out(gradU.size());
    for (std::size_t c = 0; c < gradU.size(); ++c)
    {
        const tensor& t = gradU[c];
        const scalar s[9] = {
            t.xx,                 0.5*(t.xy + t.yx),    0.5*(t.xz + t.zx),
            0.5*(t.yx + t.xy),    t.yy,                 0.5*(t.yz + t.zy),
            0.5*(t.zx + t.xz),    0.5*(t.zy + t.yz),    t.zz };
        scalar m = 0;
        for (int q = 0; q < 9; ++q) m += s[q] * s[q];
        out[c] = 2.0 * m;
    }
    return out;
}


std::vector<scalar> GbyNu0(const std::vector<tensor>& gradU)
{
    // gradU && devTwoSymm(gradU); devTwoSymm(t) = (t + t^T) - (2/3)*tr(t)*I.
    std::vector<scalar> out(gradU.size());
    for (std::size_t c = 0; c < gradU.size(); ++c)
    {
        const tensor& t = gradU[c];
        const scalar tr = t.xx + t.yy + t.zz;
        const scalar d[9] = {
            2*t.xx - (2.0/3.0)*tr,  t.xy + t.yx,            t.xz + t.zx,
            t.yx + t.xy,            2*t.yy - (2.0/3.0)*tr,  t.yz + t.zy,
            t.zx + t.xz,            t.zy + t.yz,            2*t.zz - (2.0/3.0)*tr };
        const scalar gg[9] = { t.xx, t.xy, t.xz, t.yx, t.yy, t.yz, t.zx, t.zy, t.zz };
        scalar s = 0;
        for (int q = 0; q < 9; ++q) s += gg[q] * d[q];   // double inner product A && B = sum A_ij B_ij
        out[c] = s;
    }
    return out;
}


std::vector<scalar> CDkOmega(const std::vector<vector>& gradK, const std::vector<vector>& gradOmega,
                             const std::vector<scalar>& omega, const KOmegaSSTCoeffs& co)
{
    std::vector<scalar> out(omega.size());
    for (std::size_t c = 0; c < omega.size(); ++c)
        out[c] = (2.0 * co.alphaOmega2)
               * (gradK[c].x*gradOmega[c].x + gradK[c].y*gradOmega[c].y + gradK[c].z*gradOmega[c].z)
               / omega[c];
    return out;
}


std::vector<scalar> F1(const std::vector<scalar>& k, const std::vector<scalar>& omega,
                       const std::vector<scalar>& y, const std::vector<scalar>& CD,
                       scalar nu, const KOmegaSSTCoeffs& co)
{
    std::vector<scalar> out(k.size());
    for (std::size_t c = 0; c < k.size(); ++c)
    {
        const scalar CDp = std::fmax(CD[c], 1.0e-10);
        const scalar a = std::fmax((1.0/co.betaStar)*std::sqrt(k[c])/(omega[c]*y[c]),
                                   500.0*nu/(y[c]*y[c]*omega[c]));
        const scalar b = (4.0*co.alphaOmega2)*k[c]/(CDp*y[c]*y[c]);
        const scalar arg1 = std::fmin(std::fmin(a, b), 10.0);
        const scalar a4 = arg1*arg1*arg1*arg1;                 // pow4
        out[c] = std::tanh(a4);
    }
    return out;
}


std::vector<scalar> F2(const std::vector<scalar>& k, const std::vector<scalar>& omega,
                       const std::vector<scalar>& y, scalar nu, const KOmegaSSTCoeffs& co)
{
    std::vector<scalar> out(k.size());
    for (std::size_t c = 0; c < k.size(); ++c)
    {
        const scalar arg2 = std::fmin(std::fmax((2.0/co.betaStar)*std::sqrt(k[c])/(omega[c]*y[c]),
                                                500.0*nu/(y[c]*y[c]*omega[c])), 100.0);
        out[c] = std::tanh(arg2*arg2);
    }
    return out;
}


std::vector<scalar> correctNut(const std::vector<scalar>& k, const std::vector<scalar>& omega,
                               const std::vector<scalar>& F23, const std::vector<scalar>& s2,
                               const KOmegaSSTCoeffs& co)
{
    std::vector<scalar> out(k.size());
    for (std::size_t c = 0; c < k.size(); ++c)
        out[c] = co.a1*k[c] / std::fmax(co.a1*omega[c], co.b1*F23[c]*std::sqrt(s2[c]));
    return out;
}


void correct(
    const GeometricField<vector>&  U,
    GeometricField<scalar>&        k,
    GeometricField<scalar>&        omega,
    GeometricField<scalar>&        nutField,
    const SurfaceScalarField&      phi,
    const std::vector<scalar>&     y,
    scalar                         nu,
    const PrimitiveMesh&           m,
    const FvGeometry&              g,
    const std::vector<FvPatch>&    patches,
    scalar                         relaxOmega,
    scalar                         relaxK,
    scalar                         tol,
    scalar                         relTol,
    int                            maxIter,
    const KOmegaSSTCoeffs&         co,
    SSTResiduals*                  res,
    bool                           bounded,
    bool                           limitedLinear,
    scalar                         limiterCoeff)
{
    if (co.F3)
        throw std::runtime_error(
            "kOmegaSST_cpp: the F3 near-wall switch is set. kOmegaSSTBase multiplies F23 by F3 "
            "(kOmegaSSTBase.C:F23), which changes both the eddy-viscosity limiter and the production "
            "limiter. Not implemented; refusing rather than silently running with F3 off.");

    const label nC = m.nCells();
    const scalar Cmu25 = std::pow(co.betaStar, 0.25);   // kOmegaSST's Cmu IS betaStar
    std::vector<scalar>& nutF = nutField.internal;

    // ---- production, from the CURRENT nut (the previous outer iteration's correctNut) -------------
    const std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);
    const std::vector<scalar> s2  = S2(gradU);
    const std::vector<scalar> gb0 = GbyNu0(gradU);
    std::vector<scalar> G(nC);
    for (label c = 0; c < nC; ++c) G[c] = nutF[c] * gb0[c];      // RAW GbyNu0 -- the k equation's G

    const std::vector<scalar> divU = fvc::div(phi, m, g, patches);

    // ---- CDkOmega, F1, F2 ------------------------------------------------------------------------
    const std::vector<vector> gradK  = fvc::gaussGrad(k, m, g, patches);
    const std::vector<vector> gradOm = fvc::gaussGrad(omega, m, g, patches);
    const std::vector<scalar> CD  = CDkOmega(gradK, gradOm, omega.internal, co);
    const std::vector<scalar> f1  = F1(k.internal, omega.internal, y, CD, nu, co);
    const std::vector<scalar> f23 = F2(k.internal, omega.internal, y, nu, co);

    // ---- the production limiter: omega uses the LIMITED GbyNu, k uses the raw G ------------------
    // kOmegaSSTBase.C reassigns GbyNu0 = GbyNu(GbyNu0, F23, S2) AFTER G was taken from the raw value.
    // Using one for both is the easy mistake here; they are different quantities.
    std::vector<scalar> gbLim(nC);
    for (label c = 0; c < nC; ++c)
        gbLim[c] = std::fmin(gb0[c],
                             (co.c1/co.a1)*co.betaStar*omega.internal[c]
                           * std::fmax(co.a1*omega.internal[c], co.b1*f23[c]*std::sqrt(s2[c])));

    // ---- omegaWallFunction: near-wall omega + the G override -------------------------------------
    // omega = sqrt(omegaVis^2 + omegaLog^2)  -- OF's DEFAULT blender is binomial with n = 2
    // (omegaWallFunctionFvPatchScalarField.C:445, wallFunctionBlenders(dict, BINOMIAL, 2)).
    std::vector<label> nw(nC, 0);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        if (patches[pi].type == "wall")
            for (label i = 0; i < patches[pi].size; ++i) ++nw[patches[pi].faceCells[i]];

    const std::vector<std::vector<scalar>> yWall = nearWallDist(m, g, patches);   // OF turbulence.y()
    std::vector<scalar> om0(nC, 0.0), G0(nC, 0.0);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type != "wall") continue;
        const FvPatch& wp = patches[pi];
        const std::vector<scalar>& yw = yWall[pi];
        const std::vector<scalar> nutw = nutkWallFunction(wp, yw, k.internal, nu, co.betaStar, co.kappa, co.E);
        const std::vector<vector>& Uw = U.boundary[pi]->value();
        for (label i = 0; i < wp.size; ++i)
        {
            const label c = wp.faceCells[i];
            const scalar w = 1.0 / nw[c], kc = k.internal[c];
            const scalar omegaVis = 6.0*nu/(co.beta1*yw[i]*yw[i]);
            const scalar omegaLog = std::sqrt(kc)/(Cmu25*co.kappa*yw[i]);
            const scalar magGradUw = mag((Uw[i] - U.internal[c]) * wp.deltaCoeffs[i]);
            om0[c] += w * std::sqrt(omegaVis*omegaVis + omegaLog*omegaLog);
            G0[c]  += w * (nutw[i] + nu) * magGradUw * Cmu25 * std::sqrt(kc) / (co.kappa * yw[i]);
        }
    }
    std::vector<label> wallCells;
    std::vector<scalar> omVals;
    for (label c = 0; c < nC; ++c)
        if (nw[c] > 0)
        {
            G[c] = G0[c];
            omega.internal[c] = om0[c];
            wallCells.push_back(c);
            omVals.push_back(om0[c]);
        }

    // ---- omega equation --------------------------------------------------------------------------
    {
        std::vector<scalar> DomegaEff(nC);
        for (label c = 0; c < nC; ++c)
            DomegaEff[c] = blend(f1[c], co.alphaOmega1, co.alphaOmega2)*nutF[c] + nu;
        const SurfaceScalarField Df =
            effectiveDiffusivity(DomegaEff, nutField, f1, co.alphaOmega1, co.alphaOmega2,
                                 nu, m, g, patches);

        FvScalarMatrix M = divWithScheme(phi, omega, limitedLinear, limiterCoeff, m, g, patches);
        addEqual(M, fvm::laplacian(Df, omega, m, g, patches), -1.0);
        for (label c = 0; c < nC; ++c)
        {
            const scalar gam  = blend(f1[c], co.gamma1, co.gamma2);
            const scalar beta = blend(f1[c], co.beta1,  co.beta2);
            const scalar V    = g.V()[c];
            M.source[c] += gam * gbLim[c] * V;                       // == gamma*GbyNu
            // - SuSp((2/3)*gamma*divU, omega): implicit where the coefficient is positive.
            const scalar sp1 = (2.0/3.0) * gam * divU[c];
            M.diag[c]   += V * std::fmax(sp1, 0.0);
            M.source[c] -= V * std::fmin(sp1, 0.0) * omega.internal[c];
            // - Sp(beta*omega, omega)
            M.diag[c]   += beta * omega.internal[c] * V;
            // - SuSp((F1 - 1)*CDkOmega/omega, omega)
            const scalar sp2 = (f1[c] - 1.0) * CD[c] / omega.internal[c];
            M.diag[c]   += V * std::fmax(sp2, 0.0);
            M.source[c] -= V * std::fmin(sp2, 0.0) * omega.internal[c];
            // `bounded`: - Sp(fvc::div(phi), omega). Vanishes where phi is conservative, so it cannot
            // move a converged answer -- it is there to keep the transported scalar bounded on the way.
            if (bounded) M.diag[c] -= divU[c] * V;
        }
        relaxMatrix(M, omega, m, patches, relaxOmega);
        setValues(M, omega.internal, m, patches, wallCells, omVals);
        const SolverPerformance po = pbicgstab(M, omega.internal, m, patches, tol, relTol, maxIter);
        if (res) res->omega = po.initialResidual;
        if (std::getenv("BRAE_SST_DEBUG"))
            std::printf("    [omega] init=%.3e final=%.3e nIter=%d\n",
                        po.initialResidual, po.finalResidual, po.nIterations);
        static const bool dbgBound = std::getenv("BRAE_SST_BOUND_DEBUG") != nullptr;
        if (dbgBound)
        {
            scalar mn = omega.internal[0];
            int nNeg = 0;
            for (label c = 0; c < nC; ++c)
            {
                mn = std::fmin(mn, omega.internal[c]);
                if (omega.internal[c] < 0.0) ++nNeg;
            }
            std::printf("      [bound] omega min %.6e   negative cells %d\n", mn, nNeg);
        }
        // Foam::bound(omega_, omegaMin_). A FLOOR here is not the same thing and is not survivable:
        // the next iteration divides CDkOmega by this, so a floored cell contributes ~1e15. See
        // bound_cpp.cuh for the measurement.
        omega.evaluateBoundary();
        bound(omega, 1e-15, m, g, patches);
    }

    // ---- k equation ------------------------------------------------------------------------------
    {
        std::vector<scalar> DkEff(nC);
        for (label c = 0; c < nC; ++c)
            DkEff[c] = blend(f1[c], co.alphaK1, co.alphaK2)*nutF[c] + nu;
        const SurfaceScalarField Df =
            effectiveDiffusivity(DkEff, nutField, f1, co.alphaK1, co.alphaK2,
                                 nu, m, g, patches);

        FvScalarMatrix M = divWithScheme(phi, k, limitedLinear, limiterCoeff, m, g, patches);
        addEqual(M, fvm::laplacian(Df, k, m, g, patches), -1.0);
        for (label c = 0; c < nC; ++c)
        {
            const scalar V = g.V()[c];
            // == Pk(G) = min(G, (c1*betaStar)*k*omega)
            M.source[c] += std::fmin(G[c], (co.c1*co.betaStar)*k.internal[c]*omega.internal[c]) * V;
            const scalar sp = (2.0/3.0) * divU[c];                   // - SuSp((2/3)*divU, k)
            M.diag[c]   += V * std::fmax(sp, 0.0);
            M.source[c] -= V * std::fmin(sp, 0.0) * k.internal[c];
            M.diag[c]   += co.betaStar * omega.internal[c] * V;      // - Sp(epsilonByk, k)
            if (bounded) M.diag[c] -= divU[c] * V;                   // - Sp(fvc::div(phi), k)
        }
        relaxMatrix(M, k, m, patches, relaxK);
        const SolverPerformance pk = pbicgstab(M, k.internal, m, patches, tol, relTol, maxIter);
        if (res) res->k = pk.initialResidual;
        if (std::getenv("BRAE_SST_DEBUG"))
            std::printf("    [k] init=%.3e final=%.3e nIter=%d\n",
                        pk.initialResidual, pk.finalResidual, pk.nIterations);
        k.evaluateBoundary();
        bound(k, 1e-15, m, g, patches);   // Foam::bound(k_, kMin_)
    }

    // ---- correctNut ------------------------------------------------------------------------------
    // kOmegaSSTBase.C ends with correctNut(S2): S2 is the one computed at the TOP of correct() (still in
    // scope, from the old U), but F23() inside it reads the MEMBER k_ and omega_, which the two solves
    // above have just overwritten. So S2 is old and F23 is NEW. Reusing the pre-solve f23 here moved
    // omega by 1.5e-01 and nut by 1.2e-01 off OpenFOAM's converged state.
    const std::vector<scalar> f23New = F2(k.internal, omega.internal, y, nu, co);
    nutF = correctNut(k.internal, omega.internal, f23New, s2, co);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type != "wall") continue;
        nutField.boundary[pi]->setValue(
            nutkWallFunction(patches[pi], yWall[pi], k.internal, nu, co.betaStar, co.kappa, co.E));
    }

    // OpenFOAM assigns nut_ as a FIELD -- nut_ = a1*k/max(a1*omega, b1*F23*sqrt(S2)) -- and a field
    // assignment writes the BOUNDARY as well, from the boundary k and omega. correctBoundaryConditions()
    // then leaves a `calculated` patch alone, so those patches carry the evaluated value rather than the
    // adjacent cell's. Only wall patches were being written here, which the single-iteration probe could
    // never catch: it reads nut's boundary from OpenFOAM's converged file, where it is already right.
    // Running from 0/ is what exposes it -- the tutorials ship `calculated; value uniform 0` at the
    // inlet, so the inlet eddy viscosity stayed ZERO for the entire run.
    //
    // Every operand is taken at the BOUNDARY, as a field expression does: k and omega from their patch
    // values, S2 from the boundary gradU (gaussGrad's boundary correction replaces the normal component
    // with the snGrad, gb = gc + n*(snGrad - n&gc)), and F2 from those with the owner cell's y.
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        if (patches[pi].type == "wall") continue;
        if (nutField.boundary[pi]->bcCategory() != 2) continue;   // fixedValue means the case PINNED it

        const std::vector<scalar>& kb = k.boundary[pi]->value();
        const std::vector<scalar>& ob = omega.boundary[pi]->value();
        const std::vector<vector>& ub = U.boundary[pi]->value();
        std::vector<scalar> vals(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
        {
            const label c = patches[pi].faceCells[i];
            if (!(ob[i] > 0.0) || !(y[c] > 0.0))
            {
                vals[i] = nutF[c];
                continue;
            }

            const tensor& t = gradU[c];
            const scalar gc[9] = { t.xx, t.xy, t.xz, t.yx, t.yy, t.yz, t.zx, t.zy, t.zz };
            const vector& n = patches[pi].nf[i];
            const scalar nv[3] = { n.x, n.y, n.z };
            const vector sng = (ub[i] - U.internal[c]) * patches[pi].deltaCoeffs[i];
            const scalar sn[3] = { sng.x, sng.y, sng.z };

            scalar ngc[3];
            for (int j = 0; j < 3; ++j)
            {
                ngc[j] = nv[0]*gc[0*3+j] + nv[1]*gc[1*3+j] + nv[2]*gc[2*3+j];
            }
            scalar gb[9];
            for (int a = 0; a < 3; ++a)
            {
                for (int b = 0; b < 3; ++b)
                {
                    gb[a*3+b] = gc[a*3+b] + nv[a]*(sn[b] - ngc[b]);
                }
            }
            scalar ss = 0.0;
            for (int a = 0; a < 3; ++a)
            {
                for (int b = 0; b < 3; ++b)
                {
                    const scalar sab = 0.5*(gb[a*3+b] + gb[b*3+a]);
                    ss += sab*sab;
                }
            }
            const scalar S2b = 2.0*ss;

            const scalar arg2 = std::fmax(2.0*std::sqrt(std::fmax(kb[i], 0.0))/(co.betaStar*ob[i]*y[c]),
                                          500.0*nu/(y[c]*y[c]*ob[i]));
            const scalar F2b  = std::tanh(arg2*arg2);
            vals[i] = co.a1*kb[i] / std::fmax(co.a1*ob[i], co.b1*F2b*std::sqrt(std::fmax(S2b, 0.0)));
        }
        nutField.boundary[pi]->setValue(vals);
    }
}

} // namespace kOmegaSST
} // namespace cpu
} // namespace brae
