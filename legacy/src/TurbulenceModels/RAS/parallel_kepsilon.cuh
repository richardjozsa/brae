#pragma once
// cf parallel kEpsilon::correct(), the distributed turbulence step. Mirrors the serial
// kepsilon::correct (k_epsilon.cuh): matrices via the local assembly + processor interface
// (momentumDistributed for div-laplacian, Sp/SuSp added to the diagonal/source), solves via
// parallelPBiCGStab, gradients/div via the processor-aware fvc on distributed fields. The WALL
// FUNCTIONS are unchanged and entirely local (each rank's wall faces are local boundary faces).
#include "k_epsilon.cuh"
#include "parallel_simple.cuh"
#include "fv_matrix_ops.cuh"
#include <cmath>
#include <utility>
#include <vector>

namespace brae {
namespace kepsilon {

// relax: diagonal-dominance + under-relaxation (sumOff includes interface coeffs).
inline void relaxScalar(
    DistributedMatrix& L,
    FvScalarMatrix& Mloc,
    const std::vector<scalar>& psi,
    const Partition& P,
    scalar alpha)
{
    if (alpha <= 0.0) return;
    const std::vector<FvPatch>& lp = P.lp;
    const label lnC = P.nCells(), nIf = P.Lm.mesh.nInternalFaces();
    const std::vector<scalar> D0 = L.diag;
    std::vector<scalar> sumOff(lnC, 0.0);
    for (label f = 0; f < nIf; ++f)
    {
        sumOff[L.lowerAddr[f]] += std::fabs(L.upper[f]);
        sumOff[L.upperAddr[f]] += std::fabs(L.lower[f]);
    }
    std::size_t pj = 0;
    for (const FvPatch& p : lp)
    {
        if (p.type != "processor") continue;
        for (label i = 0; i < p.size; ++i)
            sumOff[p.faceCells[i]] += std::fabs(L.interfaceCoeffs[pj][i]);
        ++pj;
    }
    for (std::size_t pi = 0; pi < lp.size(); ++pi)
        for (label i = 0; i < lp[pi].size; ++i)
            L.diag[lp[pi].faceCells[i]] += std::fabs(Mloc.internalCoeffs[pi][i]);
    for (label c = 0; c < lnC; ++c)
        L.diag[c] = std::fmax(std::fabs(L.diag[c]), sumOff[c]) / alpha;
    for (std::size_t pi = 0; pi < lp.size(); ++pi)
        for (label i = 0; i < lp[pi].size; ++i)
            L.diag[lp[pi].faceCells[i]] -= Mloc.internalCoeffs[pi][i];
    for (label c = 0; c < lnC; ++c)
        Mloc.source[c] += (L.diag[c] - D0[c]) * psi[c];
}

// setValues on the distributed matrix: decouple a constrained cell's internal faces + real-boundary
// coeffs + interface coeffs (mirrors fvMatrix::setValues; the interface zeroing handles near-wall
// cells that sit on a processor boundary).
inline void setValuesDist(
    DistributedMatrix& L,
    FvScalarMatrix& Mloc,
    const Partition& P,
    const std::vector<label>& cells,
    const std::vector<scalar>& vals)
{
    const PrimitiveMesh& m = P.Lm.mesh;
    const std::vector<FvPatch>& lp = P.lp;
    const label nIf = m.nInternalFaces();
    std::vector<std::vector<std::pair<label,bool>>> adj(m.nCells());
    for (label f = 0; f < nIf; ++f)
    {
        adj[L.lowerAddr[f]].push_back({f,true});
        adj[L.upperAddr[f]].push_back({f,false});
    }
    std::vector<std::vector<std::pair<label,label>>> badj(m.nCells()), iadj(m.nCells());
    {
        std::size_t pj = 0;
        for (std::size_t pi = 0; pi < lp.size(); ++pi)
        {
            const bool proc = lp[pi].type == "processor";
            for (label i = 0; i < lp[pi].size; ++i)
            {
                if (proc)
                    iadj[lp[pi].faceCells[i]].push_back({(label)pj, i});
                else
                    badj[lp[pi].faceCells[i]].push_back({(label)pi, i});
            }
            if (proc)
                ++pj;
        }
    }
    for (std::size_t i = 0; i < cells.size(); ++i)
    {
        const label c = cells[i];
        const scalar v = vals[i];
        for (const auto& fe : adj[c])
        {
            const label f = fe.first;
            if (fe.second)
                Mloc.source[L.upperAddr[f]] -= L.lower[f]*v;
            else
                Mloc.source[L.lowerAddr[f]] -= L.upper[f]*v;
            L.upper[f] = 0.0;
            L.lower[f] = 0.0;
        }
        for (const auto& be : badj[c])
        {
            Mloc.internalCoeffs[be.first][be.second] = 0.0;
            Mloc.boundaryCoeffs[be.first][be.second] = 0.0;
        }
        for (const auto& ie : iadj[c])
            L.interfaceCoeffs[ie.first][ie.second] = 0.0;
    }
}

// Fold + relax + setValues + solve one distributed scalar transport equation (Mloc assembled
// serially on the local mesh with Sp on the diagonal + explicit source).
inline SolverPerformance solveScalarEqn(
    const Partition& P,
    FvScalarMatrix& Mloc,
    std::vector<scalar>& psi,
    const std::vector<scalar>& outPhi,
    const std::vector<scalar>& gammaF,
    std::vector<ProcessorInterface>& ifs,
    scalar relax,
    scalar tol,
    const std::vector<label>& conCells,
    const std::vector<scalar>& conVals)
{
    const std::vector<FvPatch>& lp = P.lp;
    const label lnC = P.nCells();
    DistributedMatrix L = momentumDistributed(Mloc, P.Lm, P.lg, P.lp, outPhi, gammaF, P.procDelta);
    relaxScalar(L, Mloc, psi, P, relax);
    for (std::size_t i = 0; i < conCells.size(); ++i)
        psi[conCells[i]] = conVals[i];
    if (!conCells.empty())
        setValuesDist(L, Mloc, P, conCells, conVals);
    L.diagC = L.diag;
    L.b = Mloc.source;
    for (std::size_t pi = 0; pi < lp.size(); ++pi)
        for (label i = 0; i < lp[pi].size; ++i)
        {
            L.diagC[lp[pi].faceCells[i]] += Mloc.internalCoeffs[pi][i];
            L.b[lp[pi].faceCells[i]] += Mloc.boundaryCoeffs[pi][i];
        }
    for (std::size_t i = 0; i < conCells.size(); ++i)
    {
        L.diagC[conCells[i]] = L.diag[conCells[i]];
        L.b[conCells[i]] = conVals[i] * L.diag[conCells[i]];
    }
    return parallelPBiCGStab(L, psi, ifs, P.globalNCells, tol, 0.0, 2000);
}

// kEpsilon::correct() initial residuals (for SIMPLE residualControl on k/epsilon).
struct TurbResidual { scalar k = 0.0, eps = 0.0; };

// gamma at faces (internal interpolated + processor interpolated via a coupled field).
inline std::vector<scalar> faceGamma(const Partition& P, const std::vector<scalar>& gammaCell)
{
    const PrimitiveMesh& m = P.Lm.mesh;
    const std::vector<FvPatch>& lp = P.lp;
    const SurfaceScalarField gf = fvc::interpolate(gammaCell, m, P.lg, lp);
    GeometricField<scalar> gfld = distributeFromCells(gammaCell, P);
    std::vector<scalar> g(m.nFaces(), 0.0);
    for (label f = 0; f < m.nInternalFaces(); ++f)
        g[f] = gf.internal[f];
    for (std::size_t pi = 0; pi < lp.size(); ++pi)
        if (lp[pi].type == "processor")
            for (label i = 0; i < lp[pi].size; ++i)
                g[lp[pi].start + i] = gfld.boundary[pi]->value()[i];
    return g;
}

// Parallel kEpsilon::correct(). U/k/eps/nut are distributed fields; phi the maintained flux.
inline TurbResidual parallelCorrect(
    const Partition& P,
    const GeometricField<vector>& U,
    GeometricField<scalar>& k,
    GeometricField<scalar>& eps,
    GeometricField<scalar>& nutField,
    const SurfaceScalarField& phi,
    scalar nu,
    scalar relaxEps,
    scalar relaxK,
    scalar tol,
    bool bounded = false,
    const KEpsilonCoeffs& co = {})
{
    const PrimitiveMesh& m = P.Lm.mesh;
    const FvGeometry& g = P.lg;
    const std::vector<FvPatch>& patches = P.lp;
    const label nC = m.nCells();
    const scalar kappa = co.kappa, Cmu25 = std::pow(co.Cmu, 0.25), Cmu75 = std::pow(co.Cmu, 0.75), C3 = co.C3;
    std::vector<scalar>& nutF = nutField.internal;
    std::vector<ProcessorInterface> ifs = P.ifs;
    const std::vector<scalar> outPhi = outwardFlux(P, phi);

    const std::vector<tensor> gradU = fvc::gaussGrad(U, m, g, patches);   // parallel-aware fvc
    const std::vector<scalar> gByNu = GbyNu(gradU);
    std::vector<scalar> G = production(nutF, gByNu);
    const std::vector<scalar> divU = fvc::div(phi, m, g, patches);

    // wall functions (local).
    std::vector<label> nw(nC, 0);
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        if (patches[pi].type == "wall")
            for (label i = 0; i < patches[pi].size; ++i)
                ++nw[patches[pi].faceCells[i]];
    const std::vector<std::vector<scalar>> yWall = nearWallDist(m, g, patches);
    std::vector<scalar> eps0(nC, 0.0), G0(nC, 0.0);
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
            const scalar w = 1.0/nw[c], kc = k.internal[c];
            const scalar mg = mag((Uw[i] - U.internal[c]) * wp.deltaCoeffs[i]);
            eps0[c] += w * Cmu75 * std::pow(kc, 1.5) / (kappa * y[i]);
            G0[c]   += w * (nutw[i] + nu) * mg * Cmu25 * std::sqrt(kc) / (kappa * y[i]);
        }
    }
    std::vector<label> wallCells;
    std::vector<scalar> epsVals;
    for (label c = 0; c < nC; ++c)
        if (nw[c] > 0)
        {
            G[c] = G0[c];
            eps.internal[c] = eps0[c];
            wallCells.push_back(c);
            epsVals.push_back(eps0[c]);
        }

    // epsilon equation.
    std::vector<scalar> DepsCell(nC);
    for (label c = 0; c < nC; ++c)
        DepsCell[c] = nutF[c]/co.sigmaEps + nu;
    const SurfaceScalarField Def = fvc::interpolate(DepsCell, m, g, patches);
    const std::vector<scalar> DepsF = faceGamma(P, DepsCell);
    FvScalarMatrix epsEqn = fvm::div(phi.internal, phi.boundary, eps, m, patches);
    addEqual(epsEqn, fvm::laplacian(Def, eps, m, g, patches), -1.0);
    for (label c = 0; c < nC; ++c)
    {
        const scalar sp = ((2.0/3.0)*co.C1 - C3) * divU[c];
        epsEqn.diag[c]   += (co.C2 * eps.internal[c] / k.internal[c]) * g.V()[c] + g.V()[c]*std::fmax(sp,0.0) - (bounded?divU[c]:0.0)*g.V()[c];   // last: bounded -Sp(div(phi),eps)
        epsEqn.source[c] += (co.C1 * co.Cmu * k.internal[c] * gByNu[c]) * g.V()[c] - g.V()[c]*std::fmin(sp,0.0)*eps.internal[c];
    }
    TurbResidual res;
    res.eps = solveScalarEqn(P, epsEqn, eps.internal, outPhi, DepsF, ifs, relaxEps, tol, wallCells, epsVals).initialResidual;
    for (label c = 0; c < nC; ++c)
        eps.internal[c] = std::fmax(eps.internal[c], 1e-15);
    eps.evaluateBoundary();

    // k equation.
    std::vector<scalar> DkCell(nC);
    for (label c = 0; c < nC; ++c)
        DkCell[c] = nutF[c]/co.sigmaK + nu;
    const SurfaceScalarField Dkf = fvc::interpolate(DkCell, m, g, patches);
    const std::vector<scalar> DkF = faceGamma(P, DkCell);
    FvScalarMatrix kEqnM = fvm::div(phi.internal, phi.boundary, k, m, patches);
    addEqual(kEqnM, fvm::laplacian(Dkf, k, m, g, patches), -1.0);
    for (label c = 0; c < nC; ++c)
    {
        const scalar sp = (2.0/3.0) * divU[c];
        kEqnM.diag[c]   += (eps.internal[c] / k.internal[c]) * g.V()[c] + g.V()[c]*std::fmax(sp,0.0) - (bounded?divU[c]:0.0)*g.V()[c];   // last: bounded -Sp(div(phi),k)
        kEqnM.source[c] += G[c] * g.V()[c] - g.V()[c]*std::fmin(sp,0.0)*k.internal[c];
    }
    const std::vector<label> noCon;
    const std::vector<scalar> noVal;
    res.k = solveScalarEqn(P, kEqnM, k.internal, outPhi, DkF, ifs, relaxK, tol, noCon, noVal).initialResidual;
    for (label c = 0; c < nC; ++c)
        k.internal[c] = std::fmax(k.internal[c], 1e-15);
    k.evaluateBoundary();

    // correctNut.
    for (label c = 0; c < nC; ++c)
        nutF[c] = co.Cmu * k.internal[c] * k.internal[c] / eps.internal[c];
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
        if (patches[pi].type == "wall")
            nutField.boundary[pi]->setValue(nutkWallFunction(patches[pi], yWall[pi], k.internal, nu, co.Cmu, co.kappa, co.E));
    return res;
}

} // namespace kepsilon
} // namespace brae
