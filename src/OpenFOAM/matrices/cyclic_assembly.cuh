#pragma once
// cf cyclic Laplacian assembly + Amul. A cyclic face behaves as an internal face to a LOCAL neighbour
// cell (the owner of the matched neighbour face): coeff = deltaCoeffs * gamma * magSf, diag[own] -=
// coeff, and the Amul couples result[own] += coeff * psi[nbrCell] (transformed for vectors). Mirrors
// OpenFOAM's lduMatrix interface loop with cyclicFvPatchField::updateInterfaceMatrix.
#include "ldu_matrix.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "cyclic_interface.cuh"
#include "cf_types.cuh"
#include <vector>

namespace brae {

struct CyclicMatrix
{
    std::vector<scalar> diag, upper, lower;          // base lduMatrix (internal-face + cyclic diagonal)
    std::vector<std::vector<scalar>> ifCoeff;        // [interface][face] = gamma*magSf*deltaCoeffs (off-diag)
};

// Assemble laplacian(gamma, psi) on the cyclic mesh: internal faces + cyclic interface coupling. Real
// patches (wall/empty) are treated zeroGradient (no contribution), this is the pure interior+cyclic
// operator used to validate the cyclic coupling.
inline CyclicMatrix assembleCyclicLaplacian(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp,
    const std::vector<CyclicInterface>& cyclics,
    scalar gamma)
{
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& dc = g.deltaCoeffs();
    const std::vector<scalar>& magSf = g.magSf();

    CyclicMatrix M;
    M.diag.assign(nC, 0.0);
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
    M.ifCoeff.resize(cyclics.size());
    for (std::size_t ci = 0; ci < cyclics.size(); ++ci)
    {
        const CyclicInterface& c = cyclics[ci];
        M.ifCoeff[ci].resize(c.faceCells.size());
        for (std::size_t i = 0; i < c.faceCells.size(); ++i)
        {
            const scalar coeff = c.deltaCoeffs[i] * gamma * magSf[fvp[c.patch].start + i];
            M.ifCoeff[ci][i] = coeff;
            M.diag[c.faceCells[i]] -= coeff;
        }
    }
    return M;
}

// Amul: y = M * psi, including the cyclic interface off-diagonal coupling (local neighbour cells).
inline std::vector<scalar> cyclicAmul(
    const CyclicMatrix& M,
    const PrimitiveMesh& m,
    const std::vector<CyclicInterface>& cyclics,
    const std::vector<scalar>& psi)
{
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    std::vector<scalar> y(nC);
    for (label c = 0; c < nC; ++c) y[c] = M.diag[c] * psi[c];
    for (label f = 0; f < nIf; ++f)
    {
        y[own[f]] += M.upper[f] * psi[nei[f]];
        y[nei[f]] += M.lower[f] * psi[own[f]];
    }
    for (std::size_t ci = 0; ci < cyclics.size(); ++ci)
    {
        const CyclicInterface& c = cyclics[ci];
        for (std::size_t i = 0; i < c.faceCells.size(); ++i)
            y[c.faceCells[i]] += M.ifCoeff[ci][i] * psi[c.nbrFaceCells[i]];   // scalar: transform = identity
    }
    return y;
}

// Jacobi-preconditioned CG for the cyclic system M x = b (M must be SPD: pass the positive-definite
// form, i.e. negate the Laplacian whose diagonal is negative). Validates the cyclic interface coupling
// inside an iterative solver. Returns the iteration count.
inline int cyclicPCG(
    const CyclicMatrix& M,
    const PrimitiveMesh& m,
    const std::vector<CyclicInterface>& cyclics,
    std::vector<scalar>& x,
    const std::vector<scalar>& b,
    scalar tol,
    int maxIter)
{
    const label nC = m.nCells();
    auto dot = [&](const std::vector<scalar>& a, const std::vector<scalar>& c)
    {
        scalar s = 0;
        for (label i = 0; i < nC; ++i) s += a[i] * c[i];
        return s;
    };
    std::vector<scalar> r(nC), z(nC), p(nC), Ap;
    const std::vector<scalar> Ax = cyclicAmul(M, m, cyclics, x);
    for (label i = 0; i < nC; ++i)
    {
        r[i] = b[i] - Ax[i];
        z[i] = r[i] / M.diag[i];
    }
    p = z;
    scalar rz = dot(r, z);
    int it = 0;
    for (; it < maxIter; ++it)
    {
        Ap = cyclicAmul(M, m, cyclics, p);
        const scalar pAp = dot(p, Ap);
        if (std::fabs(pAp) < 1e-300) break;
        const scalar alpha = rz / pAp;
        for (label i = 0; i < nC; ++i)
        {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        if (std::sqrt(dot(r, r)) < tol) { ++it; break; }
        for (label i = 0; i < nC; ++i) z[i] = r[i] / M.diag[i];
        const scalar rz2 = dot(r, z), beta = rz2 / rz;
        rz = rz2;
        for (label i = 0; i < nC; ++i) p[i] = z[i] + beta * p[i];
    }
    return it;
}

} // namespace brae
