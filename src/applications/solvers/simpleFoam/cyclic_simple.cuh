#pragma once
// cf cyclic SIMPLE step (serial, laminar), the full p-U coupling with cyclic (periodic) interfaces.
// Mirrors parallelSimpleStepLaminar but the interface is LOCAL (cyclic): the base lduMatrix comes from
// fvm::div/laplacian (cyclic patches contribute zero boundary coeffs via CyclicFvPatchField), and the
// cyclic coupling is added as interface off-diagonals to the local neighbour cells:
//   momentum face i: diag[own] += max(phi_c,0) + nu*magSf*dc ;  ifCoeff = min(phi_c,0) - nu*magSf*dc
//   pressure face i: diag[own] -= rAUf*magSf*dc            ;  ifCoeff = + rAUf*magSf*dc
// A driving body force (g_x) is added to the momentum source. Validated on the periodic channel.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "cyclic_interface.cuh"
#include "cyclic_field.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"
#include <algorithm>
#include <cmath>
#include <vector>

namespace brae {

struct CyclicLdu {
    std::vector<scalar> diag, upper, lower;            // base lduMatrix incl. cyclic diagonal additions
    std::vector<std::vector<scalar>> ifCoeff;          // [interface][face]: result[own] += ifCoeff*psi[nbr]
};

inline std::vector<scalar> cyclicLduAmul(const CyclicLdu& L, const PrimitiveMesh& m,
                                         const std::vector<CyclicInterface>& cyclics, const std::vector<scalar>& psi) {
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner(); const std::vector<label>& nei = m.neighbour();
    std::vector<scalar> y(nC);
    for (label c = 0; c < nC; ++c) y[c] = L.diag[c] * psi[c];
    for (label f = 0; f < nIf; ++f) { y[own[f]] += L.upper[f] * psi[nei[f]]; y[nei[f]] += L.lower[f] * psi[own[f]]; }
    for (std::size_t ci = 0; ci < cyclics.size(); ++ci) { const CyclicInterface& c = cyclics[ci];
        for (std::size_t i = 0; i < c.faceCells.size(); ++i) y[c.faceCells[i]] += L.ifCoeff[ci][i] * psi[c.nbrFaceCells[i]]; }
    return y;
}

// Jacobi-preconditioned BiCGStab for the (possibly non-symmetric) cyclic system L x = b.
inline int cyclicBiCGStab(const CyclicLdu& L, const PrimitiveMesh& m, const std::vector<CyclicInterface>& cyclics,
                          std::vector<scalar>& x, const std::vector<scalar>& b, scalar tol, int maxIter) {
    const label nC = m.nCells();
    auto amul = [&](const std::vector<scalar>& v) { return cyclicLduAmul(L, m, cyclics, v); };
    auto dot  = [&](const std::vector<scalar>& a, const std::vector<scalar>& c) { scalar s = 0; for (label i = 0; i < nC; ++i) s += a[i]*c[i]; return s; };
    std::vector<scalar> r = b; { const std::vector<scalar> Ax = amul(x); for (label i = 0; i < nC; ++i) r[i] -= Ax[i]; }
    std::vector<scalar> r0 = r, p(nC, 0), v(nC, 0), s(nC), t, y(nC), z(nC);
    scalar rho = 1, alpha = 1, omega = 1;
    for (int it = 0; it < maxIter; ++it) {
        const scalar rho1 = dot(r0, r); if (std::fabs(rho1) < 1e-300) break;
        const scalar beta = (rho1/rho)*(alpha/omega); rho = rho1;
        for (label i = 0; i < nC; ++i) p[i] = r[i] + beta*(p[i] - omega*v[i]);
        for (label i = 0; i < nC; ++i) y[i] = p[i]/L.diag[i];
        v = amul(y); alpha = rho1/dot(r0, v);
        for (label i = 0; i < nC; ++i) s[i] = r[i] - alpha*v[i];
        if (std::sqrt(dot(s, s)) < tol) { for (label i = 0; i < nC; ++i) x[i] += alpha*y[i]; return it+1; }
        for (label i = 0; i < nC; ++i) z[i] = s[i]/L.diag[i];
        t = amul(z); const scalar tt = dot(t, t); omega = tt > 1e-300 ? dot(t, s)/tt : 0.0;
        for (label i = 0; i < nC; ++i) { x[i] += alpha*y[i] + omega*z[i]; r[i] = s[i] - omega*t[i]; }
        if (std::sqrt(dot(r, r)) < tol) return it+1;
    }
    return maxIter;
}

// Cyclic-aware fvMatrix::relax, sumOff includes the |cyclic interface coeff|.
inline void cyclicRelax(CyclicLdu& L, std::vector<vector>& source, const FvVectorMatrix& M0,
                        const GeometricField<vector>& U, const PrimitiveMesh& m, const FvGeometry& g,
                        const std::vector<FvPatch>& fvp, const std::vector<CyclicInterface>& cyclics, scalar alpha) {
    if (alpha <= 0.0) return;
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner(); const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar> D0 = L.diag;
    std::vector<scalar> sumOff(nC, 0.0);
    for (label f = 0; f < nIf; ++f) { sumOff[own[f]] += std::fabs(L.upper[f]); sumOff[nei[f]] += std::fabs(L.lower[f]); }
    for (std::size_t ci = 0; ci < cyclics.size(); ++ci) for (std::size_t i = 0; i < cyclics[ci].faceCells.size(); ++i)
        sumOff[cyclics[ci].faceCells[i]] += std::fabs(L.ifCoeff[ci][i]);
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) for (label i = 0; i < fvp[pi].size; ++i)
        L.diag[fvp[pi].faceCells[i]] += cmptMagMax(M0.internalCoeffs[pi][i]);
    for (label c = 0; c < nC; ++c) L.diag[c] = std::fmax(std::fabs(L.diag[c]), sumOff[c]) / alpha;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) for (label i = 0; i < fvp[pi].size; ++i)
        L.diag[fvp[pi].faceCells[i]] -= cmptMin(M0.internalCoeffs[pi][i]);
    for (label c = 0; c < nC; ++c) source[c] += (L.diag[c] - D0[c]) * U.internal[c];
}

// One cyclic SIMPLE iteration on the periodic channel. Drives the flow with body force g=(gx,0,0).
inline void cyclicSimpleStep(const PrimitiveMesh& m, const FvGeometry& g, const std::vector<FvPatch>& fvp,
                             const std::vector<CyclicInterface>& cyclics,
                             GeometricField<vector>& U, GeometricField<scalar>& p, SurfaceScalarField& phi,
                             scalar nu, scalar gx, scalar relaxU, scalar relaxP, scalar tolU, scalar tolP) {
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const std::vector<scalar>& V = g.V(); const std::vector<scalar>& magSf = g.magSf();
    auto pcmp = [](const vector& v, int k) { return k == 0 ? v.x : (k == 1 ? v.y : v.z); };
    auto pset = [](vector& v, int k, scalar s) { if (k==0) v.x=s; else if (k==1) v.y=s; else v.z=s; };

    // --- momentum operator: div(phi,U) - laplacian(nu,U), + body force ---
    FvVectorMatrix M0 = fvm::div(phi.internal, phi.boundary, U, m, fvp);
    addEqual(M0, fvm::laplacian(U, nu, m, g, fvp), -1.0);
    for (label c = 0; c < nC; ++c) M0.source[c].x += gx * V[c];

    CyclicLdu Lm; Lm.diag = M0.diag; Lm.upper = M0.upper; Lm.lower = M0.lower; Lm.ifCoeff.resize(cyclics.size());
    for (std::size_t ci = 0; ci < cyclics.size(); ++ci) { const CyclicInterface& c = cyclics[ci];
        Lm.ifCoeff[ci].resize(c.faceCells.size());
        for (std::size_t i = 0; i < c.faceCells.size(); ++i) {
            const scalar phi_c = phi.boundary[c.patch][i];
            const scalar cl = nu * magSf[fvp[c.patch].start + i] * c.deltaCoeffs[i];
            Lm.diag[c.faceCells[i]] += std::fmax(phi_c, 0.0) + cl;
            Lm.ifCoeff[ci][i]        = std::fmin(phi_c, 0.0) - cl;
        }
    }
    std::vector<vector> src = M0.source;
    cyclicRelax(Lm, src, M0, U, m, g, fvp, cyclics, relaxU);

    // diagC = relaxed diag + real-patch internalCoeffs (cmptAv); used for A() and the component solve.
    std::vector<scalar> diagC = Lm.diag;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) for (label i = 0; i < fvp[pi].size; ++i)
        diagC[fvp[pi].faceCells[i]] += cmptAv(M0.internalCoeffs[pi][i]);

    // momentum predictor: solve (Lm == src - V*grad(p)) per component.
    const std::vector<vector> gradP = fvc::gaussGrad(p, m, g, fvp);
    for (int k = 0; k < 3; ++k) {
        CyclicLdu Lk = Lm; Lk.diag = diagC;                      // solve uses the boundary-folded diagonal
        std::vector<scalar> b(nC), xk(nC);
        for (label c = 0; c < nC; ++c) { b[c] = pcmp(src[c], k) - V[c]*pcmp(gradP[c], k); xk[c] = pcmp(U.internal[c], k); }
        for (std::size_t pi = 0; pi < fvp.size(); ++pi) for (label i = 0; i < fvp[pi].size; ++i)
            b[fvp[pi].faceCells[i]] += pcmp(M0.boundaryCoeffs[pi][i], k);
        cyclicBiCGStab(Lk, m, cyclics, xk, b, tolU, 2000);
        for (label c = 0; c < nC; ++c) pset(U.internal[c], k, xk[c]);
    }
    U.evaluateBoundary();

    // --- A, rAU, H, HbyA ---
    std::vector<scalar> A(nC), rAU(nC);
    for (label c = 0; c < nC; ++c) { A[c] = diagC[c]/V[c]; rAU[c] = 1.0/A[c]; }
    GeometricField<vector> HbyA = buildCyclicField<vector>(std::vector<vector>(nC), fvp, cyclics, /*wallNoSlip*/true);
    for (int k = 0; k < 3; ++k) {
        CyclicLdu Lk = Lm; Lk.diag = diagC;
        std::vector<scalar> Uk(nC); for (label c = 0; c < nC; ++c) Uk[c] = pcmp(U.internal[c], k);
        const std::vector<scalar> Apsi = cyclicLduAmul(Lk, m, cyclics, Uk);
        std::vector<scalar> Hk(nC);
        for (label c = 0; c < nC; ++c) Hk[c] = diagC[c]*Uk[c] - Apsi[c] + pcmp(src[c], k);   // H = (D - L)psi + source
        for (std::size_t pi = 0; pi < fvp.size(); ++pi) for (label i = 0; i < fvp[pi].size; ++i) {
            const label fc = fvp[pi].faceCells[i]; const vector ic = M0.internalCoeffs[pi][i];
            Hk[fc] += (cmptAv(ic) - pcmp(ic, k))*Uk[fc] + pcmp(M0.boundaryCoeffs[pi][i], k); }
        for (label c = 0; c < nC; ++c) pset(HbyA.internal[c], k, rAU[c]*Hk[c]/V[c]);
    }
    HbyA.evaluateBoundary();
    SurfaceScalarField phiHbyA = fvc::flux(HbyA, m, g, fvp);

    // --- pressure: laplacian(rAU,p) == div(phiHbyA) ---
    const SurfaceScalarField rAUf = fvc::interpolate(rAU, m, g, fvp);
    GeometricField<scalar> rAUfld = buildCyclicField<scalar>(rAU, fvp, cyclics);
    FvScalarMatrix Mp = fvm::laplacian(rAUf, p, m, g, fvp);
    const std::vector<scalar> dphi = fvc::div(phiHbyA, m, g, fvp);
    for (label c = 0; c < nC; ++c) Mp.source[c] += V[c]*dphi[c];

    CyclicLdu Lp; Lp.diag = Mp.diag; Lp.upper = Mp.upper; Lp.lower = Mp.lower; Lp.ifCoeff.resize(cyclics.size());
    std::vector<scalar> pcoeff_store;                                  // cyclic pressure face coeffs (for flux)
    std::vector<std::vector<scalar>> pcoeff(cyclics.size());
    for (std::size_t ci = 0; ci < cyclics.size(); ++ci) { const CyclicInterface& c = cyclics[ci];
        Lp.ifCoeff[ci].resize(c.faceCells.size()); pcoeff[ci].resize(c.faceCells.size());
        for (std::size_t i = 0; i < c.faceCells.size(); ++i) {
            const scalar rf = rAUfld.boundary[c.patch]->value()[i];    // coupled rAU at the cyclic face
            const scalar coeff = rf * magSf[fvp[c.patch].start + i] * c.deltaCoeffs[i];
            Lp.diag[c.faceCells[i]] -= coeff; Lp.ifCoeff[ci][i] = coeff; pcoeff[ci][i] = coeff;
        }
    }
    Lp.diag = Lp.diag;  // (real pressure patches are zeroGradient here -> no internalCoeffs)
    std::vector<scalar> b(nC); for (label c = 0; c < nC; ++c) b[c] = Mp.source[c];
    // negate to SPD form and solve.
    CyclicLdu Lpn = Lp; for (auto& d : Lpn.diag) d = -d; for (auto& u : Lpn.upper) u = -u; for (auto& l : Lpn.lower) l = -l;
    for (auto& vv : Lpn.ifCoeff) for (auto& e : vv) e = -e;
    for (auto& bb : b) bb = -bb;
    // Pure-Neumann periodic pressure is singular (constant nullspace). The RHS div(phiHbyA) is zero-mean
    // (cyclic fluxes cancel pairwise, walls carry none), so a tiny SYMMETRIC diagonal shift removes the
    // singularity without exciting the nullspace or breaking x-periodicity (unlike a single-cell pin).
    scalar dsum = 0; for (scalar d : Lpn.diag) dsum += std::fabs(d);
    const scalar eps = 1e-10 * (dsum / nC);
    for (auto& d : Lpn.diag) d += eps;
    const std::vector<scalar> pPrev = p.internal;
    cyclicBiCGStab(Lpn, m, cyclics, p.internal, b, tolP, 4000);
    p.evaluateBoundary();

    // conservative flux: phi = phiHbyA - pEqn.flux()
    for (label f = 0; f < nIf; ++f) phi.internal[f] = phiHbyA.internal[f] - (Mp.upper[f]*p.internal[m.neighbour()[f]] - Mp.lower[f]*p.internal[m.owner()[f]]);
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) phi.boundary[pi].assign(fvp[pi].size, 0.0);
    for (std::size_t ci = 0; ci < cyclics.size(); ++ci) { const CyclicInterface& c = cyclics[ci];
        for (std::size_t i = 0; i < c.faceCells.size(); ++i) {
            const scalar flux = pcoeff[ci][i]*(p.internal[c.nbrFaceCells[i]] - p.internal[c.faceCells[i]]);
            phi.boundary[c.patch][i] = phiHbyA.boundary[c.patch][i] - flux;
        }
    }
    // p relaxation, momentum corrector.
    for (label c = 0; c < nC; ++c) p.internal[c] = pPrev[c] + relaxP*(p.internal[c] - pPrev[c]);
    p.evaluateBoundary();
    const std::vector<vector> gPn = fvc::gaussGrad(p, m, g, fvp);
    for (label c = 0; c < nC; ++c) U.internal[c] = HbyA.internal[c] - rAU[c]*gPn[c];
    U.evaluateBoundary();
}

} // namespace brae
