#pragma once
// cf MRF SIMPLE step (serial, laminar), the standard p-U coupling with an MRF rotating zone.
// Mirrors OpenFOAM simpleFoam + MRF: phi is the RELATIVE flux, the momentum gains the Coriolis source
// (MRF.DDt(U) -> source -= V*(Omega x U)), and phiHbyA is made relative before the pressure solve:
//   UEqn = fvm::div(phi_rel,U) - laplacian(nuEff,U) - div(nuEff dev2(T(grad U)))  + Coriolis
//   solve(UEqn == -grad p); rAU=1/A; HbyA=rAU*H; phiHbyA=flux(HbyA); MRF.makeRelative(phiHbyA);
//   pEqn = laplacian(rAU,p) == div(phiHbyA); phi = phiHbyA - pEqn.flux(); p.relax(); U = HbyA - rAU grad p
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"
#include "solve_vector.cuh"
#include "pcg.cuh"
#include "mrf_zone.cuh"
#include <vector>

namespace brae {

// constrainHbyA: HbyA boundary = U's fixed value at fixedValue patches, zeroGradient/empty elsewhere.
inline GeometricField<vector> mrfConstrainHbyA(
    const std::vector<vector>& Hcell,
    const GeometricField<vector>& U,
    const std::vector<FvPatch>& fvp)
{
    GeometricField<vector> HbyA;
    HbyA.internal = Hcell;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (fvp[pi].type == "empty")
            HbyA.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(fvp[pi]));
        else if (U.boundary[pi]->fixesValue())
            HbyA.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(fvp[pi], false, vector{}, U.boundary[pi]->value()));
        else
            HbyA.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(fvp[pi]));
    }
    return HbyA;
}

inline void mrfSimpleStep(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp,
    const MRFZone& z,
    GeometricField<vector>& U,
    GeometricField<scalar>& p,
    SurfaceScalarField& phi,
    scalar nu,
    scalar relaxU,
    scalar relaxP,
    scalar tolU,
    scalar tolP)
{
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const std::vector<scalar>& V = g.V();

    // momentum: div(phi_rel,U) - laplacian(nu,U) - div(nu dev2(T(grad U))) + Coriolis.
    FvVectorMatrix UEqn = fvm::div(phi.internal, phi.boundary, U, m, fvp);
    addEqual(UEqn, fvm::laplacian(U, nu, m, g, fvp), -1.0);
    {
        const std::vector<tensor> gC = fvc::gaussGrad(U, m, g, fvp);
        const std::vector<std::vector<tensor>> gB = fvc::gradUBoundary(U, gC, m, g, fvp);
        std::vector<tensor> sC(nC);
        for (label c = 0; c < nC; ++c)
            sC[c] = nu * dev2(transpose(gC[c]));
        std::vector<std::vector<tensor>> sB(fvp.size());
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        {
            sB[pi].resize(fvp[pi].size);
            for (label i = 0; i < fvp[pi].size; ++i)
                sB[pi][i] = nu * dev2(transpose(gB[pi][i]));
        }
        const std::vector<vector> dS = fvc::div(sC, sB, m, g, fvp);
        for (label c = 0; c < nC; ++c)
            UEqn.source[c] += V[c] * dS[c];
    }
    addCoriolis(UEqn.source, U.internal, V, z);                 // MRF.DDt(U)
    relaxMatrix(UEqn, U, m, fvp, relaxU);

    // momentum predictor.
    {
        const std::vector<vector> gP = fvc::gaussGrad(p, m, g, fvp);
        FvVectorMatrix Mp = UEqn;
        for (label c = 0; c < nC; ++c)
            Mp.source[c] += (-V[c]) * gP[c];
        solveVector(Mp, U, m, fvp, tolU, 0.0, 2000);
    }

    // rAU, HbyA, phiHbyA (relative).
    const std::vector<scalar> A = matrixA(UEqn, m, g, fvp);
    std::vector<scalar> rAU(nC);
    for (label c = 0; c < nC; ++c)
        rAU[c] = 1.0 / A[c];
    const std::vector<vector> H = matrixH(UEqn, U, m, g, fvp);
    std::vector<vector> Hcell(nC);
    for (label c = 0; c < nC; ++c)
        Hcell[c] = rAU[c] * H[c];
    GeometricField<vector> HbyA = mrfConstrainHbyA(Hcell, U, fvp);
    HbyA.evaluateBoundary();
    SurfaceScalarField phiHbyA = fvc::flux(HbyA, m, g, fvp);
    makeRelative(phiHbyA, z, g, fvp);                          // MRF.makeRelative(phiHbyA)

    // pressure: laplacian(rAU,p) == div(phiHbyA).  Closed domain -> singular; symmetric eps reg.
    const SurfaceScalarField rAUf = fvc::interpolate(rAU, m, g, fvp);
    FvScalarMatrix pEqn = fvm::laplacian(rAUf, p, m, g, fvp);
    const std::vector<scalar> dphi = fvc::div(phiHbyA, m, g, fvp);
    for (label c = 0; c < nC; ++c)
        pEqn.source[c] += V[c] * dphi[c];
    scalar dsum = 0;
    for (scalar d : pEqn.diag)
        dsum += std::fabs(d);
    const scalar eps = 1e-10 * (dsum / nC);
    for (label c = 0; c < nC; ++c)
        pEqn.diag[c] -= eps;        // remove the constant nullspace symmetrically
    const std::vector<scalar> pPrev = p.internal;
    pcg(pEqn, p.internal, m, fvp, tolP, 0.0, 4000);
    p.evaluateBoundary();

    // conservative relative flux + relax + corrector.
    const SurfaceScalarField pflux = matrixFlux(pEqn, p, m, fvp);
    for (label f = 0; f < nIf; ++f)
        phi.internal[f] = phiHbyA.internal[f] - pflux.internal[f];
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i)
            phi.boundary[pi][i] = phiHbyA.boundary[pi][i] - pflux.boundary[pi][i];
    for (label c = 0; c < nC; ++c)
        p.internal[c] = pPrev[c] + relaxP * (p.internal[c] - pPrev[c]);
    p.evaluateBoundary();
    const std::vector<vector> gPn = fvc::gaussGrad(p, m, g, fvp);
    for (label c = 0; c < nC; ++c)
        U.internal[c] = HbyA.internal[c] - rAU[c] * gPn[c];
    U.evaluateBoundary();
}

} // namespace brae
