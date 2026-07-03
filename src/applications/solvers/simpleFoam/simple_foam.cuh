#pragma once
// brae::simpleStep, one SIMPLE iteration (laminar, simplified momentum div(phi,U)-laplacian(nu,U),
// Gauss upwind), transcribed from OpenFOAM simpleFoam UEqn.H/pEqn.H:
//   UEqn = div(phi,U) - laplacian(nu,U); UEqn.relax();
//   solve(UEqn == -grad(p));                          (momentum predictor)
//   rAU=1/A; HbyA=constrainHbyA(rAU*H); phiHbyA=flux(HbyA);
//   pEqn = laplacian(rAU,p) == div(phiHbyA); solve; phi = phiHbyA - pEqn.flux();
//   p.relax(); U = HbyA - rAU*grad(p);
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"
#include "solve_vector.cuh"
#include "gamg.cuh"
#include "k_epsilon.cuh"
#include <vector>

namespace brae {

struct SimpleControls {
    scalar nu       = 1e-5;
    scalar relaxU   = 0.7;
    scalar relaxP   = 0.3;
    scalar tolU     = 1e-10, relTolU = 0.0;
    scalar tolP     = 1e-10, relTolP = 0.0;
    int    maxIter  = 2000;
    bool   bounded  = false;   // "bounded Gauss upwind": add -Sp(div(phi),U). Set from fvSchemes; default off (plain).
};

struct StepResidual { scalar p = 0.0; scalar Ux = 0.0; };

// One SIMPLE iteration. nut == nullptr -> laminar (nuEff = nu); otherwise turbulent
// (nuEff = nu + nut). The momentum is the full incompressible stress, mirroring OpenFOAM
// simpleFoam UEqn.H == turbulence->divDevReff(U):
//   fvm::div(phi,U) - fvm::laplacian(nuEff,U) - fvc::div(nuEff*dev2(T(grad U))).
inline StepResidual simpleStep(GeometricField<vector>& U, GeometricField<scalar>& p, SurfaceScalarField& phi,
                               const FieldData<vector>& Udata,
                               const PrimitiveMesh& m, const FvGeometry& g, const std::vector<FvPatch>& patches,
                               const SimpleControls& ctl,
                               const GeometricField<scalar>* nut = nullptr) {
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<scalar>& wt = g.weights();
    StepResidual res;

    // nuEff = nu + nut (cell + boundary).
    std::vector<scalar> nuEffC(nC, ctl.nu);
    std::vector<std::vector<scalar>> nuEffB(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) nuEffB[pi].assign(patches[pi].size, ctl.nu);
    if (nut) {
        for (label c = 0; c < nC; ++c) nuEffC[c] = ctl.nu + nut->internal[c];
        for (std::size_t pi = 0; pi < patches.size(); ++pi) {
            const std::vector<scalar>& nv = nut->boundary[pi]->value();
            for (label i = 0; i < patches[pi].size; ++i) nuEffB[pi][i] = ctl.nu + nv[i];
        }
    }

    // --- Momentum equation: div(phi,U) - laplacian(nuEff,U) - div(nuEff*dev2(T(grad U)))
    FvVectorMatrix UEqn = fvm::div(phi.internal, phi.boundary, U, m, patches);
    SurfaceScalarField nuEff_f;                                  // nuEff at faces (boundary = nuEffB)
    nuEff_f.internal.resize(nIf);
    for (label f = 0; f < nIf; ++f) nuEff_f.internal[f] = wt[f] * nuEffC[own[f]] + (1.0 - wt[f]) * nuEffC[nei[f]];
    nuEff_f.boundary = nuEffB;
    addEqual(UEqn, fvm::laplacian(nuEff_f, U, m, g, patches), -1.0);
    // explicit transpose stress: source += V * fvc::div(nuEff*dev2(T(grad U)))
    const std::vector<tensor> gradC = fvc::gaussGrad(U, m, g, patches);
    const std::vector<std::vector<tensor>> gradB = fvc::gradUBoundary(U, gradC, m, g, patches);
    std::vector<tensor> sigC(nC);
    for (label c = 0; c < nC; ++c) sigC[c] = nuEffC[c] * dev2(transpose(gradC[c]));
    std::vector<std::vector<tensor>> sigB(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) {
        sigB[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i) sigB[pi][i] = nuEffB[pi][i] * dev2(transpose(gradB[pi][i]));
    }
    const std::vector<vector> divSig = fvc::div(sigC, sigB, m, g, patches);
    for (label c = 0; c < nC; ++c) UEqn.source[c] += g.V()[c] * divSig[c];
    // bounded Gauss upwind: - fvm::Sp(fvc::div(phi), U). Diagonal gets -V*div(phi), stabilises the transient
    // (e.g. a rest-start inlet cell where div(phi)!=0 at low nu); vanishes at convergence (div(phi)->0).
    if (ctl.bounded) { const std::vector<scalar> divPhi = fvc::div(phi, m, g, patches);
        for (label c = 0; c < nC; ++c) UEqn.diag[c] -= divPhi[c] * g.V()[c]; }
    relaxMatrix(UEqn, U, m, patches, ctl.relaxU);

    // Momentum predictor: solve(UEqn == -grad(p)) on a copy (UEqn unchanged for A()/H()).
    {
        const std::vector<vector> gradP = fvc::gaussGrad(p, m, g, patches);
        FvVectorMatrix Mp = UEqn;
        for (label c = 0; c < nC; ++c) Mp.source[c] += (-g.V()[c]) * gradP[c];
        const SolverPerformance up = solveVector(Mp, U, m, patches, ctl.tolU, ctl.relTolU, ctl.maxIter);
        res.Ux = up.initialResidual;
    }

    // --- Pressure equation
    const std::vector<scalar> A = matrixA(UEqn, m, g, patches);
    std::vector<scalar> rAU(nC);
    for (label c = 0; c < nC; ++c) rAU[c] = 1.0 / A[c];
    const std::vector<vector> H = matrixH(UEqn, U, m, g, patches);

    GeometricField<vector> HbyA = buildField<vector>(Udata, patches, nC);   // constrainHbyA via U's BCs
    for (label c = 0; c < nC; ++c) HbyA.internal[c] = rAU[c] * H[c];
    HbyA.evaluateBoundary();
    const SurfaceScalarField phiHbyA = fvc::flux(HbyA, m, g, patches);

    const SurfaceScalarField gammaf = fvc::interpolate(rAU, m, g, patches);
    FvScalarMatrix pEqn = fvm::laplacian(gammaf, p, m, g, patches);
    const std::vector<scalar> divPhiHbyA = fvc::div(phiHbyA, m, g, patches);
    for (label c = 0; c < nC; ++c) pEqn.source[c] += g.V()[c] * divPhiHbyA[c];

    const std::vector<scalar> pPrev = p.internal;
    const SolverPerformance pp = gamg(pEqn, p.internal, m, g, patches, ctl.tolP, ctl.relTolP, ctl.maxIter);
    res.p = pp.initialResidual;

    // Conservative flux: phi = phiHbyA - pEqn.flux()
    const SurfaceScalarField pflux = matrixFlux(pEqn, p, m, patches);
    phi.internal.resize(nIf);
    for (label f = 0; f < nIf; ++f) phi.internal[f] = phiHbyA.internal[f] - pflux.internal[f];
    phi.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) {
        phi.boundary[pi].resize(patches[pi].size);
        for (label i = 0; i < patches[pi].size; ++i)
            phi.boundary[pi][i] = phiHbyA.boundary[pi][i] - pflux.boundary[pi][i];
    }

    // p relaxation, then momentum corrector.
    for (label c = 0; c < nC; ++c) p.internal[c] = pPrev[c] + ctl.relaxP * (p.internal[c] - pPrev[c]);
    p.evaluateBoundary();

    const std::vector<vector> gradPnew = fvc::gaussGrad(p, m, g, patches);
    for (label c = 0; c < nC; ++c) U.internal[c] = HbyA.internal[c] - rAU[c] * gradPnew[c];
    U.evaluateBoundary();

    return res;
}

// The SIMPLE loop: iterate simpleStep until both residuals fall below resControl. Returns the
// number of outer iterations performed. Tracks the worst residual seen (for diagnostics).
inline int runSimple(GeometricField<vector>& U, GeometricField<scalar>& p, SurfaceScalarField& phi,
                     const FieldData<vector>& Udata,
                     const PrimitiveMesh& m, const FvGeometry& g, const std::vector<FvPatch>& patches,
                     const SimpleControls& ctl, int maxOuter, scalar resControl,
                     scalar* firstP = nullptr, scalar* lastP = nullptr) {
    int iter = 0;
    for (; iter < maxOuter; ++iter) {
        const StepResidual r = simpleStep(U, p, phi, Udata, m, g, patches, ctl);
        if (iter == 0 && firstP) *firstP = r.p;
        if (lastP) *lastP = r.p;
        if (r.p < resControl && r.Ux < resControl) { ++iter; break; }
    }
    return iter;
}

struct TurbControls { scalar relaxK = 0.7; scalar relaxEps = 0.7; scalar tol = 1e-8; scalar relTol = 0.0; int maxIter = 1000; };

// Turbulent SIMPLE loop: each outer iteration runs the momentum+pressure step (nuEff = nu + nut)
// then turbulence->correct(), mirroring OpenFOAM simpleFoam's loop body (UEqn.H, pEqn.H, then
// turbulence->correct()). k/epsilon/nut are carried as fields with boundaries between iterations.
inline int runSimpleTurbulent(GeometricField<vector>& U, GeometricField<scalar>& p, SurfaceScalarField& phi,
                              GeometricField<scalar>& k, GeometricField<scalar>& eps, GeometricField<scalar>& nut,
                              const FieldData<vector>& Udata,
                              const PrimitiveMesh& m, const FvGeometry& g, const std::vector<FvPatch>& patches,
                              const SimpleControls& ctl, const TurbControls& tctl, int maxOuter, scalar resControl,
                              scalar* firstP = nullptr, scalar* lastP = nullptr) {
    int iter = 0;
    for (; iter < maxOuter; ++iter) {
        const StepResidual r = simpleStep(U, p, phi, Udata, m, g, patches, ctl, &nut);
        kepsilon::correct(U, k, eps, nut, phi, ctl.nu, m, g, patches,
                          tctl.relaxEps, tctl.relaxK, tctl.tol, tctl.relTol, tctl.maxIter);
        if (iter == 0 && firstP) *firstP = r.p;
        if (lastP) *lastP = r.p;
        if (r.p < resControl && r.Ux < resControl) { ++iter; break; }
    }
    return iter;
}

} // namespace brae
