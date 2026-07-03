#pragma once
// cf parallel SIMPLE, the distributed simpleFoam step, assembled from the validated parallel ops.
// A Partition bundles one rank's local mesh/geometry/patches + the interface addressing; the step
// does momentum predictor + pEqn + corrector on the local fields, maintaining the conservative phi
// (internal + processor faces) across iterations. Laminar/simplified momentum div(phi,U)-laplacian
// (nuEff,U); pass a per-cell nuEff (= nu for laminar, nu+nut for turbulent).
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "scotch_decomposition.cuh"
#include "local_mesh.cuh"
#include "local_assembly.cuh"
#include "field_distribute.cuh"
#include "parallel_amul.cuh"
#include "parallel_pcg.cuh"
#include "parallel_pbicgstab.cuh"
#include "parallel_matrix_ops.cuh"
#include "cf_pstream.cuh"
#include <cmath>
#include <memory>
#include <vector>

namespace brae {

inline scalar pcmp(const vector& v, int c) { return c == 0 ? v.x : (c == 1 ? v.y : v.z); }
inline void   pset(vector& v, int c, scalar s) { if (c == 0) v.x = s; else if (c == 1) v.y = s; else v.z = s; }

// One rank's decomposed view. globalNCells is needed for the parallel solvers' normFactor.
struct Partition {
    int rank = 0; label globalNCells = 0;
    std::vector<PatchInfo> gPatches;
    LocalMesh Lm; FvGeometry lg; std::vector<FvPatch> lp;
    std::vector<std::vector<scalar>> procDelta, procW;
    std::vector<ProcessorInterface> ifs;

    Partition(const PrimitiveMesh& gm, const std::vector<label>& cellToPart, int myRank)
        : rank(myRank), globalNCells(gm.nCells()), gPatches(gm.patches()), Lm(buildLocalMesh(gm, cellToPart, myRank)) {
        lg.build(Lm.mesh);
        lp = buildPatches(Lm.mesh, lg);
        procDelta = computeProcDeltaCoeffs(Lm, lg, lp);
        procW     = computeProcWeights(Lm, lg, lp);
        for (std::size_t j = 0; j < Lm.procNbr.size(); ++j) ifs.emplace_back(myRank, Lm.procNbr[j], Lm.procFaceCells[j]);
    }
    label nCells() const { return Lm.mesh.nCells(); }
};

// Build a coupled field from local cell values: real patches zeroGradient (cell value), processor
// patches interpolating (w*local + (1-w)*halo) so value() gives the coupled face value. Works for
// scalar (gamma fields) and tensor (the divDevReff stress sigma).
template <typename T>
inline GeometricField<T> distributeFromCells(const std::vector<T>& cells, const Partition& P) {
    GeometricField<T> gf; gf.internal = cells;
    std::size_t pj = 0;
    for (const FvPatch& p : P.lp) {
        if (p.type == "processor") { auto pf = std::make_unique<ProcessorFvPatchField<T>>(p, P.Lm.procNbr[pj], 0); pf->setWeights(P.procW[pj]); gf.boundary.push_back(std::move(pf)); ++pj; }
        else gf.boundary.push_back(std::make_unique<ZeroGradientPatchField<T>>(p));
    }
    gf.evaluateBoundary();
    return gf;
}

// Explicit divDevReff stress term fvc::div(nuEff*dev2(T(grad U))) on the distributed mesh. The cell
// gradient is the processor-aware gaussGrad; processor FACES use the INTERPOLATED cell sigma (coupled
// face, via a distributed tensor field), real faces use the snGrad-corrected boundary gradient.
inline std::vector<vector> parallelDivDevReff(const Partition& P, const GeometricField<vector>& U,
                                              const std::vector<scalar>& nuEffCell,
                                              const std::vector<std::vector<scalar>>& nuEffBnd) {
    const PrimitiveMesh& m = P.Lm.mesh; const FvGeometry& g = P.lg; const std::vector<FvPatch>& lp = P.lp;
    const std::vector<tensor> gradC = fvc::gaussGrad(U, m, g, lp);
    std::vector<tensor> sigC(m.nCells());
    for (label c = 0; c < m.nCells(); ++c) sigC[c] = nuEffCell[c] * dev2(transpose(gradC[c]));
    GeometricField<tensor> sigFld = distributeFromCells<tensor>(sigC, P);            // processor-interpolated sigma
    const std::vector<std::vector<tensor>> gradB = fvc::gradUBoundary(U, gradC, m, g, lp);
    std::vector<std::vector<tensor>> sigB(lp.size());
    for (std::size_t pi = 0; pi < lp.size(); ++pi) {
        sigB[pi].resize(lp[pi].size);
        if (lp[pi].type == "processor") for (label i = 0; i < lp[pi].size; ++i) sigB[pi][i] = sigFld.boundary[pi]->value()[i];
        else                            for (label i = 0; i < lp[pi].size; ++i) sigB[pi][i] = nuEffBnd[pi][i] * dev2(transpose(gradB[pi][i]));
    }
    return fvc::div(sigC, sigB, m, g, lp);
}

// outward flux per LOCAL face from the maintained phi (internal + boundary). Processor faces carry
// the conservative interface flux; real-boundary entries are the patch flux (used by fvm::div).
inline std::vector<scalar> outwardFlux(const Partition& P, const SurfaceScalarField& phi) {
    std::vector<scalar> o(P.Lm.mesh.nFaces(), 0.0);
    for (label f = 0; f < P.Lm.mesh.nInternalFaces(); ++f) o[f] = phi.internal[f];
    for (std::size_t pi = 0; pi < P.lp.size(); ++pi) for (label i = 0; i < P.lp[pi].size; ++i) o[P.lp[pi].start + i] = phi.boundary[pi][i];
    return o;
}

// SIMPLE outer-iteration initial residuals (for residualControl). Ux = max over U components.
struct ParStepResidual { scalar p = 0.0, Ux = 0.0; };

// One laminar/simplified SIMPLE iteration on the distributed fields; updates U, p, phi in place.
// Returns the momentum-predictor and pressure initial residuals (OF SIMPLE convergence measure).
inline ParStepResidual parallelSimpleStepLaminar(const Partition& P, GeometricField<vector>& U, GeometricField<scalar>& p,
                                      SurfaceScalarField& phi, const std::vector<scalar>& nuEffCell,
                                      const std::vector<std::vector<scalar>>& nuEffBnd,
                                      const FieldData<vector>& UbcShape, scalar relaxU, scalar relaxP,
                                      scalar tolU, scalar tolP, bool bounded = false) {
    ParStepResidual res;
    const PrimitiveMesh& m = P.Lm.mesh; const FvGeometry& lg = P.lg; const std::vector<FvPatch>& lp = P.lp;
    const label lnC = m.nCells(), nIf = m.nInternalFaces();
    std::vector<ProcessorInterface> ifs = P.ifs;   // local copy (mutable exchange buffers)

    // validComponents (fvMesh::validComponents): an empty-patch direction is not solved, so it must
    // not pollute the U residualControl measure. Mirror OF, exclude empty directions from res.Ux.
    bool validC[3] = {true, true, true};
    for (std::size_t pi = 0; pi < lp.size(); ++pi) if (lp[pi].type == "empty" && lp[pi].size > 0) {
        scalar ax = 0, ay = 0, az = 0;
        for (label i = 0; i < lp[pi].size; ++i) { const vector& n = lg.Sf()[lp[pi].start + i]; ax += std::fabs(n.x); ay += std::fabs(n.y); az += std::fabs(n.z); }
        validC[(ax >= ay && ax >= az) ? 0 : (ay >= az ? 1 : 2)] = false;
    }

    // nuEff at faces: interpolate internal; real boundary = nuEffBnd (= nu+nut_b); processor = interpolated.
    GeometricField<scalar> nuEffFld = distributeFromCells<scalar>(nuEffCell, P);
    SurfaceScalarField nuEfff0 = fvc::interpolate(nuEffCell, m, lg, lp);
    nuEfff0.boundary = nuEffBnd;
    std::vector<scalar> nuF(m.nFaces(), 0.0);
    for (label f = 0; f < nIf; ++f) nuF[f] = nuEfff0.internal[f];
    for (std::size_t pi = 0; pi < lp.size(); ++pi) if (lp[pi].type == "processor") for (label i = 0; i < lp[pi].size; ++i) nuF[lp[pi].start + i] = nuEffFld.boundary[pi]->value()[i];

    const std::vector<scalar> outPhi = outwardFlux(P, phi);

    // momentum = div(phi,U) - laplacian(nuEff,U) - div(nuEff*dev2(T(grad U)))  (full divDevReff).
    FvVectorMatrix Ml = fvm::div(phi.internal, phi.boundary, U, m, lp);
    addEqual(Ml, fvm::laplacian(nuEfff0, U, m, lg, lp), -1.0);
    const std::vector<vector> divSig = parallelDivDevReff(P, U, nuEffCell, nuEffBnd);
    for (label c = 0; c < lnC; ++c) Ml.source[c] += lg.V()[c] * divSig[c];
    // bounded Gauss upwind: - fvm::Sp(fvc::div(phi), U). Diagonal gets -V*div(phi), stabilises the transient
    // (a rest-start inlet cell where div(phi)!=0 at low nu blows up otherwise); vanishes at convergence.
    if (bounded) { const std::vector<scalar> divPhi = fvc::div(phi, m, lg, lp);
        for (label c = 0; c < lnC; ++c) Ml.diag[c] -= divPhi[c] * lg.V()[c]; }
    DistributedMatrix L = momentumDistributed(Ml, P.Lm, lg, lp, outPhi, nuF, P.procDelta);
    parallelRelaxMatrix(L, Ml, U.internal, P.Lm, lp, relaxU);
    { const std::vector<vector> gP = fvc::gaussGrad(p, m, lg, lp);
      for (int k = 0; k < 3; ++k) {
        DistributedMatrix Lc = L; Lc.diagC = L.diag; Lc.b.assign(lnC, 0.0);
        for (label c = 0; c < lnC; ++c) Lc.b[c] = pcmp(Ml.source[c], k) - lg.V()[c] * pcmp(gP[c], k);
        for (std::size_t pi = 0; pi < lp.size(); ++pi) for (label i = 0; i < lp[pi].size; ++i) { Lc.diagC[lp[pi].faceCells[i]] += pcmp(Ml.internalCoeffs[pi][i], k); Lc.b[lp[pi].faceCells[i]] += pcmp(Ml.boundaryCoeffs[pi][i], k); }
        std::vector<scalar> x(lnC); for (label c = 0; c < lnC; ++c) x[c] = pcmp(U.internal[c], k);
        const SolverPerformance up = parallelPBiCGStab(Lc, x, ifs, P.globalNCells, tolU, 0.0, 2000);
        if (validC[k] && up.initialResidual > res.Ux) res.Ux = up.initialResidual;   // max over solved components
        for (label c = 0; c < lnC; ++c) pset(U.internal[c], k, x[c]);
      } }

    // rAU, HbyA, phiHbyA
    L.diagC = L.diag;
    for (std::size_t pi = 0; pi < lp.size(); ++pi) for (label i = 0; i < lp[pi].size; ++i) L.diagC[lp[pi].faceCells[i]] += cmptAv(Ml.internalCoeffs[pi][i]);
    const std::vector<scalar> A = parallelMatrixA(L, lg.V());
    std::vector<scalar> rAU(lnC); for (label c = 0; c < lnC; ++c) rAU[c] = 1.0 / A[c];
    GeometricField<vector> HbyA = distributeField<vector>(UbcShape, P.gPatches, P.Lm, lp, P.procW, P.rank);
    for (int k = 0; k < 3; ++k) {
        std::vector<scalar> Uk(lnC); for (label c = 0; c < lnC; ++c) Uk[c] = pcmp(U.internal[c], k);
        const std::vector<scalar> Apsi = parallelAmul(L, Uk, ifs);
        std::vector<scalar> Hk(lnC);
        for (label c = 0; c < lnC; ++c) Hk[c] = L.diag[c] * Uk[c] - Apsi[c] + pcmp(Ml.source[c], k);
        for (std::size_t pi = 0; pi < lp.size(); ++pi) for (label i = 0; i < lp[pi].size; ++i) { const vector ic = Ml.internalCoeffs[pi][i]; const label fc = lp[pi].faceCells[i]; Hk[fc] += (cmptAv(ic) - pcmp(ic, k)) * Uk[fc] + pcmp(Ml.boundaryCoeffs[pi][i], k); }
        for (label c = 0; c < lnC; ++c) pset(HbyA.internal[c], k, rAU[c] * Hk[c] / lg.V()[c]);
    }
    HbyA.evaluateBoundary();
    const SurfaceScalarField phiHbyA = fvc::flux(HbyA, m, lg, lp);

    // pEqn = laplacian(rAU,p) == div(phiHbyA)
    GeometricField<scalar> rAUfld = distributeFromCells(rAU, P);
    const SurfaceScalarField rAUf = fvc::interpolate(rAU, m, lg, lp);
    std::vector<scalar> rAUFf(m.nFaces(), 0.0);
    for (label f = 0; f < nIf; ++f) rAUFf[f] = rAUf.internal[f];
    for (std::size_t pi = 0; pi < lp.size(); ++pi) if (lp[pi].type == "processor") for (label i = 0; i < lp[pi].size; ++i) rAUFf[lp[pi].start + i] = rAUfld.boundary[pi]->value()[i];
    FvScalarMatrix Mpl = fvm::laplacian(rAUf, p, m, lg, lp);
    const std::vector<scalar> dphi = fvc::div(phiHbyA, m, lg, lp);
    for (label c = 0; c < lnC; ++c) Mpl.source[c] += lg.V()[c] * dphi[c];
    DistributedMatrix Lp = assembleLocalLaplacianF(P.Lm, lg, lp, P.procDelta, rAUFf);
    Lp.diag = Mpl.diag; Lp.upper = Mpl.upper; Lp.lower = Mpl.lower;
    { std::size_t pj = 0; for (std::size_t pi = 0; pi < lp.size(); ++pi) { if (lp[pi].type != "processor") continue; for (label i = 0; i < lp[pi].size; ++i) Lp.diag[lp[pi].faceCells[i]] -= rAUFf[lp[pi].start + i] * lg.magSf()[lp[pi].start + i] * P.procDelta[pj][i]; ++pj; } }
    Lp.diagC = Lp.diag; Lp.b.resize(lnC);
    for (label c = 0; c < lnC; ++c) Lp.b[c] = Mpl.source[c];
    for (std::size_t pi = 0; pi < lp.size(); ++pi) for (label i = 0; i < lp[pi].size; ++i) { Lp.diagC[lp[pi].faceCells[i]] += Mpl.internalCoeffs[pi][i]; Lp.b[lp[pi].faceCells[i]] += Mpl.boundaryCoeffs[pi][i]; }
    const std::vector<scalar> pPrev = p.internal;
    res.p = parallelPCG(Lp, p.internal, ifs, P.globalNCells, tolP, 0.0, 2000).initialResidual;
    p.evaluateBoundary();
    const SurfaceScalarField pflux = parallelMatrixFlux(Mpl, Lp, p, m, lp);

    // conservative phi + relax + corrector
    for (label f = 0; f < nIf; ++f) phi.internal[f] = phiHbyA.internal[f] - pflux.internal[f];
    for (std::size_t pi = 0; pi < lp.size(); ++pi) for (label i = 0; i < lp[pi].size; ++i) phi.boundary[pi][i] = phiHbyA.boundary[pi][i] - pflux.boundary[pi][i];
    for (label c = 0; c < lnC; ++c) p.internal[c] = pPrev[c] + relaxP * (p.internal[c] - pPrev[c]);
    p.evaluateBoundary();
    const std::vector<vector> gPn = fvc::gaussGrad(p, m, lg, lp);
    for (label c = 0; c < lnC; ++c) U.internal[c] = HbyA.internal[c] - rAU[c] * gPn[c];
    U.evaluateBoundary();
    return res;
}

} // namespace brae
