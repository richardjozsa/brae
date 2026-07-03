// lduInterface increment 4d: the parallel TURBULENT loop. Wire nuEff=nu+nut into the SIMPLE step +
// kEpsilon correct() each outer iteration; iterate N times and match the serial turbulent loop
// (simplified momentum, no divDevReff). Run: mpirun -np {1,2,4,8} test_parallel_turbloop <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"
#include "solve_vector.cuh"
#include "pcg.cuh"
#include "k_epsilon.cuh"
#include "scotch_decomposition.cuh"
#include "parallel_simple.cuh"
#include "parallel_kepsilon.cuh"
#include "cf_pstream.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

static SurfaceScalarField readPhi(const std::string& path, const std::vector<FvPatch>& patches) {
    const FieldData<scalar> phiF = readField<scalar>(path);
    SurfaceScalarField phi; phi.internal = phiF.internalField; phi.boundary.resize(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi) { phi.boundary[pi].assign(patches[pi].size, 0.0);
        for (const auto& b : phiF.boundary) if (b.name == patches[pi].name && b.hasValue && (label)b.values.size() == patches[pi].size) phi.boundary[pi] = b.values; }
    return phi;
}

int main(int argc, char** argv) {
    Pstream::init(argc, argv);
    const int rank = Pstream::myProcNo(), nproc = Pstream::nProcs();
    if (argc < 2) { if (Pstream::master()) std::printf("usage: %s <caseDir>\n", argv[0]); Pstream::finalize(); return 2; }
    const std::string caseDir = argv[1];
    const scalar nu = 1e-5, relaxU = 0.7, relaxP = 0.3, tol = 1e-9; const int N = 5;

    PrimitiveMesh gm;  gm.read(caseDir + "/constant/polyMesh");
    FvGeometry gg;     gg.build(gm);
    const std::vector<FvPatch> gp = buildPatches(gm, gg);
    const label nC = gm.nCells(), nIf = gm.nInternalFaces();
    const FieldData<vector> Ufd = readField<vector>(caseDir + "/282/U");
    const FieldData<scalar> pfd = readField<scalar>(caseDir + "/282/p"), kfd = readField<scalar>(caseDir + "/282/k"), efd = readField<scalar>(caseDir + "/282/epsilon"), nfd = readField<scalar>(caseDir + "/282/nut");

    // ---- serial simplified turbulent loop ----
    std::vector<scalar> Userx(nC), pser(nC), kser(nC), eser(nC);
    {
        GeometricField<vector> U = buildField<vector>(Ufd, gp, nC);  U.evaluateBoundary();
        GeometricField<scalar> p = buildField<scalar>(pfd, gp, nC);  p.evaluateBoundary();
        GeometricField<scalar> k = buildField<scalar>(kfd, gp, nC);  k.evaluateBoundary();
        GeometricField<scalar> eps = buildField<scalar>(efd, gp, nC);  eps.evaluateBoundary();
        GeometricField<scalar> nut = buildField<scalar>(nfd, gp, nC);  nut.evaluateBoundary();
        SurfaceScalarField phi = readPhi(caseDir + "/282/phi", gp);
        for (int it = 0; it < N; ++it) {
            std::vector<scalar> nuEff(nC); for (label c = 0; c < nC; ++c) nuEff[c] = nu + nut.internal[c];
            std::vector<std::vector<scalar>> nuEffB(gp.size());
            for (std::size_t pi = 0; pi < gp.size(); ++pi) { const auto& nv = nut.boundary[pi]->value(); nuEffB[pi].resize(gp[pi].size); for (label i = 0; i < gp[pi].size; ++i) nuEffB[pi][i] = nu + nv[i]; }
            SurfaceScalarField nuEfff = fvc::interpolate(nuEff, gm, gg, gp); nuEfff.boundary = nuEffB;
            FvVectorMatrix UEqn = fvm::div(phi.internal, phi.boundary, U, gm, gp);
            addEqual(UEqn, fvm::laplacian(nuEfff, U, gm, gg, gp), -1.0);
            { const std::vector<tensor> gC = fvc::gaussGrad(U, gm, gg, gp); const auto gB = fvc::gradUBoundary(U, gC, gm, gg, gp);
              std::vector<tensor> sC(nC); for (label c = 0; c < nC; ++c) sC[c] = nuEff[c] * dev2(transpose(gC[c]));
              std::vector<std::vector<tensor>> sB(gp.size()); for (std::size_t pi = 0; pi < gp.size(); ++pi) { sB[pi].resize(gp[pi].size); for (label i = 0; i < gp[pi].size; ++i) sB[pi][i] = nuEffB[pi][i] * dev2(transpose(gB[pi][i])); }
              const std::vector<vector> dS = fvc::div(sC, sB, gm, gg, gp); for (label c = 0; c < nC; ++c) UEqn.source[c] += gg.V()[c] * dS[c]; }
            relaxMatrix(UEqn, U, gm, gp, relaxU);
            { const std::vector<vector> gP = fvc::gaussGrad(p, gm, gg, gp); FvVectorMatrix Mp = UEqn; for (label c = 0; c < nC; ++c) Mp.source[c] += (-gg.V()[c]) * gP[c]; solveVector(Mp, U, gm, gp, tol, 0.0, 2000); }
            const std::vector<scalar> A = matrixA(UEqn, gm, gg, gp); std::vector<scalar> rAU(nC); for (label c = 0; c < nC; ++c) rAU[c] = 1.0/A[c];
            const std::vector<vector> H = matrixH(UEqn, U, gm, gg, gp);
            GeometricField<vector> HbyA = buildField<vector>(Ufd, gp, nC); for (label c = 0; c < nC; ++c) HbyA.internal[c] = rAU[c]*H[c]; HbyA.evaluateBoundary();
            const SurfaceScalarField phiHbyA = fvc::flux(HbyA, gm, gg, gp);
            const SurfaceScalarField rAUf = fvc::interpolate(rAU, gm, gg, gp);
            FvScalarMatrix pEqn = fvm::laplacian(rAUf, p, gm, gg, gp);
            const std::vector<scalar> dphi = fvc::div(phiHbyA, gm, gg, gp); for (label c = 0; c < nC; ++c) pEqn.source[c] += gg.V()[c]*dphi[c];
            const std::vector<scalar> pPrev = p.internal; pcg(pEqn, p.internal, gm, gp, tol, 0.0, 2000);
            const SurfaceScalarField pflux = matrixFlux(pEqn, p, gm, gp);
            for (label f = 0; f < nIf; ++f) phi.internal[f] = phiHbyA.internal[f] - pflux.internal[f];
            for (std::size_t pi = 0; pi < gp.size(); ++pi) for (label i = 0; i < gp[pi].size; ++i) phi.boundary[pi][i] = phiHbyA.boundary[pi][i] - pflux.boundary[pi][i];
            for (label c = 0; c < nC; ++c) p.internal[c] = pPrev[c] + relaxP*(p.internal[c] - pPrev[c]); p.evaluateBoundary();
            const std::vector<vector> gPn = fvc::gaussGrad(p, gm, gg, gp); for (label c = 0; c < nC; ++c) U.internal[c] = HbyA.internal[c] - rAU[c]*gPn[c]; U.evaluateBoundary();
            kepsilon::correct(U, k, eps, nut, phi, nu, gm, gg, gp, 1.0, 1.0, tol, 0.0, 2000);
        }
        for (label c = 0; c < nC; ++c) { Userx[c] = U.internal[c].x; pser[c] = p.internal[c]; kser[c] = k.internal[c]; eser[c] = eps.internal[c]; }
    }

    // ---- parallel turbulent loop ----
    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    Partition P(gm, cellToPart, rank);
    GeometricField<vector> U = distributeField<vector>(Ufd, gm.patches(), P.Lm, P.lp, P.procW, rank);  U.evaluateBoundary();
    GeometricField<scalar> p = distributeField<scalar>(pfd, gm.patches(), P.Lm, P.lp, P.procW, rank);  p.evaluateBoundary();
    GeometricField<scalar> k = distributeField<scalar>(kfd, gm.patches(), P.Lm, P.lp, P.procW, rank);  k.evaluateBoundary();
    GeometricField<scalar> eps = distributeField<scalar>(efd, gm.patches(), P.Lm, P.lp, P.procW, rank);  eps.evaluateBoundary();
    GeometricField<scalar> nut = distributeField<scalar>(nfd, gm.patches(), P.Lm, P.lp, P.procW, rank);  nut.evaluateBoundary();
    // local phi
    const SurfaceScalarField gphi = readPhi(caseDir + "/282/phi", gp);
    std::vector<scalar> gBnd(gm.nFaces(), 0.0); for (std::size_t g2 = 0; g2 < gp.size(); ++g2) for (label i = 0; i < gp[g2].size; ++i) gBnd[gp[g2].start + i] = gphi.boundary[g2][i];
    SurfaceScalarField phi; phi.internal.resize(P.Lm.mesh.nInternalFaces()); phi.boundary.resize(P.lp.size());
    for (label f = 0; f < P.Lm.mesh.nInternalFaces(); ++f) { const label gf = P.Lm.faceGlobal[f]; phi.internal[f] = gphi.internal[gf] * (P.Lm.faceFlip[f] ? -1.0 : 1.0); }
    for (std::size_t pi = 0; pi < P.lp.size(); ++pi) { phi.boundary[pi].resize(P.lp[pi].size); for (label i = 0; i < P.lp[pi].size; ++i) { const label gf = P.Lm.faceGlobal[P.lp[pi].start + i]; phi.boundary[pi][i] = (gf < nIf) ? gphi.internal[gf]*(P.Lm.faceFlip[P.lp[pi].start+i]?-1.0:1.0) : gBnd[gf]; } }

    for (int it = 0; it < N; ++it) {
        std::vector<scalar> nuEff(P.nCells()); for (label c = 0; c < P.nCells(); ++c) nuEff[c] = nu + nut.internal[c];
        std::vector<std::vector<scalar>> nuEffBnd(P.lp.size());
        for (std::size_t pi = 0; pi < P.lp.size(); ++pi) { const auto& nv = nut.boundary[pi]->value(); nuEffBnd[pi].resize(P.lp[pi].size); for (label i = 0; i < P.lp[pi].size; ++i) nuEffBnd[pi][i] = nu + nv[i]; }
        parallelSimpleStepLaminar(P, U, p, phi, nuEff, nuEffBnd, Ufd, relaxU, relaxP, tol, tol);
        kepsilon::parallelCorrect(P, U, k, eps, nut, phi, nu, 1.0, 1.0, tol);
    }

    scalar mU=0,gU=0,mP=0,gP=0,mK=0,gK=0,mE=0,gE=0;
    for (label lc = 0; lc < P.nCells(); ++lc) { const label gc = P.Lm.cellProcAddr[lc];
        mU = std::fmax(mU, std::fabs(U.internal[lc].x - Userx[gc])); gU = std::fmax(gU, std::fabs(Userx[gc]));
        mP = std::fmax(mP, std::fabs(p.internal[lc] - pser[gc])); gP = std::fmax(gP, std::fabs(pser[gc]));
        mK = std::fmax(mK, std::fabs(k.internal[lc] - kser[gc])); gK = std::fmax(gK, std::fabs(kser[gc]));
        mE = std::fmax(mE, std::fabs(eps.internal[lc] - eser[gc])); gE = std::fmax(gE, std::fabs(eser[gc])); }
    const scalar rU=Pstream::allReduce(mU,ReduceOp::Max)/Pstream::allReduce(gU,ReduceOp::Max), rP=Pstream::allReduce(mP,ReduceOp::Max)/Pstream::allReduce(gP,ReduceOp::Max);
    const scalar rK=Pstream::allReduce(mK,ReduceOp::Max)/Pstream::allReduce(gK,ReduceOp::Max), rE=Pstream::allReduce(mE,ReduceOp::Max)/Pstream::allReduce(gE,ReduceOp::Max);
    if (Pstream::master()) {
        const scalar gate = (nproc == 1) ? 1e-6 : 1e-3;
        std::printf("test_parallel_turbloop np=%d (%d iters): U=%.3e p=%.3e k=%.3e eps=%.3e\n", nproc, N, rU, rP, rK, rE);
        std::printf("%s\n", (rU<gate && rP<gate && rK<gate && rE<gate) ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return 0;
}
