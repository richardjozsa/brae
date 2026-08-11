// lduInterface increment 4d: the parallel SIMPLE LOOP. Iterate the distributed laminar step N times
// (maintaining the conservative phi across iterations) and match the serial loop cell-by-cell.
// Run: mpirun -np {1,2,4,8} test_parallel_loop <caseDir>
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
#include "scotch_decomposition.cuh"
#include "parallel_simple.cuh"
#include "cf_pstream.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    Pstream::init(argc, argv);
    const int rank = Pstream::myProcNo(), nproc = Pstream::nProcs();
    if (argc < 2) { if (Pstream::master()) std::printf("usage: %s <caseDir>\n", argv[0]); Pstream::finalize(); return 2; }
    const std::string caseDir = argv[1];
    const scalar nu = 1e-5, relaxU = 0.7, relaxP = 0.3, tol = 1e-9;
    const int N = 10;

    PrimitiveMesh gm;  gm.read(caseDir + "/constant/polyMesh");
    FvGeometry gg;     gg.build(gm);
    const std::vector<FvPatch> gp = buildPatches(gm, gg);
    const label nC = gm.nCells();
    const FieldData<vector> Ufd = readField<vector>(caseDir + "/282/U");
    const FieldData<scalar> pfd = readField<scalar>(caseDir + "/282/p");

    // ---- serial simplified laminar loop ----
    std::vector<scalar> Userx(nC), Usery(nC), pser(nC);
    {
        GeometricField<vector> U = buildField<vector>(Ufd, gp, nC);  U.evaluateBoundary();
        GeometricField<scalar> p = buildField<scalar>(pfd, gp, nC);  p.evaluateBoundary();
        SurfaceScalarField phi = fvc::flux(U, gm, gg, gp);
        for (int it = 0; it < N; ++it) {
            FvVectorMatrix UEqn = fvm::div(phi.internal, phi.boundary, U, gm, gp);
            addEqual(UEqn, fvm::laplacian(U, nu, gm, gg, gp), -1.0);
            { const std::vector<tensor> gC = fvc::gaussGrad(U, gm, gg, gp); const auto gB = fvc::gradUBoundary(U, gC, gm, gg, gp);
              std::vector<tensor> sC(nC); for (label c = 0; c < nC; ++c) sC[c] = nu * dev2(transpose(gC[c]));
              std::vector<std::vector<tensor>> sB(gp.size()); for (std::size_t pi = 0; pi < gp.size(); ++pi) { sB[pi].resize(gp[pi].size); for (label i = 0; i < gp[pi].size; ++i) sB[pi][i] = nu * dev2(transpose(gB[pi][i])); }
              const std::vector<vector> dS = fvc::div(sC, sB, gm, gg, gp); for (label c = 0; c < nC; ++c) UEqn.source[c] += gg.V()[c] * dS[c]; }
            relaxMatrix(UEqn, U, gm, gp, relaxU);
            { const std::vector<vector> gP = fvc::gaussGrad(p, gm, gg, gp);
              FvVectorMatrix Mp = UEqn; for (label c = 0; c < nC; ++c) Mp.source[c] += (-gg.V()[c]) * gP[c];
              solveVector(Mp, U, gm, gp, tol, 0.0, 2000); }
            const std::vector<scalar> A = matrixA(UEqn, gm, gg, gp);
            std::vector<scalar> rAU(nC); for (label c = 0; c < nC; ++c) rAU[c] = 1.0 / A[c];
            const std::vector<vector> H = matrixH(UEqn, U, gm, gg, gp);
            GeometricField<vector> HbyA = buildField<vector>(Ufd, gp, nC);
            for (label c = 0; c < nC; ++c) HbyA.internal[c] = rAU[c] * H[c];
            HbyA.evaluateBoundary();
            const SurfaceScalarField phiHbyA = fvc::flux(HbyA, gm, gg, gp);
            const SurfaceScalarField rAUf = fvc::interpolate(rAU, gm, gg, gp);
            FvScalarMatrix pEqn = fvm::laplacian(rAUf, p, gm, gg, gp);
            const std::vector<scalar> dphi = fvc::div(phiHbyA, gm, gg, gp);
            for (label c = 0; c < nC; ++c) pEqn.source[c] += gg.V()[c] * dphi[c];
            const std::vector<scalar> pPrev = p.internal;
            pcg(pEqn, p.internal, gm, gp, tol, 0.0, 2000);
            const SurfaceScalarField pflux = matrixFlux(pEqn, p, gm, gp);
            for (label f = 0; f < gm.nInternalFaces(); ++f) phi.internal[f] = phiHbyA.internal[f] - pflux.internal[f];
            for (std::size_t pi = 0; pi < gp.size(); ++pi) for (label i = 0; i < gp[pi].size; ++i) phi.boundary[pi][i] = phiHbyA.boundary[pi][i] - pflux.boundary[pi][i];
            for (label c = 0; c < nC; ++c) p.internal[c] = pPrev[c] + relaxP * (p.internal[c] - pPrev[c]);
            p.evaluateBoundary();
            const std::vector<vector> gPn = fvc::gaussGrad(p, gm, gg, gp);
            for (label c = 0; c < nC; ++c) U.internal[c] = HbyA.internal[c] - rAU[c] * gPn[c];
            U.evaluateBoundary();
        }
        for (label c = 0; c < nC; ++c) { Userx[c] = U.internal[c].x; Usery[c] = U.internal[c].y; pser[c] = p.internal[c]; }
    }

    // ---- distributed laminar loop ----
    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    Partition P(gm, cellToPart, rank);
    GeometricField<vector> U = distributeField<vector>(Ufd, gm.patches(), P.Lm, P.lp, P.procW, rank);  U.evaluateBoundary();
    GeometricField<scalar> p = distributeField<scalar>(pfd, gm.patches(), P.Lm, P.lp, P.procW, rank);  p.evaluateBoundary();
    SurfaceScalarField phi = fvc::flux(U, P.Lm.mesh, P.lg, P.lp);
    const std::vector<scalar> nuEffCell(P.nCells(), nu);
    std::vector<std::vector<scalar>> nuEffBnd(P.lp.size());
    for (std::size_t pi = 0; pi < P.lp.size(); ++pi) nuEffBnd[pi].assign(P.lp[pi].size, nu);
    for (int it = 0; it < N; ++it) parallelSimpleStepLaminar(P, U, p, phi, nuEffCell, nuEffBnd, Ufd, relaxU, relaxP, tol, tol);

    scalar mU = 0, gU = 0, mP = 0, gP = 0;
    for (label lc = 0; lc < P.nCells(); ++lc) { const label gc = P.Lm.cellProcAddr[lc];
        mU = std::fmax(mU, std::fabs(U.internal[lc].x - Userx[gc])); mU = std::fmax(mU, std::fabs(U.internal[lc].y - Usery[gc])); gU = std::fmax(gU, std::fabs(Userx[gc]));
        mP = std::fmax(mP, std::fabs(p.internal[lc] - pser[gc])); gP = std::fmax(gP, std::fabs(pser[gc])); }
    const scalar rU = Pstream::allReduce(mU, ReduceOp::Max) / Pstream::allReduce(gU, ReduceOp::Max);
    const scalar rP = Pstream::allReduce(mP, ReduceOp::Max) / Pstream::allReduce(gP, ReduceOp::Max);
    if (Pstream::master()) {
        const scalar gate = (nproc == 1) ? 1e-6 : 1e-4;   // 10 iterations of parallel-solver-floor drift
        std::printf("test_parallel_loop np=%d (%d iters): U rel=%.3e  p rel=%.3e\n", nproc, N, rU, rP);
        std::printf("%s\n", (rU < gate && rP < gate) ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return 0;
}
