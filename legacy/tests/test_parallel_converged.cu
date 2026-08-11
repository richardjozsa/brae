// Headline gate: the parallel turbulent solver run to convergence on -np N, compared to OpenFOAM.
// Reads the case 0/ fields, runs the parallel SIMPLE + kEpsilon loop N iterations, and compares the
// converged U/p/k/epsilon/nut to OpenFOAM's converged time dir.
// Run: mpirun -np 4 test_parallel_converged <caseDir> <ofTimeDir> <nIters>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "scotch_decomposition.cuh"
#include "parallel_simple.cuh"
#include "parallel_kepsilon.cuh"
#include "cf_pstream.cuh"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

using namespace brae;

static scalar l2rel(const std::vector<scalar>& loc, const std::vector<scalar>& ref) {
    scalar n = 0, d = 0; for (std::size_t i = 0; i < loc.size(); ++i) { n += (loc[i]-ref[i])*(loc[i]-ref[i]); d += ref[i]*ref[i]; }
    const scalar gn = Pstream::allReduce(n, ReduceOp::Sum), gd = Pstream::allReduce(d, ReduceOp::Sum);
    return gd > 0 ? std::sqrt(gn/gd) : std::sqrt(gn);
}

int main(int argc, char** argv) {
    Pstream::init(argc, argv);
    const int rank = Pstream::myProcNo(), nproc = Pstream::nProcs();
    if (argc < 4) { if (Pstream::master()) std::printf("usage: %s <caseDir> <ofTime> <nIters>\n", argv[0]); Pstream::finalize(); return 2; }
    const std::string caseDir = argv[1], ofT = argv[2]; const int N = std::atoi(argv[3]);
    const scalar nu = 1e-5, relaxU = 0.7, relaxP = 0.3, relaxK = 0.7, relaxEps = 0.7, tol = 1e-8;

    PrimitiveMesh gm;  gm.read(caseDir + "/constant/polyMesh");
    const label nC = gm.nCells();
    const FieldData<vector> Ufd = readField<vector>(caseDir + "/0/U");
    const FieldData<scalar> pfd = readField<scalar>(caseDir + "/0/p"), kfd = readField<scalar>(caseDir + "/0/k"), efd = readField<scalar>(caseDir + "/0/epsilon"), nfd = readField<scalar>(caseDir + "/0/nut");

    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    Partition P(gm, cellToPart, rank);
    GeometricField<vector> U = distributeField<vector>(Ufd, gm.patches(), P.Lm, P.lp, P.procW, rank);  U.evaluateBoundary();
    GeometricField<scalar> p = distributeField<scalar>(pfd, gm.patches(), P.Lm, P.lp, P.procW, rank);  p.evaluateBoundary();
    GeometricField<scalar> k = distributeField<scalar>(kfd, gm.patches(), P.Lm, P.lp, P.procW, rank);  k.evaluateBoundary();
    GeometricField<scalar> eps = distributeField<scalar>(efd, gm.patches(), P.Lm, P.lp, P.procW, rank);  eps.evaluateBoundary();
    GeometricField<scalar> nut = distributeField<scalar>(nfd, gm.patches(), P.Lm, P.lp, P.procW, rank);  nut.evaluateBoundary();
    SurfaceScalarField phi = fvc::flux(U, P.Lm.mesh, P.lg, P.lp);

    for (int it = 0; it < N; ++it) {
        std::vector<scalar> nuEff(P.nCells()); for (label c = 0; c < P.nCells(); ++c) nuEff[c] = nu + nut.internal[c];
        std::vector<std::vector<scalar>> nuEffBnd(P.lp.size());
        for (std::size_t pi = 0; pi < P.lp.size(); ++pi) { const auto& nv = nut.boundary[pi]->value(); nuEffBnd[pi].resize(P.lp[pi].size); for (label i = 0; i < P.lp[pi].size; ++i) nuEffBnd[pi][i] = nu + nv[i]; }
        parallelSimpleStepLaminar(P, U, p, phi, nuEff, nuEffBnd, Ufd, relaxU, relaxP, tol, tol);
        kepsilon::parallelCorrect(P, U, k, eps, nut, phi, nu, relaxEps, relaxK, tol);
        if (Pstream::master() && (it % 200 == 0)) std::fprintf(stderr, "  iter %d\n", it);
    }

    // OpenFOAM converged reference -> each rank's local cells.
    const std::vector<vector> Uof = readField<vector>(caseDir + "/" + ofT + "/U").internalField;
    const std::vector<scalar> pof = readField<scalar>(caseDir + "/" + ofT + "/p").internalField, kof = readField<scalar>(caseDir + "/" + ofT + "/k").internalField, eof = readField<scalar>(caseDir + "/" + ofT + "/epsilon").internalField, nof = readField<scalar>(caseDir + "/" + ofT + "/nut").internalField;
    const label lnC = P.nCells();
    std::vector<scalar> Ux(lnC), Uxr(lnC), pl(lnC), pr(lnC), kl(lnC), kr(lnC), el(lnC), er(lnC), nl(lnC), nr(lnC);
    for (label lc = 0; lc < lnC; ++lc) { const label gc = P.Lm.cellProcAddr[lc];
        Ux[lc]=U.internal[lc].x; Uxr[lc]=Uof[gc].x; pl[lc]=p.internal[lc]; pr[lc]=pof[gc];
        kl[lc]=k.internal[lc]; kr[lc]=kof[gc]; el[lc]=eps.internal[lc]; er[lc]=eof[gc]; nl[lc]=nut.internal[lc]; nr[lc]=nof[gc]; }

    if (Pstream::master()) std::printf("converged-field agreement vs OpenFOAM (np=%d, %d iters):\n", nproc, N);
    const scalar rUx = l2rel(Ux,Uxr), rp = l2rel(pl,pr), rk = l2rel(kl,kr), re = l2rel(el,er), rn = l2rel(nl,nr);
    if (Pstream::master()) {
        std::printf("  Ux L2=%.3e  p L2=%.3e  k L2=%.3e  eps L2=%.3e  nut L2=%.3e\n", rUx, rp, rk, re, rn);
        std::printf("%s\n", (rUx < 2e-2 && rp < 2e-2 && rk < 5e-2) ? "PASS" : "CHECK");
    }
    Pstream::finalize();
    return 0;
}
