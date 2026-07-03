// lduInterface increment 4d: one FULL parallel SIMPLE step (laminar, simplified momentum
// div(phi,U)-laplacian(nu,U)) == the serial step. Momentum predictor + rAU/HbyA/phiHbyA + pEqn +
// conservative flux + corrector, all on the validated parallel ops. The decomposed U/p/phi match
// the serial one-step result. Run: mpirun -np {1,2,4,8} test_parallel_simplestep <caseDir>
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
#include "local_mesh.cuh"
#include "local_assembly.cuh"
#include "field_distribute.cuh"
#include "parallel_amul.cuh"
#include "parallel_pcg.cuh"
#include "parallel_pbicgstab.cuh"
#include "parallel_matrix_ops.cuh"
#include "cf_pstream.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

static scalar cmp(const vector& v, int c) { return c == 0 ? v.x : (c == 1 ? v.y : v.z); }
static void setc(vector& v, int c, scalar s) { if (c == 0) v.x = s; else if (c == 1) v.y = s; else v.z = s; }

int main(int argc, char** argv) {
    Pstream::init(argc, argv);
    const int rank = Pstream::myProcNo(), nproc = Pstream::nProcs();
    if (argc < 2) { if (Pstream::master()) std::printf("usage: %s <caseDir>\n", argv[0]); Pstream::finalize(); return 2; }
    const std::string caseDir = argv[1];
    const scalar nu = 1e-5, relaxU = 0.7, relaxP = 0.3, tol = 1e-9;

    PrimitiveMesh gm;  gm.read(caseDir + "/constant/polyMesh");
    FvGeometry gg;     gg.build(gm);
    const std::vector<FvPatch> gp = buildPatches(gm, gg);
    const label nC = gm.nCells(), nIf = gm.nInternalFaces();
    const FieldData<vector> Ufd = readField<vector>(caseDir + "/282/U");
    const FieldData<scalar> pfd = readField<scalar>(caseDir + "/282/p");

    // ============ serial simplified SIMPLE step ============
    std::vector<scalar> Userx(nC), Usery(nC), pser(nC); std::vector<scalar> phiserInt(nIf);
    {
        GeometricField<vector> U = buildField<vector>(Ufd, gp, nC);  U.evaluateBoundary();
        GeometricField<scalar> p = buildField<scalar>(pfd, gp, nC);  p.evaluateBoundary();
        SurfaceScalarField phi = fvc::flux(U, gm, gg, gp);
        FvVectorMatrix UEqn = fvm::div(phi.internal, phi.boundary, U, gm, gp);
        addEqual(UEqn, fvm::laplacian(U, nu, gm, gg, gp), -1.0);
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
        for (label f = 0; f < nIf; ++f) phi.internal[f] = phiHbyA.internal[f] - pflux.internal[f];
        for (label c = 0; c < nC; ++c) p.internal[c] = pPrev[c] + relaxP * (p.internal[c] - pPrev[c]);
        p.evaluateBoundary();
        const std::vector<vector> gPn = fvc::gaussGrad(p, gm, gg, gp);
        for (label c = 0; c < nC; ++c) U.internal[c] = HbyA.internal[c] - rAU[c] * gPn[c];
        for (label c = 0; c < nC; ++c) { Userx[c] = U.internal[c].x; Usery[c] = U.internal[c].y; pser[c] = p.internal[c]; }
        phiserInt = phi.internal;
    }

    // ============ distributed simplified SIMPLE step ============
    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    LocalMesh Lm = buildLocalMesh(gm, cellToPart, rank);
    FvGeometry lg; lg.build(Lm.mesh);
    const std::vector<FvPatch> lp = buildPatches(Lm.mesh, lg);
    const label lnC = Lm.mesh.nCells();
    const auto procDelta = computeProcDeltaCoeffs(Lm, lg, lp);
    const auto procW     = computeProcWeights(Lm, lg, lp);
    std::vector<ProcessorInterface> ifs;
    for (std::size_t j = 0; j < Lm.procNbr.size(); ++j) ifs.emplace_back(rank, Lm.procNbr[j], Lm.procFaceCells[j]);

    GeometricField<vector> U = distributeField<vector>(Ufd, gm.patches(), Lm, lp, procW, rank);  U.evaluateBoundary();
    GeometricField<scalar> p = distributeField<scalar>(pfd, gm.patches(), Lm, lp, procW, rank);  p.evaluateBoundary();

    // build outward-flux per local face from the global flux (so the local fvm div matches serial).
    std::vector<scalar> gBnd(gm.nFaces(), 0.0);
    GeometricField<vector> Ugref = buildField<vector>(Ufd, gp, nC); Ugref.evaluateBoundary();
    const SurfaceScalarField gphi = fvc::flux(Ugref, gm, gg, gp);
    for (std::size_t g2 = 0; g2 < gp.size(); ++g2) for (label i = 0; i < gp[g2].size; ++i) gBnd[gp[g2].start + i] = gphi.boundary[g2][i];
    SurfaceScalarField lphi; lphi.internal.resize(Lm.mesh.nInternalFaces()); lphi.boundary.resize(lp.size());
    std::vector<scalar> outPhi(Lm.mesh.nFaces(), 0.0);
    for (label f = 0; f < Lm.mesh.nFaces(); ++f) { const label gf = Lm.faceGlobal[f]; if (gf < nIf) outPhi[f] = gphi.internal[gf] * (Lm.faceFlip[f] ? -1.0 : 1.0); }
    for (label f = 0; f < Lm.mesh.nInternalFaces(); ++f) lphi.internal[f] = outPhi[f];
    for (std::size_t pi = 0; pi < lp.size(); ++pi) { lphi.boundary[pi].resize(lp[pi].size); for (label i = 0; i < lp[pi].size; ++i) { const label gf = Lm.faceGlobal[lp[pi].start + i]; lphi.boundary[pi][i] = (gf < nIf) ? 0.0 : gBnd[gf]; } }

    // momentum
    FvVectorMatrix Ml = fvm::div(lphi.internal, lphi.boundary, U, Lm.mesh, lp);
    addEqual(Ml, fvm::laplacian(U, nu, Lm.mesh, lg, lp), -1.0);
    const std::vector<scalar> nuF(Lm.mesh.nFaces(), nu);
    DistributedMatrix L = momentumDistributed(Ml, Lm, lg, lp, outPhi, nuF, procDelta);
    parallelRelaxMatrix(L, Ml, U.internal, Lm, lp, relaxU);
    { const std::vector<vector> gPl = fvc::gaussGrad(p, Lm.mesh, lg, lp);
      for (int k = 0; k < 3; ++k) {
        DistributedMatrix Lc = L; Lc.diagC = L.diag; Lc.b.assign(lnC, 0.0);
        for (label c = 0; c < lnC; ++c) Lc.b[c] = cmp(Ml.source[c], k) - lg.V()[c] * cmp(gPl[c], k);
        for (std::size_t pi = 0; pi < lp.size(); ++pi) for (label i = 0; i < lp[pi].size; ++i) { Lc.diagC[lp[pi].faceCells[i]] += cmp(Ml.internalCoeffs[pi][i], k); Lc.b[lp[pi].faceCells[i]] += cmp(Ml.boundaryCoeffs[pi][i], k); }
        std::vector<scalar> x(lnC); for (label c = 0; c < lnC; ++c) x[c] = cmp(U.internal[c], k);
        parallelPBiCGStab(Lc, x, ifs, nC, tol, 0.0, 2000);
        for (label c = 0; c < lnC; ++c) setc(U.internal[c], k, x[c]);
      } }

    // rAU, HbyA, phiHbyA. A() = (diag + cmptAv(internalCoeffs)) / V.
    L.diagC = L.diag;
    for (std::size_t pi = 0; pi < lp.size(); ++pi) for (label i = 0; i < lp[pi].size; ++i) L.diagC[lp[pi].faceCells[i]] += cmptAv(Ml.internalCoeffs[pi][i]);
    const std::vector<scalar> A = parallelMatrixA(L, lg.V());
    std::vector<scalar> rAU(lnC); for (label c = 0; c < lnC; ++c) rAU[c] = 1.0 / A[c];
    GeometricField<vector> HbyA = distributeField<vector>(Ufd, gm.patches(), Lm, lp, procW, rank);
    for (int k = 0; k < 3; ++k) {                                 // full vector H per component
        std::vector<scalar> Uk(lnC); for (label c = 0; c < lnC; ++c) Uk[c] = cmp(U.internal[c], k);
        const std::vector<scalar> Apsi = parallelAmul(L, Uk, ifs);
        std::vector<scalar> Hk(lnC);
        for (label c = 0; c < lnC; ++c) Hk[c] = L.diag[c] * Uk[c] - Apsi[c] + cmp(Ml.source[c], k);   // bdiag + lduH + src
        for (std::size_t pi = 0; pi < lp.size(); ++pi) for (label i = 0; i < lp[pi].size; ++i) {
            const vector ic = Ml.internalCoeffs[pi][i]; const label fc = lp[pi].faceCells[i];
            Hk[fc] += (cmptAv(ic) - cmp(ic, k)) * Uk[fc] + cmp(Ml.boundaryCoeffs[pi][i], k);
        }
        for (label c = 0; c < lnC; ++c) setc(HbyA.internal[c], k, rAU[c] * Hk[c] / lg.V()[c]);
    }
    HbyA.evaluateBoundary();
    const SurfaceScalarField phiHbyA = fvc::flux(HbyA, Lm.mesh, lg, lp);

    // pEqn: laplacian(rAU,p) == div(phiHbyA). rAU as a coupled field for interpolation at proc faces.
    GeometricField<scalar> rAUfld = distributeField<scalar>(pfd, gm.patches(), Lm, lp, procW, rank);  // borrow p's BC shape
    rAUfld.internal = rAU; rAUfld.evaluateBoundary();
    // rAUf for the laplacian: fvc::interpolate keeps the correct REAL-boundary value (cell value;
    // processor patches are ignored by the laplacian). rAUFf carries the per-face gamma for the
    // assembleLocalLaplacianF processor faces (internal interpolated + processor = interpolated rAU).
    SurfaceScalarField rAUf = fvc::interpolate(rAU, Lm.mesh, lg, lp);
    std::vector<scalar> rAUFf(Lm.mesh.nFaces(), 0.0);
    for (label f = 0; f < Lm.mesh.nInternalFaces(); ++f) rAUFf[f] = rAUf.internal[f];
    for (std::size_t pi = 0; pi < lp.size(); ++pi) if (lp[pi].type == "processor") for (label i = 0; i < lp[pi].size; ++i) rAUFf[lp[pi].start + i] = rAUfld.boundary[pi]->value()[i];

    FvScalarMatrix Mpl = fvm::laplacian(rAUf, p, Lm.mesh, lg, lp);
    const std::vector<scalar> dphil = fvc::div(phiHbyA, Lm.mesh, lg, lp);
    for (label c = 0; c < lnC; ++c) Mpl.source[c] += lg.V()[c] * dphil[c];
    DistributedMatrix Lp = assembleLocalLaplacianF(Lm, lg, lp, procDelta, rAUFf);   // raw laplacian (internal+processor)
    // overwrite the internal/diag with the serial-fvm version so real-boundary handling is exact:
    Lp.diag = Mpl.diag; Lp.upper = Mpl.upper; Lp.lower = Mpl.lower;
    // re-add processor diagonal contributions (Mpl has 0 there):
    { std::size_t pj = 0; for (std::size_t pi = 0; pi < lp.size(); ++pi) { if (lp[pi].type != "processor") continue; for (label i = 0; i < lp[pi].size; ++i) Lp.diag[lp[pi].faceCells[i]] -= rAUFf[lp[pi].start + i] * lg.magSf()[lp[pi].start + i] * procDelta[pj][i]; ++pj; } }
    // fold real boundary from Mpl
    Lp.diagC = Lp.diag; Lp.b.resize(lnC);
    for (label c = 0; c < lnC; ++c) Lp.b[c] = Mpl.source[c];
    for (std::size_t pi = 0; pi < lp.size(); ++pi) for (label i = 0; i < lp[pi].size; ++i) { Lp.diagC[lp[pi].faceCells[i]] += Mpl.internalCoeffs[pi][i]; Lp.b[lp[pi].faceCells[i]] += Mpl.boundaryCoeffs[pi][i]; }

    const std::vector<scalar> pPrev = p.internal;
    parallelPCG(Lp, p.internal, ifs, nC, tol, 0.0, 2000);
    p.evaluateBoundary();
    const SurfaceScalarField pflux = parallelMatrixFlux(Mpl, Lp, p, Lm.mesh, lp);
    SurfaceScalarField phi; phi.internal.resize(Lm.mesh.nInternalFaces());
    for (label f = 0; f < Lm.mesh.nInternalFaces(); ++f) phi.internal[f] = phiHbyA.internal[f] - pflux.internal[f];
    for (label c = 0; c < lnC; ++c) p.internal[c] = pPrev[c] + relaxP * (p.internal[c] - pPrev[c]);
    p.evaluateBoundary();
    const std::vector<vector> gPn = fvc::gaussGrad(p, Lm.mesh, lg, lp);
    for (label c = 0; c < lnC; ++c) U.internal[c] = HbyA.internal[c] - rAU[c] * gPn[c];

    // compare U(x,y), p, phi(internal local) to serial
    scalar mU = 0, gU = 0, mP = 0, gP = 0, mPhi = 0, gPhi = 0;
    for (label lc = 0; lc < lnC; ++lc) { const label gc = Lm.cellProcAddr[lc];
        mU = std::fmax(mU, std::fabs(U.internal[lc].x - Userx[gc])); mU = std::fmax(mU, std::fabs(U.internal[lc].y - Usery[gc])); gU = std::fmax(gU, std::fabs(Userx[gc]));
        mP = std::fmax(mP, std::fabs(p.internal[lc] - pser[gc])); gP = std::fmax(gP, std::fabs(pser[gc])); }
    for (label f = 0; f < Lm.mesh.nInternalFaces(); ++f) { const label gf = Lm.faceGlobal[f]; const scalar s = (Lm.faceFlip[f] ? -1.0 : 1.0); mPhi = std::fmax(mPhi, std::fabs(phi.internal[f] - s * phiserInt[gf])); gPhi = std::fmax(gPhi, std::fabs(phiserInt[gf])); }
    const scalar rU = Pstream::allReduce(mU, ReduceOp::Max) / Pstream::allReduce(gU, ReduceOp::Max);
    const scalar rP = Pstream::allReduce(mP, ReduceOp::Max) / Pstream::allReduce(gP, ReduceOp::Max);
    const scalar rPhi = Pstream::allReduce(mPhi, ReduceOp::Max) / Pstream::allReduce(gPhi, ReduceOp::Max);
    if (Pstream::master()) {
        const scalar gate = (nproc == 1) ? 1e-7 : 1e-5;
        std::printf("test_parallel_simplestep np=%d: U rel=%.3e  p rel=%.3e  phi rel=%.3e\n", nproc, rU, rP, rPhi);
        std::printf("%s\n", (rU < gate && rP < gate && rPhi < gate) ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return 0;
}
