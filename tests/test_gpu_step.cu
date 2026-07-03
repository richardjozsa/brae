// GPU offload, the CLOSED device SIMPLE step. Orchestrate every validated kernel (G3-G7) into one full
// SIMPLE iteration on the GPU: gradP -> momentum assemble+solve -> A/rAU/H/HbyA -> phiHbyA -> pressure
// assemble+fold+solve -> conservative phi -> corrector. Validate the resulting U and p against one CPU
// simpleStep (same simplified momentum: div - laplacian, no under-relaxation). Boundary coefficients come
// from the CPU-assembled matrices (the documented host-BC dependency); all field math runs on device.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"
#include "solve_vector.cuh"
#include "pcg.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_ldu.cuh"
#include "device_blas.cuh"
#include "device_pcg.cuh"
#include "device_simple.cuh"
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    const std::string caseDir = argc > 1 ? argv[1] : "validation/pitzDaily";
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const std::vector<label>& own = m.owner(); const std::vector<label>& nei = m.neighbour();
    const scalar nu = 1e-3, tol = 1e-10;

    GeometricField<vector> U; U.internal.resize(nC);
    for (label c = 0; c < nC; ++c) U.internal[c] = { 1.0 + 0.2*std::sin(0.01*c), 0.1*std::cos(0.013*c), 0.0 };
    for (const FvPatch& q : fvp) {
        if (q.type == "empty")      U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
        else if (q.name == "inlet") U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(q, true, vector{10,0,0}, std::vector<vector>{}));
        else if (q.type == "wall")  U.boundary.push_back(std::make_unique<NoSlipPatchField<vector>>(q));
        else                        U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
    }
    U.evaluateBoundary();
    GeometricField<scalar> p; p.internal.assign(nC, 0.0);
    for (const FvPatch& q : fvp) {
        if (q.type == "empty")        p.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q));
        else if (q.name == "outlet")  p.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(q, true, 0.0, std::vector<scalar>{}));
        else                          p.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
    }
    p.evaluateBoundary();
    const SurfaceScalarField phi = fvc::flux(U, m, g, fvp);
    const std::vector<vector> Uin = U.internal; const std::vector<scalar> pin = p.internal;

    auto flat = [](const std::vector<std::vector<scalar>>& v) { std::vector<scalar> o; for (auto& a : v) o.insert(o.end(), a.begin(), a.end()); return o; };
    auto normF = [&](const std::vector<scalar>& dC, const std::vector<scalar>& up, const std::vector<scalar>& lo, const std::vector<scalar>& b, const std::vector<scalar>& psi) {
        std::vector<scalar> yA(nC); for (label c = 0; c < nC; ++c) yA[c] = dC[c]*psi[c];
        for (label f = 0; f < nIf; ++f) { yA[nei[f]] += lo[f]*psi[own[f]]; yA[own[f]] += up[f]*psi[nei[f]]; }
        std::vector<scalar> sA = dC; for (label f = 0; f < nIf; ++f) { sA[nei[f]] += lo[f]; sA[own[f]] += up[f]; }
        scalar xR = 0; for (scalar v : psi) xR += v; xR /= nC; scalar nf = 0;
        for (label c = 0; c < nC; ++c) { const scalar t = sA[c]*xR; nf += std::fabs(yA[c]-t) + std::fabs(b[c]-t); } return nf + 1e-20; };

    // ---------------- CPU reference step ----------------
    std::vector<scalar> Ucpu_x, Ucpu_y, pcpu;
    FvVectorMatrix UEqn = fvm::div(phi.internal, phi.boundary, U, m, fvp);
    addEqual(UEqn, fvm::laplacian(U, nu, m, g, fvp), -1.0);
    FvScalarMatrix pEqnSaved; SurfaceScalarField phiHbyASaved;
    {
        const std::vector<vector> gradP = fvc::gaussGrad(p, m, g, fvp);
        FvVectorMatrix Mp = UEqn; for (label c = 0; c < nC; ++c) Mp.source[c] += (-g.V()[c]) * gradP[c];
        solveVector(Mp, U, m, fvp, tol, 0.0, 2000);                                     // U predicted
        const std::vector<scalar> A = matrixA(UEqn, m, g, fvp);
        std::vector<scalar> rAU(nC); for (label c = 0; c < nC; ++c) rAU[c] = 1.0/A[c];
        const std::vector<vector> H = matrixH(UEqn, U, m, g, fvp);
        GeometricField<vector> HbyA; HbyA.internal.resize(nC); for (label c = 0; c < nC; ++c) HbyA.internal[c] = rAU[c]*H[c];
        for (const FvPatch& q : fvp) { if (q.type=="empty") HbyA.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
            else if (U.boundary[&q-&fvp[0]]->fixesValue()) HbyA.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(q, false, vector{}, U.boundary[&q-&fvp[0]]->value()));
            else HbyA.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q)); }
        HbyA.evaluateBoundary();
        phiHbyASaved = fvc::flux(HbyA, m, g, fvp);
        const SurfaceScalarField rAUf = fvc::interpolate(rAU, m, g, fvp);
        FvScalarMatrix pEqn = fvm::laplacian(rAUf, p, m, g, fvp);
        const std::vector<scalar> dphi = fvc::div(phiHbyASaved, m, g, fvp);
        for (label c = 0; c < nC; ++c) pEqn.source[c] += g.V()[c]*dphi[c];
        pcg(pEqn, p.internal, m, fvp, tol, 0.0, 10000, 0); p.evaluateBoundary();
        pEqnSaved = pEqn;
        const SurfaceScalarField pflux = matrixFlux(pEqn, p, m, fvp);
        const std::vector<vector> gradPn = fvc::gaussGrad(p, m, g, fvp);
        for (label c = 0; c < nC; ++c) U.internal[c] = HbyA.internal[c] - rAU[c]*gradPn[c];
        Ucpu_x.resize(nC); Ucpu_y.resize(nC); pcpu = p.internal;
        for (label c = 0; c < nC; ++c) { Ucpu_x[c] = U.internal[c].x; Ucpu_y[c] = U.internal[c].y; }
    }

    // ---------------- DEVICE step ----------------
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    // momentum matrix
    DeviceBuffer<scalar> dphi(phi.internal), gammaf(std::vector<scalar>(nIf, nu));
    DeviceBuffer<scalar> mDiag, mUp, mLo, lD, lU, lL;
    deviceDivUpwindCoeffs(dm, dphi, mDiag, mUp, mLo);
    deviceLaplacianCoeffs(dm, gammaf, lD, lU, lL);
    deviceAxpy(-1.0, lD, mDiag); deviceAxpy(-1.0, lU, mUp); deviceAxpy(-1.0, lL, mLo);
    const DeviceLduView Uview = deviceLduView(dm, mDiag, mUp, mLo);
    // gradP (current p), momentum solve per component
    DeviceBuffer<scalar> dp0(pin), dgx, dgy, dgz;
    { GeometricField<scalar> ptmp; ptmp.internal = pin; for (const FvPatch& q : fvp) { if (q.type=="empty") ptmp.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q)); else if (q.name=="outlet") ptmp.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(q,true,0.0,std::vector<scalar>{})); else ptmp.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q)); } ptmp.evaluateBoundary();
      std::vector<std::vector<scalar>> pb; for (std::size_t pi = 0; pi < fvp.size(); ++pi) pb.push_back(ptmp.boundary[pi]->value());
      DeviceBuffer<scalar> pbval(flat(pb)); deviceGaussGrad(dm, dp0, pbval, dgx, dgy, dgz); }
    DeviceBuffer<scalar>* dg[3] = { &dgx, &dgy, &dgz };
    DeviceBuffer<scalar> Ustar[3];
    for (int k = 0; k < 3; ++k) {
        std::vector<scalar> iC, bC, psi0(nC), bk(nC), dCk = UEqn.diag;
        for (label c = 0; c < nC; ++c) psi0[c] = component(Uin[c], k);
        for (std::size_t pi = 0; pi < fvp.size(); ++pi) for (label i = 0; i < fvp[pi].size; ++i) { iC.push_back(component(UEqn.internalCoeffs[pi][i],k)); bC.push_back(component(UEqn.boundaryCoeffs[pi][i],k)); }
        DeviceBuffer<scalar> src; deviceHadamard(src, dm.V, *dg[k]); deviceScale(src, -1.0);   // src = -V*gradP_k
        DeviceBuffer<scalar> diC(iC), dbC(bC), diagCk, bkk; deviceFold(dm, mDiag, src, diC, dbC, diagCk, bkk);
        // normFactor from host-folded matrix
        std::vector<scalar> srcH(nC); { std::vector<scalar> gh = dg[k]->host(); for (label c=0;c<nC;++c) srcH[c] = -g.V()[c]*gh[c]; }
        for (label c = 0; c < nC; ++c) bk[c] = srcH[c];
        for (std::size_t pi = 0; pi < fvp.size(); ++pi) for (label i = 0; i < fvp[pi].size; ++i) { const label c = fvp[pi].faceCells[i]; dCk[c] += component(UEqn.internalCoeffs[pi][i],k); bk[c] += component(UEqn.boundaryCoeffs[pi][i],k); }
        const scalar nf = normF(dCk, UEqn.upper, UEqn.lower, bk, psi0);
        DeviceBuffer<scalar> dpsi(psi0); deviceJacobiBiCGStab(deviceLduView(dm, diagCk, mUp, mLo), bkk, dpsi, nf, tol, 0.0, 5000);
        Ustar[k] = std::move(dpsi);
    }
    // A/rAU
    std::vector<scalar> iCav; for (std::size_t pi=0;pi<fvp.size();++pi) for (label i=0;i<fvp[pi].size;++i) iCav.push_back(cmptAv(UEqn.internalCoeffs[pi][i]));
    DeviceBuffer<scalar> diCav(iCav), zB(std::vector<scalar>(iCav.size(),0.0)), zS(std::vector<scalar>(nC,0.0)), diagA, dum, drAU;
    deviceFold(dm, mDiag, zS, diCav, zB, diagA, dum); deviceReciprocalV(dm, diagA, drAU);
    // H, HbyA
    DeviceBuffer<scalar> HbyA[3];
    for (int k = 0; k < 3; ++k) {
        std::vector<scalar> bdDiag, bdSrc, srcUEqn(nC);
        for (label c = 0; c < nC; ++c) srcUEqn[c] = component(UEqn.source[c], k);
        for (std::size_t pi=0;pi<fvp.size();++pi) for (label i=0;i<fvp[pi].size;++i) { const vector ic=UEqn.internalCoeffs[pi][i]; bdDiag.push_back(cmptAv(ic)-component(ic,k)); bdSrc.push_back(component(UEqn.boundaryCoeffs[pi][i],k)); }
        DeviceBuffer<scalar> dsrc(srcUEqn), dbd(bdDiag), dbs(bdSrc), Hk; deviceMatrixH(Uview, dm, Ustar[k], dsrc, dbd, dbs, Hk);
        deviceHadamard(HbyA[k], drAU, Hk);
    }
    // phiHbyA internal (device) + boundary (from CPU phiHbyA)
    DeviceBuffer<scalar> dphiH; deviceVectorFlux(dm, HbyA[0], HbyA[1], HbyA[2], dphiH);
    DeviceBuffer<scalar> dphiHbval(flat(phiHbyASaved.boundary));
    // pressure assemble + fold + solve
    DeviceBuffer<scalar> drAUf; deviceInterpolate(dm, drAU, drAUf);
    DeviceBuffer<scalar> pD, pU, pL; deviceLaplacianCoeffs(dm, drAUf, pD, pU, pL);
    DeviceBuffer<scalar> ddiv; deviceDiv(dm, dphiH, dphiHbval, ddiv);
    DeviceBuffer<scalar> piC(flat(pEqnSaved.internalCoeffs)), pbC(flat(pEqnSaved.boundaryCoeffs)), diagCp, bp;
    deviceFoldPressure(dm, pD, ddiv, piC, pbC, diagCp, bp);
    std::vector<scalar> bpHost = bp.host(); scalar pnf = 0; for (scalar v : bpHost) pnf += std::fabs(v); pnf += 1e-20;  // psi0=pin? use 0 like CPU? CPU starts from current p
    // CPU pcg starts from current p (pin); replicate normFactor with pin
    { std::vector<scalar> dCp = diagCp.host(); pnf = normF(dCp, pEqnSaved.upper, pEqnSaved.lower, bpHost, pin); }
    DeviceBuffer<scalar> dpsol(pin); deviceJacobiPCG(deviceLduView(dm, diagCp, pU, pL), bp, dpsol, pnf, tol, 0.0, 30000);
    const std::vector<scalar> pgpu = dpsol.host();
    // corrector: gradP(new p) then U = HbyA - rAU*gradP
    GeometricField<scalar> pnew; pnew.internal = pgpu; for (const FvPatch& q : fvp) { if (q.type=="empty") pnew.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q)); else if (q.name=="outlet") pnew.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(q,true,0.0,std::vector<scalar>{})); else pnew.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q)); }
    pnew.evaluateBoundary();
    std::vector<std::vector<scalar>> pnb; for (std::size_t pi=0;pi<fvp.size();++pi) pnb.push_back(pnew.boundary[pi]->value());
    DeviceBuffer<scalar> dpnew(pgpu), pnbval(flat(pnb)), gnx, gny, gnz; deviceGaussGrad(dm, dpnew, pnbval, gnx, gny, gnz);
    DeviceBuffer<scalar> Ux, Uy; deviceCorrector(HbyA[0], drAU, gnx, Ux); deviceCorrector(HbyA[1], drAU, gny, Uy);

    const std::vector<scalar> uxg = Ux.host(), uyg = Uy.host();
    scalar nUx=0,dUx=0,nUy=0,dUy=0,nP=0,dP=0;
    for (label c = 0; c < nC; ++c) {
        nUx=std::fmax(nUx,std::fabs(uxg[c]-Ucpu_x[c])); dUx=std::fmax(dUx,std::fabs(Ucpu_x[c]));
        nUy=std::fmax(nUy,std::fabs(uyg[c]-Ucpu_y[c])); dUy=std::fmax(dUy,std::fabs(Ucpu_y[c]));
        nP=std::fmax(nP,std::fabs(pgpu[c]-pcpu[c])); dP=std::fmax(dP,std::fabs(pcpu[c])); }
    std::printf("GPU closed SIMPLE step (nCells=%d):\n", nC);
    std::printf("  U.x device vs CPU : %.3e\n", nUx/dUx);
    std::printf("  U.y device vs CPU : %.3e\n", nUy/dUy);
    std::printf("  p   device vs CPU : %.3e\n", nP/dP);
    const bool pass = nUx/dUx < 1e-5 && nUy/dUy < 1e-5 && nP/dP < 1e-5;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
