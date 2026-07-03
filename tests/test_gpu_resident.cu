// GPU offload, the FULLY DEVICE-RESIDENT SIMPLE loop. U/p/phi live as DeviceBuffers across iterations;
// every matrix + boundary coefficient is rebuilt on device each step (on-device BC eval), and there is no
// host round-trip except the scalar reductions inside the solvers. Validate N steps against the CPU loop
// (same simplified momentum: div - laplacian, no under-relaxation). This is the residency capstone.
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
#include "device_boundary.cuh"
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
    const scalar nu = 1e-3, tol = 1e-10, relaxU = 0.7, relaxP = 0.3; const int N = 10;

    auto mkU = [&]() { GeometricField<vector> U; U.internal.resize(nC);
        for (label c = 0; c < nC; ++c) U.internal[c] = { 1.0 + 0.2*std::sin(0.01*c), 0.1*std::cos(0.013*c), 0.0 };
        for (const FvPatch& q : fvp) { if (q.type=="empty") U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
            else if (q.name=="inlet") U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(q,true,vector{10,0,0},std::vector<vector>{}));
            else if (q.type=="wall") U.boundary.push_back(std::make_unique<NoSlipPatchField<vector>>(q));
            else U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q)); } U.evaluateBoundary(); return U; };
    auto mkP = [&]() { GeometricField<scalar> p; p.internal.assign(nC, 0.0);
        for (const FvPatch& q : fvp) { if (q.type=="empty") p.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q));
            else if (q.name=="outlet") p.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(q,true,0.0,std::vector<scalar>{}));
            else p.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q)); } p.evaluateBoundary(); return p; };

    // ---------------- CPU reference loop (N simplified steps) ----------------
    std::vector<scalar> Ucx, Ucy, pc;
    {
        GeometricField<vector> U = mkU(); GeometricField<scalar> p = mkP();
        SurfaceScalarField phi = fvc::flux(U, m, g, fvp);
        for (int it = 0; it < N; ++it) {
            const std::vector<vector> gP = fvc::gaussGrad(p, m, g, fvp);
            FvVectorMatrix UEqn = fvm::div(phi.internal, phi.boundary, U, m, fvp);
            addEqual(UEqn, fvm::laplacian(U, nu, m, g, fvp), -1.0);
            relaxMatrix(UEqn, U, m, fvp, relaxU);
            FvVectorMatrix Mp = UEqn; for (label c = 0; c < nC; ++c) Mp.source[c] += (-g.V()[c])*gP[c];
            solveVector(Mp, U, m, fvp, tol, 0.0, 2000);
            const std::vector<scalar> A = matrixA(UEqn, m, g, fvp);
            std::vector<scalar> rAU(nC); for (label c = 0; c < nC; ++c) rAU[c] = 1.0/A[c];
            const std::vector<vector> H = matrixH(UEqn, U, m, g, fvp);
            GeometricField<vector> HbyA = mkU(); for (label c = 0; c < nC; ++c) HbyA.internal[c] = rAU[c]*H[c]; HbyA.evaluateBoundary();
            SurfaceScalarField phiHbyA = fvc::flux(HbyA, m, g, fvp);
            const SurfaceScalarField rAUf = fvc::interpolate(rAU, m, g, fvp);
            FvScalarMatrix pEqn = fvm::laplacian(rAUf, p, m, g, fvp);
            const std::vector<scalar> dphi = fvc::div(phiHbyA, m, g, fvp);
            for (label c = 0; c < nC; ++c) pEqn.source[c] += g.V()[c]*dphi[c];
            const std::vector<scalar> pPrev = p.internal;
            pcg(pEqn, p.internal, m, fvp, tol, 0.0, 10000, 0);
            const SurfaceScalarField pflux = matrixFlux(pEqn, p, m, fvp);
            for (label f = 0; f < nIf; ++f) phi.internal[f] = phiHbyA.internal[f] - pflux.internal[f];
            for (std::size_t pi = 0; pi < fvp.size(); ++pi) for (label i = 0; i < fvp[pi].size; ++i) phi.boundary[pi][i] = phiHbyA.boundary[pi][i] - pflux.boundary[pi][i];
            for (label c = 0; c < nC; ++c) p.internal[c] = pPrev[c] + relaxP*(p.internal[c] - pPrev[c]);
            p.evaluateBoundary();
            const std::vector<vector> gPn = fvc::gaussGrad(p, m, g, fvp);
            for (label c = 0; c < nC; ++c) U.internal[c] = HbyA.internal[c] - rAU[c]*gPn[c]; U.evaluateBoundary();
        }
        Ucx.resize(nC); Ucy.resize(nC); pc = p.internal;
        for (label c = 0; c < nC; ++c) { Ucx[c] = U.internal[c].x; Ucy[c] = U.internal[c].y; }
    }

    // ---------------- DEVICE resident loop ----------------
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    GeometricField<vector> U0 = mkU(); GeometricField<scalar> p0 = mkP();
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U0, fvp, g);
    const DeviceBoundary dbP = buildDeviceBoundary(p0, fvp, g);
    const SurfaceScalarField phi0 = fvc::flux(U0, m, g, fvp);
    DeviceBuffer<scalar> Uk[3], dp(p0.internal), phiInt(phi0.internal);
    { std::vector<scalar> ux(nC),uy(nC),uz(nC); for (label c=0;c<nC;++c){ux[c]=U0.internal[c].x;uy[c]=U0.internal[c].y;uz[c]=U0.internal[c].z;} Uk[0].copyFrom(ux);Uk[1].copyFrom(uy);Uk[2].copyFrom(uz); }
    DeviceBuffer<scalar> phiBnd(([&]{ std::vector<scalar> o; for (auto& a : phi0.boundary) o.insert(o.end(),a.begin(),a.end()); return o; }()));
    DeviceBuffer<scalar> nuU(std::vector<scalar>(nIf, nu)), zeroBnd(std::vector<scalar>(dbP.n, 0.0)), zeroSrc(std::vector<scalar>(nC, 0.0)), zeroBndU(std::vector<scalar>(dbU.n, 0.0));
    DeviceBuffer<scalar> lD, lU, lL; deviceLaplacianCoeffs(dm, nuU, lD, lU, lL);   // momentum laplacian (nu const)
    auto sumMag = [](DeviceBuffer<scalar>& b){ return deviceSumMag(b) + 1e-20; };

    for (int it = 0; it < N; ++it) {
        // gradP
        DeviceBuffer<scalar> pbv; deviceBCValue(dbP, dp, pbv);
        DeviceBuffer<scalar> gx, gy, gz; deviceGaussGrad(dm, dp, pbv, gx, gy, gz);
        // momentum matrix
        DeviceBuffer<scalar> mDiag, mUp, mLo; deviceDivUpwindCoeffs(dm, phiInt, mDiag, mUp, mLo);
        deviceAxpy(-1.0, lD, mDiag); deviceAxpy(-1.0, lU, mUp); deviceAxpy(-1.0, lL, mLo);
        DeviceBuffer<scalar>* gg[3] = { &gx, &gy, &gz };
        DeviceBuffer<scalar> nuCell(std::vector<scalar>(nC, nu));   // cell diffusivity for the laplacian BC coeffs
        // RELAX: combined boundary internalCoeffs (comp 0, isotropic) -> relaxed diag + delta
        DeviceBuffer<scalar> r0IC, r0BC, r0lIC, r0lBC; deviceBCDivCoeffs(dbU.comp[0], phiBnd, r0IC, r0BC); deviceBCLaplacianCoeffs(dbU.comp[0], nuCell, r0lIC, r0lBC);
        deviceAxpy(-1.0, r0lIC, r0IC);
        DeviceBuffer<scalar> mDiagR, delta; deviceRelaxDiag(deviceLduView(dm, mDiag, mUp, mLo), dm, r0IC, relaxU, mDiagR, delta);
        const DeviceLduView Uview = deviceLduView(dm, mDiagR, mUp, mLo);
        DeviceBuffer<scalar> iC[3], bCb[3], src[3], relaxSrc[3];
        for (int k = 0; k < 3; ++k) {
            DeviceBuffer<scalar> dIC, dBC, lIC, lBC;
            deviceBCDivCoeffs(dbU.comp[k], phiBnd, dIC, dBC);
            deviceBCLaplacianCoeffs(dbU.comp[k], nuCell, lIC, lBC);
            deviceAxpy(-1.0, lIC, dIC); deviceAxpy(-1.0, lBC, dBC);   // UEqn = div - laplacian
            iC[k] = std::move(dIC); bCb[k] = std::move(dBC);
            deviceHadamard(relaxSrc[k], delta, Uk[k]);                // relaxation source = delta*U (pre-solve)
            DeviceBuffer<scalar> s; deviceHadamard(s, dm.V, *gg[k]); deviceScale(s, -1.0); deviceAxpy(1.0, relaxSrc[k], s); src[k] = std::move(s);  // relaxSrc - V*gradP
        }
        // momentum solve per component (relaxed diag)
        for (int k = 0; k < 3; ++k) {
            DeviceBuffer<scalar> diagC, b; deviceFold(dm, mDiagR, src[k], iC[k], bCb[k], diagC, b);
            deviceJacobiBiCGStab(deviceLduView(dm, diagC, mUp, mLo), b, Uk[k], sumMag(b), tol, 0.0, 5000);
        }
        // A / rAU (relaxed diag)
        DeviceBuffer<scalar> diagA, dumb, rAU; deviceFold(dm, mDiagR, zeroSrc, iC[0], zeroBndU, diagA, dumb); deviceReciprocalV(dm, diagA, rAU);
        // H / HbyA (relaxed matrix; UEqn.source = relaxSrc)
        DeviceBuffer<scalar> HbyA[3];
        for (int k = 0; k < 3; ++k) {
            DeviceBuffer<scalar> Hk; deviceMatrixH(Uview, dm, Uk[k], relaxSrc[k], zeroBndU, bCb[k], Hk);   // bdDiag=0 (isotropic)
            deviceHadamard(HbyA[k], rAU, Hk);
        }
        // phiHbyA (internal + boundary)
        DeviceBuffer<scalar> phiHi; deviceVectorFlux(dm, HbyA[0], HbyA[1], HbyA[2], phiHi);
        DeviceBuffer<scalar> hxb, hyb, hzb; deviceBCValue(dbU.comp[0], HbyA[0], hxb); deviceBCValue(dbU.comp[1], HbyA[1], hyb); deviceBCValue(dbU.comp[2], HbyA[2], hzb);
        DeviceBuffer<scalar> phiHb; deviceBoundaryFlux(dm, hxb, hyb, hzb, phiHb);
        // pressure
        DeviceBuffer<scalar> rAUf; deviceInterpolate(dm, rAU, rAUf);
        DeviceBuffer<scalar> pD, pU, pL; deviceLaplacianCoeffs(dm, rAUf, pD, pU, pL);
        DeviceBuffer<scalar> pIC, pBC; deviceBCLaplacianCoeffs(dbP, rAU, pIC, pBC);
        DeviceBuffer<scalar> divPhiH; deviceDiv(dm, phiHi, phiHb, divPhiH);
        DeviceBuffer<scalar> diagCp, bp; deviceFoldPressure(dm, pD, divPhiH, pIC, pBC, diagCp, bp);
        DeviceBuffer<scalar> pPrev; deviceCopy(pPrev, dp);   // save before solve (field relaxation)
        deviceJacobiPCG(deviceLduView(dm, diagCp, pU, pL), bp, dp, sumMag(bp), tol, 0.0, 30000);
        // conservative phi (from the SOLVED p)
        const DeviceLduView pview = deviceLduView(dm, diagCp, pU, pL);
        DeviceBuffer<scalar> pfi; deviceMatrixFluxInternal(pview, dp, pfi);
        deviceCopy(phiInt, phiHi); deviceAxpy(-1.0, pfi, phiInt);
        DeviceBuffer<scalar> pfb; deviceMatrixFluxBoundary(dbP, pIC, pBC, dp, pfb);
        deviceCopy(phiBnd, phiHb); deviceAxpy(-1.0, pfb, phiBnd);
        // p field relaxation: p = pPrev + relaxP*(p - pPrev), then corrector uses the relaxed p
        deviceScale(dp, relaxP); deviceAxpy(1.0 - relaxP, pPrev, dp);
        // corrector
        DeviceBuffer<scalar> pbv2; deviceBCValue(dbP, dp, pbv2);
        DeviceBuffer<scalar> gnx, gny, gnz; deviceGaussGrad(dm, dp, pbv2, gnx, gny, gnz);
        DeviceBuffer<scalar>* gn[3] = { &gnx, &gny, &gnz };
        for (int k = 0; k < 3; ++k) { DeviceBuffer<scalar> Unew; deviceCorrector(HbyA[k], rAU, *gn[k], Unew); Uk[k] = std::move(Unew); }
    }

    const std::vector<scalar> uxg = Uk[0].host(), uyg = Uk[1].host(), pg = dp.host();
    scalar nUx=0,dUx=0,nUy=0,dUy=0,nP=0,dP=0;
    for (label c = 0; c < nC; ++c) {
        nUx=std::fmax(nUx,std::fabs(uxg[c]-Ucx[c])); dUx=std::fmax(dUx,std::fabs(Ucx[c]));
        nUy=std::fmax(nUy,std::fabs(uyg[c]-Ucy[c])); dUy=std::fmax(dUy,std::fabs(Ucy[c]));
        nP=std::fmax(nP,std::fabs(pg[c]-pc[c])); dP=std::fmax(dP,std::fabs(pc[c])); }
    std::printf("GPU resident SIMPLE loop (%d steps, nCells=%d, fields stay on GPU):\n", N, nC);
    std::printf("  U.x device vs CPU : %.3e\n", nUx/dUx);
    std::printf("  U.y device vs CPU : %.3e\n", nUy/dUy);
    std::printf("  p   device vs CPU : %.3e\n", nP/dP);
    const bool pass = nUx/dUx < 1e-4 && nUy/dUy < 1e-4 && nP/dP < 1e-4;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
