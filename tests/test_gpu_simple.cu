// GPU offload G7: the SIMPLE p-U coupling glue on device. From the momentum matrix UEqn = div - laplacian,
// validate every coupling quantity the SIMPLE step needs against the CPU: rAU=1/A, H(), the HbyA flux
// phiHbyA, the pressure flux pEqn.flux(), and the velocity corrector U = HbyA - rAU*grad(p). With G5/G6
// (the two solves) this completes the kernel set for a fully device-resident SIMPLE loop.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_ldu.cuh"
#include "device_blas.cuh"
#include "device_simple.cuh"
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace brae;

static scalar relMax(const std::vector<scalar>& a, const std::vector<scalar>& b) {
    scalar n = 0, d = 0; for (std::size_t i = 0; i < a.size(); ++i) { n = std::fmax(n, std::fabs(a[i]-b[i])); d = std::fmax(d, std::fabs(b[i])); }
    return d > 0 ? n / d : n;
}

int main(int argc, char** argv) {
    const std::string caseDir = argc > 1 ? argv[1] : "validation/pitzDaily";
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    const scalar nu = 1e-3;

    GeometricField<vector> U; U.internal.resize(nC);
    for (label c = 0; c < nC; ++c) U.internal[c] = { 1.0 + 0.2*std::sin(0.01*c), 0.1*std::cos(0.013*c), 0.0 };
    for (const FvPatch& q : fvp) {
        if (q.type == "empty")      U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
        else if (q.name == "inlet") U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(q, true, vector{10,0,0}, std::vector<vector>{}));
        else if (q.type == "wall")  U.boundary.push_back(std::make_unique<NoSlipPatchField<vector>>(q));
        else                        U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
    }
    U.evaluateBoundary();
    GeometricField<scalar> p; p.internal.resize(nC);
    for (label c = 0; c < nC; ++c) p.internal[c] = std::sin(0.008 * c);
    for (const FvPatch& q : fvp) { if (q.type == "empty") p.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q));
        else p.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q)); }
    p.evaluateBoundary();

    const SurfaceScalarField phi = fvc::flux(U, m, g, fvp);
    FvVectorMatrix UEqn = fvm::div(phi.internal, phi.boundary, U, m, fvp);
    addEqual(UEqn, fvm::laplacian(U, nu, m, g, fvp), -1.0);

    // CPU coupling quantities.
    const std::vector<scalar> A = matrixA(UEqn, m, g, fvp);
    std::vector<scalar> rAUcpu(nC); for (label c = 0; c < nC; ++c) rAUcpu[c] = 1.0 / A[c];
    const std::vector<vector> H = matrixH(UEqn, U, m, g, fvp);
    GeometricField<vector> HbyA; HbyA.internal.resize(nC);
    for (label c = 0; c < nC; ++c) HbyA.internal[c] = rAUcpu[c] * H[c];
    for (const FvPatch& q : fvp) { if (q.type == "empty") HbyA.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q)); else HbyA.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q)); }
    HbyA.evaluateBoundary();
    const SurfaceScalarField phiHbyA = fvc::flux(HbyA, m, g, fvp);
    const SurfaceScalarField rAUf = fvc::interpolate(rAUcpu, m, g, fvp);
    const FvScalarMatrix pEqn = fvm::laplacian(rAUf, p, m, g, fvp);
    const SurfaceScalarField pflux = matrixFlux(pEqn, p, m, fvp);
    const std::vector<vector> gradP = fvc::gaussGrad(p, m, g, fvp);
    std::vector<scalar> UcorrX(nC); for (label c = 0; c < nC; ++c) UcorrX[c] = HbyA.internal[c].x - rAUcpu[c] * gradP[c].x;

    // --- DEVICE ---
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    DeviceBuffer<scalar> dphi(phi.internal), gammaf(std::vector<scalar>(nIf, nu));
    DeviceBuffer<scalar> dvDiag, dvUp, dvLo, lpDiag, lpUp, lpLo;
    deviceDivUpwindCoeffs(dm, dphi, dvDiag, dvUp, dvLo);
    deviceLaplacianCoeffs(dm, gammaf, lpDiag, lpUp, lpLo);
    deviceAxpy(-1.0, lpDiag, dvDiag); deviceAxpy(-1.0, lpUp, dvUp); deviceAxpy(-1.0, lpLo, dvLo);
    const DeviceLduView Uview = deviceLduView(dm, dvDiag, dvUp, dvLo);

    // rAU = V/diagC, diagC = diag + internalCoeffs cmptAv (boundary gather).
    std::vector<scalar> iCcmptAvFlat, zeroBnd;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) for (label i = 0; i < fvp[pi].size; ++i) { iCcmptAvFlat.push_back(cmptAv(UEqn.internalCoeffs[pi][i])); zeroBnd.push_back(0.0); }
    DeviceBuffer<scalar> diCcmptAv(iCcmptAvFlat), dzeroB(zeroBnd), zeroSrc(std::vector<scalar>(nC, 0.0)), diagC, dummyB, drAU;
    deviceFold(dm, dvDiag, zeroSrc, diCcmptAv, dzeroB, diagC, dummyB);
    deviceReciprocalV(dm, diagC, drAU);

    // H per component, HbyA = rAU*H.
    DeviceBuffer<scalar> dHbyAx, dHbyAy, dHbyAz;
    for (int k = 0; k < 3; ++k) {
        std::vector<scalar> src(nC), psi(nC), bdDiag, bdSrc;
        for (label c = 0; c < nC; ++c) { src[c] = component(UEqn.source[c], k); psi[c] = component(U.internal[c], k); }
        for (std::size_t pi = 0; pi < fvp.size(); ++pi) for (label i = 0; i < fvp[pi].size; ++i) {
            const vector ic = UEqn.internalCoeffs[pi][i]; bdDiag.push_back(cmptAv(ic) - component(ic, k)); bdSrc.push_back(component(UEqn.boundaryCoeffs[pi][i], k)); }
        DeviceBuffer<scalar> dsrc(src), dpsi(psi), dbdDiag(bdDiag), dbdSrc(bdSrc), Hk;
        deviceMatrixH(Uview, dm, dpsi, dsrc, dbdDiag, dbdSrc, Hk);
        DeviceBuffer<scalar>& dst = (k == 0 ? dHbyAx : (k == 1 ? dHbyAy : dHbyAz));
        deviceHadamard(dst, drAU, Hk);
    }
    DeviceBuffer<scalar> dphiHbyA; deviceVectorFlux(dm, dHbyAx, dHbyAy, dHbyAz, dphiHbyA);

    // pEqn flux + corrector.
    DeviceBuffer<scalar> drAUf; deviceInterpolate(dm, drAU, drAUf);
    DeviceBuffer<scalar> pDiag, pUp, pLo; deviceLaplacianCoeffs(dm, drAUf, pDiag, pUp, pLo);
    DeviceBuffer<scalar> dp(p.internal), dpflux; deviceMatrixFluxInternal(deviceLduView(dm, pDiag, pUp, pLo), dp, dpflux);
    std::vector<scalar> gradPx(nC); for (label c = 0; c < nC; ++c) gradPx[c] = gradP[c].x;
    DeviceBuffer<scalar> dgradPx(gradPx), dUcorrX; deviceCorrector(dHbyAx, drAU, dgradPx, dUcorrX);

    std::vector<scalar> phiHcpu(nIf), pfluxcpu(nIf); for (label f = 0; f < nIf; ++f) { phiHcpu[f] = phiHbyA.internal[f]; pfluxcpu[f] = pflux.internal[f]; }
    std::vector<scalar> Hxcpu(nC); for (label c = 0; c < nC; ++c) Hxcpu[c] = H[c].x;
    std::vector<scalar> dHx = dHbyAx.host(); for (label c = 0; c < nC; ++c) dHx[c] /= rAUcpu[c];   // recover H from HbyA for the H check

    std::printf("GPU G7 SIMPLE coupling glue (nCells=%d):\n", nC);
    std::printf("  rAU            : %.3e\n", relMax(drAU.host(), rAUcpu));
    std::printf("  H().x          : %.3e\n", relMax(dHx, Hxcpu));
    std::printf("  phiHbyA (flux) : %.3e\n", relMax(dphiHbyA.host(), phiHcpu));
    std::printf("  pEqn.flux()    : %.3e\n", relMax(dpflux.host(), pfluxcpu));
    std::printf("  corrector U.x  : %.3e\n", relMax(dUcorrX.host(), UcorrX));
    const bool pass = relMax(drAU.host(), rAUcpu) < 1e-12 && relMax(dHx, Hxcpu) < 1e-11
                   && relMax(dphiHbyA.host(), phiHcpu) < 1e-12 && relMax(dpflux.host(), pfluxcpu) < 1e-12
                   && relMax(dUcorrX.host(), UcorrX) < 1e-11;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
