// GPU offload G6: the SIMPLE momentum predictor on device. Assemble UEqn = div(phi,U) - laplacian(nu,U)
// on the GPU (G4), then solve the 3 velocity components with the device Jacobi-BiCGStab (the matrix is
// NON-symmetric -> BiCGStab, not CG). Validated against the CPU solveVector (pbicgstab/DILU) on the same
// momentum equation: the converged U must agree.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"
#include "solve_vector.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_blas.cuh"
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

    // U: fixedValue inlet (10,0,0), noSlip walls, zeroGradient outlet, empty front/back.
    GeometricField<vector> U; U.internal.resize(nC);
    for (label c = 0; c < nC; ++c) U.internal[c] = { 1.0 + 0.2*std::sin(0.01*c), 0.1*std::cos(0.013*c), 0.0 };
    for (const FvPatch& q : fvp) {
        if (q.type == "empty")        U.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
        else if (q.name == "inlet")   U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(q, true, vector{10,0,0}, std::vector<vector>{}));
        else if (q.type == "wall")    U.boundary.push_back(std::make_unique<NoSlipPatchField<vector>>(q));
        else                          U.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q));
    }
    U.evaluateBoundary();
    GeometricField<scalar> p; p.internal.resize(nC);
    for (label c = 0; c < nC; ++c) p.internal[c] = std::sin(0.008 * c);
    for (const FvPatch& q : fvp) { if (q.type == "empty") p.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q));
        else p.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q)); }
    p.evaluateBoundary();

    const SurfaceScalarField phi = fvc::flux(U, m, g, fvp);
    const std::vector<vector> gradP = fvc::gaussGrad(p, m, g, fvp);

    // CPU momentum predictor.
    FvVectorMatrix UEqn = fvm::div(phi.internal, phi.boundary, U, m, fvp);
    addEqual(UEqn, fvm::laplacian(U, nu, m, g, fvp), -1.0);
    FvVectorMatrix Mp = UEqn;
    for (label c = 0; c < nC; ++c) Mp.source[c] += (-g.V()[c]) * gradP[c];
    const std::vector<vector> Uinit = U.internal;
    solveVector(Mp, U, m, fvp, tol, 0.0, 2000);
    const std::vector<vector> Ucpu = U.internal;

    // --- DEVICE momentum predictor ---
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    DeviceBuffer<scalar> dphi(phi.internal), gammaf(std::vector<scalar>(nIf, nu));
    DeviceBuffer<scalar> dvDiag, dvUp, dvLo, lpDiag, lpUp, lpLo;
    deviceDivUpwindCoeffs(dm, dphi, dvDiag, dvUp, dvLo);
    deviceLaplacianCoeffs(dm, gammaf, lpDiag, lpUp, lpLo);
    deviceAxpy(-1.0, lpDiag, dvDiag); deviceAxpy(-1.0, lpUp, dvUp); deviceAxpy(-1.0, lpLo, dvLo);  // UEqn = div - laplacian

    auto normFactorOF = [&](const std::vector<scalar>& diagC, const std::vector<scalar>& b, const std::vector<scalar>& psi) {
        std::vector<scalar> yA(nC); for (label c = 0; c < nC; ++c) yA[c] = diagC[c]*psi[c];
        for (label f = 0; f < nIf; ++f) { yA[nei[f]] += Mp.lower[f]*psi[own[f]]; yA[own[f]] += Mp.upper[f]*psi[nei[f]]; }
        std::vector<scalar> sumA = diagC; for (label f = 0; f < nIf; ++f) { sumA[nei[f]] += Mp.lower[f]; sumA[own[f]] += Mp.upper[f]; }
        scalar xRef = 0; for (label c = 0; c < nC; ++c) xRef += psi[c]; xRef /= nC;
        scalar nf = 0; for (label c = 0; c < nC; ++c) { const scalar t = sumA[c]*xRef; nf += std::fabs(yA[c]-t) + std::fabs(b[c]-t); } return nf + 1e-20;
    };

    std::vector<vector> Ugpu(nC);
    for (int k = 0; k < 3; ++k) {
        // per-component source (-V*gradP), boundary coeffs, initial guess.
        std::vector<scalar> src(nC), iCflat, bCflat, psi0(nC), diagCk = Mp.diag, bk(nC);
        for (label c = 0; c < nC; ++c) { src[c] = component(Mp.source[c], k); psi0[c] = component(Uinit[c], k); bk[c] = src[c]; }
        for (std::size_t pi = 0; pi < fvp.size(); ++pi) for (label i = 0; i < fvp[pi].size; ++i) {
            iCflat.push_back(component(Mp.internalCoeffs[pi][i], k)); bCflat.push_back(component(Mp.boundaryCoeffs[pi][i], k));
            const label c = fvp[pi].faceCells[i]; diagCk[c] += component(Mp.internalCoeffs[pi][i], k); bk[c] += component(Mp.boundaryCoeffs[pi][i], k);
        }
        const scalar nf = normFactorOF(diagCk, bk, psi0);
        DeviceBuffer<scalar> dsrc(src), diC(iCflat), dbC(bCflat), diagC, db, dpsi(psi0);
        deviceFold(dm, dvDiag, dsrc, diC, dbC, diagC, db);
        deviceJacobiBiCGStab(deviceLduView(dm, diagC, dvUp, dvLo), db, dpsi, nf, tol, 0.0, 5000);
        const std::vector<scalar> uk = dpsi.host();
        for (label c = 0; c < nC; ++c) setComponent(Ugpu[c], k, uk[c]);
    }

    scalar nX = 0, dX = 0, nY = 0, dY = 0;
    for (label c = 0; c < nC; ++c) {
        nX = std::fmax(nX, std::fabs(Ugpu[c].x - Ucpu[c].x)); dX = std::fmax(dX, std::fabs(Ucpu[c].x));
        nY = std::fmax(nY, std::fabs(Ugpu[c].y - Ucpu[c].y)); dY = std::fmax(dY, std::fabs(Ucpu[c].y));
    }
    std::printf("GPU G6 momentum predictor (nCells=%d):\n", nC);
    std::printf("  Ux device vs CPU : %.3e\n", nX / dX);
    std::printf("  Uy device vs CPU : %.3e\n", nY / dY);
    const bool pass = nX / dX < 1e-6 && nY / dY < 1e-6;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
