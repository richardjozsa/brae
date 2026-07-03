// GPU offload G5: the SIMPLE pressure equation solved END-TO-END on the device. From rAU and the HbyA flux
// (the momentum step's outputs), the device: interpolates rAU to faces (G3) -> assembles laplacian(rAUf,p)
// (G4) -> builds the RHS V*div(phiHbyA) (G3) -> folds the boundary (diagC += internalCoeffs, b +=
// boundaryCoeffs) -> solves with Jacobi-PCG (G2). Validated against the CPU pEqn solve on the SAME system.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "pcg.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
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
    const label nC = m.nCells();

    // p: fixedValue outlet (-> non-singular), zeroGradient inlet/walls, empty front/back.
    GeometricField<scalar> p; p.internal.assign(nC, 0.0);
    for (const FvPatch& q : fvp) {
        if (q.type == "empty")       p.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(q));
        else if (q.name == "outlet") p.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(q, true, 0.0, std::vector<scalar>{}));
        else                         p.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(q));
    }
    p.evaluateBoundary();

    // rAU (= 1/A, positive) and the HbyA flux.
    std::vector<scalar> rAU(nC); for (label c = 0; c < nC; ++c) rAU[c] = 0.1 + 0.05 * std::sin(0.01 * c);
    GeometricField<vector> HbyA; HbyA.internal.resize(nC);
    for (label c = 0; c < nC; ++c) HbyA.internal[c] = { std::sin(0.013*c), 0.3*std::cos(0.01*c), 0.0 };
    for (const FvPatch& q : fvp) { if (q.type == "empty") HbyA.boundary.push_back(std::make_unique<EmptyPatchField<vector>>(q));
        else HbyA.boundary.push_back(std::make_unique<ZeroGradientPatchField<vector>>(q)); }
    HbyA.evaluateBoundary();

    const SurfaceScalarField rAUf    = fvc::interpolate(rAU, m, g, fvp);
    const SurfaceScalarField phiHbyA = fvc::flux(HbyA, m, g, fvp);

    // CPU pEqn solve.
    FvScalarMatrix pEqn = fvm::laplacian(rAUf, p, m, g, fvp);
    const std::vector<scalar> divP = fvc::div(phiHbyA, m, g, fvp);
    for (label c = 0; c < nC; ++c) pEqn.source[c] += g.V()[c] * divP[c];
    const scalar tol = 1e-10;
    std::vector<scalar> xCpu(nC, 0.0);
    const SolverPerformance scpu = pcg(pEqn, xCpu, m, fvp, tol, 0.0, 10000, 0);

    // normFactor (psi0=0 -> sumMag of the folded b).
    std::vector<scalar> bHost = pEqn.source;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) for (label i = 0; i < fvp[pi].size; ++i) bHost[fvp[pi].faceCells[i]] += pEqn.boundaryCoeffs[pi][i];
    scalar normFactor = 0; for (label c = 0; c < nC; ++c) normFactor += std::fabs(bHost[c]); normFactor += 1e-20;

    // --- DEVICE pressure solve ---
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    DeviceBuffer<scalar> drAU(rAU), drAUfInt; deviceInterpolate(dm, drAU, drAUfInt);   // rAUf on device
    DeviceBuffer<scalar> diag, upper, lower; deviceLaplacianCoeffs(dm, drAUfInt, diag, upper, lower);  // assemble
    DeviceBuffer<scalar> dphiInt(phiHbyA.internal), dphiBval(flattenBoundary(phiHbyA.boundary)), divPhi;
    deviceDiv(dm, dphiInt, dphiBval, divPhi);                                            // V*div RHS source
    DeviceBuffer<scalar> diC(flattenBoundary(pEqn.internalCoeffs)), dbC(flattenBoundary(pEqn.boundaryCoeffs));
    DeviceBuffer<scalar> diagC, db; deviceFoldPressure(dm, diag, divPhi, diC, dbC, diagC, db);  // fold boundary + source
    DeviceBuffer<scalar> dx(std::vector<scalar>(nC, 0.0));
    const DeviceSolverPerf sgpu = deviceJacobiPCG(deviceLduView(dm, diagC, upper, lower), db, dx, normFactor, tol, 0.0, 30000);
    const std::vector<scalar> xGpu = dx.host();

    scalar nD = 0, dD = 0; for (label c = 0; c < nC; ++c) { nD = std::fmax(nD, std::fabs(xGpu[c] - xCpu[c])); dD = std::fmax(dD, std::fabs(xCpu[c])); }
    const scalar solDiff = nD / dD;

    std::printf("GPU G5 device pressure equation (nCells=%d):\n", nC);
    std::printf("  CPU pEqn(DIC) : %d iters, finalRes %.2e\n", scpu.nIterations, scpu.finalResidual);
    std::printf("  device pEqn   : %d iters, finalRes %.2e\n", sgpu.nIterations, sgpu.finalResidual);
    std::printf("  solution diff (device vs CPU) : %.3e\n", solDiff);
    const bool pass = sgpu.finalResidual < tol && solDiff < 1e-6;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
