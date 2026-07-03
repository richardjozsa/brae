// GPU offload G2: device-resident Jacobi-PCG. Solve the SAME linear system A x = b with cf's CPU pcg
// (DIC) and the device Jacobi-PCG, and compare the converged solutions, a non-singular SPD system has a
// unique solution, so both solvers must agree (to ~tolerance) despite different preconditioners and
// iteration counts. Matrix: a Gauss Laplacian on pitzDaily with fixedValue inlet/outlet + a volumetric source.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "pcg.cuh"
#include "device_buffer.cuh"
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
    const label nC = m.nCells(), nIf = m.nInternalFaces();

    // field with fixedValue inlet/outlet (-> non-singular), zeroGradient walls, empty front/back.
    GeometricField<scalar> f; f.internal.assign(nC, 0.0);
    for (const FvPatch& p : fvp) {
        if (p.type == "empty")     f.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(p));
        else if (p.type == "wall") f.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(p));
        else                       f.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(p, true, 0.0, std::vector<scalar>{}));
    }
    f.evaluateBoundary();
    FvScalarMatrix M = fvm::laplacian(f, 1.0, m, g, fvp);
    for (label c = 0; c < nC; ++c) M.source[c] = g.V()[c] * std::sin(0.01 * c + 0.4);   // manufactured source

    const scalar tol = 1e-10;
    // CPU solve (DIC pcg).
    std::vector<scalar> xCpu(nC, 0.0);
    const SolverPerformance scpu = pcg(M, xCpu, m, fvp, tol, 0.0, 10000, 0);

    // Fold boundary -> diagC, b (exactly as brae::pcg does internally).
    std::vector<scalar> diagC = M.diag, b = M.source;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i) { const label c = fvp[pi].faceCells[i];
            diagC[c] += M.internalCoeffs[pi][i]; b[c] += M.boundaryCoeffs[pi][i]; }
    scalar normFactor = 0.0; for (label c = 0; c < nC; ++c) normFactor += std::fabs(b[c]); normFactor += 1e-20;  // psi0=0

    // Device solve (Jacobi PCG).
    const std::vector<label> ownerInt(m.owner().begin(), m.owner().begin() + nIf);
    const DeviceLduMatrix dM = buildDeviceLdu(diagC, M.upper, M.lower, ownerInt, m.neighbour(), nC);
    DeviceBuffer<scalar> db(b), dx(std::vector<scalar>(nC, 0.0));
    const DeviceSolverPerf sgpu = deviceJacobiPCG(dM.view(), db, dx, normFactor, tol, 0.0, 30000);
    const std::vector<scalar> xGpu = dx.host();

    scalar nD = 0, dD = 0;
    for (label c = 0; c < nC; ++c) { nD = std::fmax(nD, std::fabs(xGpu[c] - xCpu[c])); dD = std::fmax(dD, std::fabs(xCpu[c])); }
    const scalar solDiff = nD / dD;

    std::printf("GPU G2 Jacobi-PCG (nCells=%d):\n", nC);
    std::printf("  CPU pcg(DIC)   : %d iters, finalRes %.2e\n", scpu.nIterations, scpu.finalResidual);
    std::printf("  device PCG     : %d iters, finalRes %.2e\n", sgpu.nIterations, sgpu.finalResidual);
    std::printf("  solution diff (device vs CPU): %.3e\n", solDiff);
    const bool pass = sgpu.finalResidual < tol && solDiff < 1e-6;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
