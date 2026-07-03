// GPU offload (#6): AMG preconditioner. Solve a pressure-like Laplacian on pitzDaily with both the device
// Jacobi-PCG and the device AMG-PCG; they must reach the SAME solution, and the AMG V-cycle must cut the
// iteration count (the #1 perf lever vs Jacobi).
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "device_buffer.cuh"
#include "device_ldu.cuh"
#include "device_pcg.cuh"
#include "device_amg.cuh"
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

    GeometricField<scalar> f; f.internal.assign(nC, 0.0);
    for (const FvPatch& p : fvp) {
        if (p.type == "empty")     f.boundary.push_back(std::make_unique<EmptyPatchField<scalar>>(p));
        else if (p.type == "wall") f.boundary.push_back(std::make_unique<ZeroGradientPatchField<scalar>>(p));
        else                       f.boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(p, true, 0.0, std::vector<scalar>{}));
    }
    f.evaluateBoundary();
    FvScalarMatrix M = fvm::laplacian(f, 1.0, m, g, fvp);
    for (label c = 0; c < nC; ++c) M.source[c] = g.V()[c] * std::sin(0.01 * c + 0.4);

    std::vector<scalar> diagC = M.diag, b = M.source;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) for (label i = 0; i < fvp[pi].size; ++i) { const label c = fvp[pi].faceCells[i]; diagC[c] += M.internalCoeffs[pi][i]; b[c] += M.boundaryCoeffs[pi][i]; }
    scalar normFactor = 0; for (label c = 0; c < nC; ++c) normFactor += std::fabs(b[c]); normFactor += 1e-20;
    const std::vector<label> ownerInt(m.owner().begin(), m.owner().begin() + nIf);
    const scalar tol = 1e-10;

    DeviceLduMatrix dM = buildDeviceLdu(diagC, M.upper, M.lower, ownerInt, m.neighbour(), nC);
    DeviceBuffer<scalar> db(b);

    // Jacobi-PCG.
    DeviceBuffer<scalar> xJ(std::vector<scalar>(nC, 0.0));
    const DeviceSolverPerf sJ = deviceJacobiPCG(dM.view(), db, xJ, normFactor, tol, 0.0, 30000);

    // AMG-PCG (faceWeights = |Sf| over the internal faces).
    const std::vector<scalar> magSfInt(g.magSf().begin(), g.magSf().begin() + nIf);
    AMGData amg = buildAMG(ownerInt, m.neighbour(), magSfInt, nC);
    amgGalerkin(amg, dM.diag, dM.upper, dM.lower);
    DeviceBuffer<scalar> xA(std::vector<scalar>(nC, 0.0));
    const DeviceSolverPerf sA = deviceAMGPCG(dM.view(), amg, db, xA, normFactor, tol, 0.0, 30000);

    const std::vector<scalar> xj = xJ.host(), xa = xA.host();
    scalar n = 0, d = 0; for (label c = 0; c < nC; ++c) { n = std::fmax(n, std::fabs(xa[c]-xj[c])); d = std::fmax(d, std::fabs(xj[c])); }
    const scalar solDiff = n / d;

    std::printf("GPU AMG preconditioner (nCells=%d -> %d coarse):\n", nC, amg.nCoarse);
    std::printf("  Jacobi-PCG : %d iters, res %.2e\n", sJ.nIterations, sJ.finalResidual);
    std::printf("  AMG-PCG    : %d iters, res %.2e\n", sA.nIterations, sA.finalResidual);
    std::printf("  solution diff (AMG vs Jacobi): %.3e\n", solDiff);
    std::printf("  iteration reduction          : %.2fx\n", (double)sJ.nIterations / std::max(1, sA.nIterations));
    const bool pass = sA.finalResidual < tol && solDiff < 1e-6 && sA.nIterations < sJ.nIterations;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
