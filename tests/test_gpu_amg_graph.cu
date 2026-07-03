// GPU offload (#7c): CUDA-graph replay of the AMG V-cycle. The V-cycle has no host-scalar dependencies, so it
// is captured once and replayed every PCG iteration instead of re-launching ~80 kernels from the host each time
//, directly attacking the launch-overhead bottleneck. Validate the graph path gives the IDENTICAL solution and
// iteration count as the direct-launch path, and MEASURE the wall-clock of a full pressure solve both ways.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "device_buffer.cuh"
#include "device_ldu.cuh"
#include "device_amg.cuh"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>
#include <cuda_runtime.h>

using namespace brae;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

static int run(const std::string& caseDir) {
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
    std::vector<scalar> diagC = M.diag, bSrc = M.source;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) for (label i = 0; i < fvp[pi].size; ++i) { const label c = fvp[pi].faceCells[i]; diagC[c] += M.internalCoeffs[pi][i]; bSrc[c] += M.boundaryCoeffs[pi][i]; }
    scalar normFactor = 0; for (label c = 0; c < nC; ++c) normFactor += std::fabs(bSrc[c]); normFactor += 1e-20;
    const std::vector<label> ownerInt(m.owner().begin(), m.owner().begin() + nIf);
    const std::vector<scalar> magSfInt(g.magSf().begin(), g.magSf().begin() + nIf);
    const scalar tol = 1e-10;

    AMGData amg = buildAMG(ownerInt, m.neighbour(), magSfInt, nC);
    DeviceLduMatrix dM = buildDeviceLdu(diagC, M.upper, M.lower, ownerInt, m.neighbour(), nC);
    amgGalerkin(amg, dM.diag, dM.upper, dM.lower);
    DeviceBuffer<scalar> db(bSrc);

    // direct-launch path
    DeviceBuffer<scalar> xDirect(std::vector<scalar>(nC, 0.0));
    const DeviceSolverPerf pD = deviceAMGPCG(dM.view(), amg, db, xDirect, normFactor, tol, 0.0, 30000, /*captureVcycle*/ false);
    // graph-replay path
    DeviceBuffer<scalar> xGraph(std::vector<scalar>(nC, 0.0));
    const DeviceSolverPerf pG = deviceAMGPCG(dM.view(), amg, db, xGraph, normFactor, tol, 0.0, 30000, /*captureVcycle*/ true);
    cudaDeviceSynchronize();

    const std::vector<scalar> a = xGraph.host(), b = xDirect.host();
    scalar n = 0, d = 0; for (label c = 0; c < nC; ++c) { n = std::fmax(n, std::fabs(a[c]-b[c])); d = std::fmax(d, std::fabs(b[c])); }
    const scalar rel = d > 0 ? n / d : n;

    // timing: full solves both ways (each builds + destroys its graph; rAU is fixed here so one capture per solve)
    const int R = 30; cudaDeviceSynchronize();
    auto t0 = Clock::now(); for (int r = 0; r < R; ++r) { DeviceBuffer<scalar> x(std::vector<scalar>(nC,0.0)); deviceAMGPCG(dM.view(), amg, db, x, normFactor, tol, 0.0, 30000, false); } cudaDeviceSynchronize();
    const double tD = ms(t0, Clock::now());
    t0 = Clock::now(); for (int r = 0; r < R; ++r) { DeviceBuffer<scalar> x(std::vector<scalar>(nC,0.0)); deviceAMGPCG(dM.view(), amg, db, x, normFactor, tol, 0.0, 30000, true); } cudaDeviceSynchronize();
    const double tG = ms(t0, Clock::now());

    std::printf("  nCells=%6d levels=%d: iters direct=%d graph=%d | solDiff %.2e | solve direct %.2f ms graph %.2f ms | %.2fx\n",
                nC, amg.nLevels(), pD.nIterations, pG.nIterations, rel, tD/R, tG/R, tD/tG);
    return (rel < 1e-12 && pD.nIterations == pG.nIterations) ? 0 : 1;
}

int main(int argc, char** argv) {
    std::printf("GPU AMG V-cycle CUDA-graph replay vs direct launch (identical result + timing):\n");
    int bad = 0;
    bad |= run(argc > 1 ? argv[1] : "validation/pitzDaily");
    if (argc > 2) bad |= run(argv[2]);
    std::printf("%s\n", bad == 0 ? "PASS" : "FAIL");
    return bad;
}
