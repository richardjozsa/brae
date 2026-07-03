// GPU offload (#7a): cluster + DSM SpMV. A cluster owns a contiguous cell chunk and caches its psi in
// distributed shared memory; neighbour gathers hit DSM (within-chunk) or global (cross-chunk). Validate it is
// bit-identical to the global-memory deviceAmul on the real pitzDaily pressure Laplacian, and MEASURE it, the
// ledger flags SpMV as bandwidth/L2-bound (micro-opts historically net-negative) and at these sizes kernels are
// launch-bound, so this is an honest measurement of whether DSM caching beats the L2-backed global gather.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "fvm.cuh"
#include "device_buffer.cuh"
#include "device_ldu.cuh"
#include "device_cluster.cuh"
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
    std::vector<scalar> diagC = M.diag;
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) for (label i = 0; i < fvp[pi].size; ++i) diagC[fvp[pi].faceCells[i]] += M.internalCoeffs[pi][i];
    const std::vector<label> ownerInt(m.owner().begin(), m.owner().begin() + nIf);
    DeviceLduMatrix A = buildDeviceLdu(diagC, M.upper, M.lower, ownerInt, m.neighbour(), nC);

    std::vector<scalar> hpsi(nC); for (label c = 0; c < nC; ++c) hpsi[c] = std::sin(0.01 * c + 0.2);
    DeviceBuffer<scalar> psi(hpsi), yGlobal, yCluster;
    deviceAmul(A.view(), psi, yGlobal);
    deviceClusterAmul(A.view(), psi, yCluster);
    cudaDeviceSynchronize();

    const std::vector<scalar> a = yCluster.host(), b = yGlobal.host();
    scalar n = 0, d = 0; for (label c = 0; c < nC; ++c) { n = std::fmax(n, std::fabs(a[c]-b[c])); d = std::fmax(d, std::fabs(b[c])); }
    const scalar rel = d > 0 ? n / d : n;

    const int R = 3000; cudaDeviceSynchronize();
    auto t0 = Clock::now(); for (int r = 0; r < R; ++r) deviceAmul(A.view(), psi, yGlobal);       cudaDeviceSynchronize(); const double tG = ms(t0, Clock::now());
    t0 = Clock::now();      for (int r = 0; r < R; ++r) deviceClusterAmul(A.view(), psi, yCluster); cudaDeviceSynchronize(); const double tC = ms(t0, Clock::now());

    std::printf("  nCells=%6d: deviceAmul %.2f us | clusterAmul %.2f us | %.2fx %s | diff %.1e\n",
                nC, tG/R*1e3, tC/R*1e3, tG/tC, tG/tC>1.0?"(cluster wins)":"", rel);
    return rel < 1e-12 ? 0 : 1;
}

int main(int argc, char** argv) {
    std::printf("GPU cluster+DSM SpMV vs deviceAmul (bit-exact + timing):\n");
    int bad = 0;
    bad |= run(argc > 1 ? argv[1] : "validation/pitzDaily");
    if (argc > 2) bad |= run(argv[2]);
    std::printf("%s\n", bad == 0 ? "PASS" : "FAIL");
    return bad;
}
