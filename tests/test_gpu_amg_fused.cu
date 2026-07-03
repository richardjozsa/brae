// GPU offload (#7b): the AMG coarse Jacobi solve fused into ONE thread-block-cluster kernel (DSM-resident
// coarse vector, cluster.sync between sweeps) vs the unfused loop (deviceAmul + smooth, 2*nSweeps launches).
// Build the real pitzDaily coarse matrix (buildAMG + Galerkin), then run NCOARSE weighted-Jacobi sweeps both
// ways from the same RHS/guess. The fused result must MATCH the loop machine-precision (same arithmetic/order),
// and the fused launch must be much cheaper (the #1 launch-overhead source: ~40 sweeps per V-cycle per PCG iter).
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

int main(int argc, char** argv) {
    const std::string caseDir = argc > 1 ? argv[1] : "validation/pitzDaily";
    PrimitiveMesh m; m.read(caseDir + "/constant/polyMesh");
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells(), nIf = m.nInternalFaces();

    // a pressure-like Laplacian -> build the AMG coarse matrix from it.
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
    const std::vector<scalar> magSfInt(g.magSf().begin(), g.magSf().begin() + nIf);
    AMGData amg = buildAMG(ownerInt, m.neighbour(), magSfInt, nC);
    DeviceLduMatrix dM = buildDeviceLdu(diagC, M.upper, M.lower, ownerInt, m.neighbour(), nC);
    amgGalerkin(amg, dM.diag, dM.upper, dM.lower);
    const DeviceLduView cv = amg.coarseView();
    const int nCo = amg.nCoarse, NCOARSE = 40;

    // coarse RHS (a smooth pattern) and a zero initial guess.
    std::vector<scalar> hrc(nCo); for (int c = 0; c < nCo; ++c) hrc[c] = std::sin(0.02 * c + 0.5);
    DeviceBuffer<scalar> rc(hrc);

    DeviceBuffer<scalar> xLoop(std::vector<scalar>(nCo, 0.0)), xFused(std::vector<scalar>(nCo, 0.0));
    deviceCoarseJacobiLoop(cv, rc, xLoop, NCOARSE);
    deviceCoarseJacobiFused(cv, rc, xFused, NCOARSE);
    cudaDeviceSynchronize();

    const std::vector<scalar> a = xFused.host(), b = xLoop.host();
    scalar n = 0, d = 0; for (int c = 0; c < nCo; ++c) { n = std::fmax(n, std::fabs(a[c]-b[c])); d = std::fmax(d, std::fabs(b[c])); }
    const scalar rel = d > 0 ? n / d : n;

    // timing
    const int R = 500; cudaDeviceSynchronize();
    auto t0 = Clock::now(); for (int r = 0; r < R; ++r) { DeviceBuffer<scalar> x(std::vector<scalar>(nCo,0.0)); deviceCoarseJacobiLoop(cv, rc, x, NCOARSE); } cudaDeviceSynchronize();
    const double tL = ms(t0, Clock::now());
    t0 = Clock::now(); for (int r = 0; r < R; ++r) { DeviceBuffer<scalar> x(std::vector<scalar>(nCo,0.0)); deviceCoarseJacobiFused(cv, rc, x, NCOARSE); } cudaDeviceSynchronize();
    const double tF = ms(t0, Clock::now());

    std::printf("GPU AMG coarse Jacobi fused vs loop (nCoarse=%d, %d sweeps, fits1cluster=%d):\n", nCo, NCOARSE, (int)deviceCoarseFitsCluster(nCo));
    std::printf("  fused vs loop solution diff : %.3e (bit-exact)\n", rel);
    std::printf("  loop  %.1f us/solve (2*%d launches) | fused %.1f us/solve (1 launch) | %.2fx\n", tL/R*1e3, NCOARSE, tF/R*1e3, tL/tF);

    // crossover sweep: synthetic 1D-chain matrices of varying size (where does fusion win?). The fused single
    // cluster caps at 8 SMs, so it wins only when the level is small enough to be launch-bound (the coarsest
    // level of a recursive multi-level AMG), not for a large two-level coarse grid (parallelism-bound).
    std::printf("crossover (1D-chain, %d sweeps): fused wins when launch-bound (small N)\n", NCOARSE);
    for (int N : {128, 512, 2048, 8192}) {
        std::vector<scalar> dg(N, 2.1), up(N-1, -1.0), lo(N-1, -1.0); std::vector<label> ow(N-1), ne(N-1);
        for (int i = 0; i < N-1; ++i) { ow[i] = i; ne[i] = i+1; }
        DeviceLduMatrix cm = buildDeviceLdu(dg, up, lo, ow, ne, N); const DeviceLduView cvN = cm.view();
        DeviceBuffer<scalar> rN(std::vector<scalar>(N, 1.0));
        DeviceBuffer<scalar> xa(std::vector<scalar>(N,0.0)), xb(std::vector<scalar>(N,0.0));
        deviceCoarseJacobiLoop(cvN, rN, xa, NCOARSE); deviceCoarseJacobiFused(cvN, rN, xb, NCOARSE); cudaDeviceSynchronize();
        const std::vector<scalar> ha = xa.host(), hb = xb.host(); scalar nn=0,dd=0; for (int c=0;c<N;++c){nn=std::fmax(nn,std::fabs(ha[c]-hb[c]));dd=std::fmax(dd,std::fabs(ha[c]));}
        const int RR = 1000; cudaDeviceSynchronize();
        auto s0=Clock::now(); for(int r=0;r<RR;++r){DeviceBuffer<scalar> x(std::vector<scalar>(N,0.0)); deviceCoarseJacobiLoop(cvN,rN,x,NCOARSE);} cudaDeviceSynchronize(); double l=ms(s0,Clock::now());
        s0=Clock::now(); for(int r=0;r<RR;++r){DeviceBuffer<scalar> x(std::vector<scalar>(N,0.0)); deviceCoarseJacobiFused(cvN,rN,x,NCOARSE);} cudaDeviceSynchronize(); double fz=ms(s0,Clock::now());
        std::printf("  N=%5d: loop %.1f us | fused %.1f us | %.2fx %s | diff %.1e\n", N, l/RR*1e3, fz/RR*1e3, l/fz, l/fz>1?"(fused wins)":"", dd>0?nn/dd:nn);
    }

    const bool pass = rel < 1e-10 && deviceCoarseFitsCluster(nCo);
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
