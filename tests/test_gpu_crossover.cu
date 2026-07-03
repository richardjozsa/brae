// GPU offload (#8 re-measure): the GPU-vs-CPU crossover WITH all the #6/#7 wins (multi-level AMG + cached V-cycle
// graph). Compares the device-resident DeviceSimpleSolver against the from-source CPU runSimple oracle on the SAME
// faithful laminar SIMPLE algorithm. GB10 has large clock/thermal variance, so this controls for it: WARM UP first
// (clocks to steady state + graph captured), time several repeats and take the MIN (warm/peak-clock state; the slow
// runs are cold/throttled outliers), and run CPU+GPU back-to-back in one process. Reports ms/step and the ratio.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "simple_foam.cuh"            // CPU oracle: runSimple
#include "device_simple_foam.cuh"    // device-resident solver (multi-level AMG + cached graph)
#include <chrono>
#include <cmath>
#include <cstdio>
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
    const label nC = m.nCells();
    const int N = 12, W = 2, R = 3;       // N timed steps, W warm-up steps, R repeats (take min)
    const scalar nu = 1e-2;
    const FieldData<vector> Ufd = readField<vector>(caseDir + "/0/U");
    const FieldData<scalar> pfd = readField<scalar>(caseDir + "/0/p");

    // CPU: from-source runSimple (faithful laminar SIMPLE, gamg pressure).
    double cpuBest = 1e300; std::vector<vector> Ucpu; std::vector<scalar> pcpu;
    for (int r = 0; r < R; ++r) {
        GeometricField<vector> U = buildField<vector>(Ufd, fvp, nC); U.evaluateBoundary();
        GeometricField<scalar> p = buildField<scalar>(pfd, fvp, nC); p.evaluateBoundary();
        SurfaceScalarField phi = fvc::flux(U, m, g, fvp);
        SimpleControls ctl; ctl.nu = nu; ctl.relaxU = 0.7; ctl.relaxP = 0.3; ctl.tolU = 1e-6; ctl.tolP = 1e-6; ctl.maxIter = 5000;
        runSimple(U, p, phi, Ufd, m, g, fvp, ctl, W, 0.0);          // warm
        const auto t0 = Clock::now(); runSimple(U, p, phi, Ufd, m, g, fvp, ctl, N, 0.0); const double e = ms(t0, Clock::now());
        cpuBest = std::fmin(cpuBest, e / N);
        if (r == R-1) { Ucpu = U.internal; pcpu = p.internal; }
    }

    // GPU: device-resident solver (multi-level AMG + cached graph).
    double gpuBest = 1e300; std::vector<vector> Ugpu; std::vector<scalar> pgpu;
    for (int r = 0; r < R; ++r) {
        GeometricField<vector> U = buildField<vector>(Ufd, fvp, nC); U.evaluateBoundary();
        GeometricField<scalar> p = buildField<scalar>(pfd, fvp, nC); p.evaluateBoundary();
        SurfaceScalarField phi = fvc::flux(U, m, g, fvp);
        DeviceSimpleControls ctl; ctl.nu = nu; ctl.relaxU = 0.7; ctl.relaxP = 0.3; ctl.tolU = 1e-6; ctl.tolP = 1e-6; ctl.turbulent = false;
        DeviceSimpleSolver s(m, g, fvp, U, p, phi, ctl);
        for (int it = 0; it < W; ++it) s.step();                    // warm (captures graph, raises clocks)
        cudaDeviceSynchronize();
        const auto t0 = Clock::now(); for (int it = 0; it < N; ++it) s.step(); cudaDeviceSynchronize(); const double e = ms(t0, Clock::now());
        gpuBest = std::fmin(gpuBest, e / N);
        if (r == R-1) { Ugpu = s.U(); pgpu = s.p(); }
    }

    scalar nU=0,dU=0; for (label c=0;c<nC;++c){ nU=std::fmax(nU,mag(Ugpu[c]-Ucpu[c])); dU=std::fmax(dU,mag(Ucpu[c])); }
    const scalar eU = dU>0?nU/dU:nU;
    std::printf("  nCells=%6d: CPU %.2f ms/step | GPU %.2f ms/step | GPU %s %.2fx | fieldU diff %.1e\n",
                nC, cpuBest, gpuBest, cpuBest/gpuBest>=1?"WINS":"loses", cpuBest/gpuBest, eU);
    return eU < 5e-3 ? 0 : 1;          // same SIMPLE algorithm -> fields agree (loose: independent solvers/iters)
}

int main(int argc, char** argv) {
    std::printf("GPU-vs-CPU crossover (warm + min-of-repeats, all #6/#7 wins; laminar SIMPLE):\n");
    int bad = 0;
    bad |= run(argc > 1 ? argv[1] : "validation/pitzDaily");
    if (argc > 2) bad |= run(argv[2]);
    std::printf("%s\n", bad == 0 ? "PASS" : "FAIL");
    return bad;
}
