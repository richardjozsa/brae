// GPU offload (#7c-loop): the cached V-cycle CUDA graph across SIMPLE steps. The pressure matrix buffers are now
// persistent in DeviceSimpleSolver, so the graph is captured ONCE (first step) and replayed every step. Run the
// device-resident loop both ways (useGraph on/off) on a real case, validate the fields are identical, and MEASURE
// the loop wall-clock A/B in one process (reliable, avoids GB10 clock variance across separate runs).
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "device_simple_foam.cuh"
#include <chrono>
#include <cmath>
#include <cstdio>
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
    const label nC = m.nCells();
    const int N = 80;
    const scalar nu = 1e-2;

    const FieldData<vector> Ufd = readField<vector>(caseDir + "/0/U");
    const FieldData<scalar> pfd = readField<scalar>(caseDir + "/0/p");

    auto runLoop = [&](bool useGraph, std::vector<vector>& U, std::vector<scalar>& p, double& tms) {
        GeometricField<vector> U0 = buildField<vector>(Ufd, fvp, nC); U0.evaluateBoundary();
        GeometricField<scalar> p0 = buildField<scalar>(pfd, fvp, nC); p0.evaluateBoundary();
        SurfaceScalarField phi = fvc::flux(U0, m, g, fvp);
        DeviceSimpleControls ctl; ctl.nu = nu; ctl.relaxU = 0.7; ctl.relaxP = 0.3; ctl.tolU = 1e-6; ctl.tolP = 1e-6;
        ctl.turbulent = false; ctl.useGraph = useGraph;
        DeviceSimpleSolver s(m, g, fvp, U0, p0, phi, ctl);
        s.step();                                              // warm up (captures the graph on step 1)
        cudaDeviceSynchronize();
        const auto t0 = Clock::now();
        for (int it = 1; it < N; ++it) s.step();
        cudaDeviceSynchronize();
        tms = ms(t0, Clock::now());
        U = s.U(); p = s.p();
    };

    std::vector<vector> Ua, Ub; std::vector<scalar> pa, pb; double tOff = 0, tOn = 0;
    runLoop(false, Ua, pa, tOff);
    runLoop(true,  Ub, pb, tOn);

    scalar nU = 0, dU = 0, nP = 0, dP = 0;
    for (label c = 0; c < nC; ++c) { nU = std::fmax(nU, mag(Ub[c]-Ua[c])); dU = std::fmax(dU, mag(Ua[c]));
                                     nP = std::fmax(nP, std::fabs(pb[c]-pa[c])); dP = std::fmax(dP, std::fabs(pa[c])); }
    const scalar eU = dU>0?nU/dU:nU, eP = dP>0?nP/dP:nP;

    std::printf("GPU resident loop, cached V-cycle graph vs direct (%d steps, nCells=%d):\n", N, nC);
    std::printf("  field diff graph-vs-direct: U %.2e  p %.2e\n", eU, eP);
    std::printf("  loop time: direct %.1f ms (%.2f ms/step) | graph %.1f ms (%.2f ms/step) | %.2fx\n",
                tOff, tOff/(N-1), tOn, tOn/(N-1), tOff/tOn);
    const bool pass = eU < 1e-9 && eP < 1e-9;                  // graph replay is the same kernels -> machine-precision
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
