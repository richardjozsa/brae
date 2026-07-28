// NVRTC device-coded-BC core: compile a per-face snippet at runtime (nvrtc -> PTX -> driver-API load) and launch it on
// the GPU, verifying the exposed per-face API (position fx/fy/fz, time t, adjacent-cell U cx/cy/cz) and the offset
// indexing produce exactly the host-computed values. Proves the coded BC is fully device-resident (no host round-trip).
#include "device_coded_bc.cuh"
#include "device_buffer.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

int main()
{
    int fails = 0;
    auto chk = [&](const char* nm, double got, double exp, double tol) {
        const bool ok = std::fabs(got - exp) <= tol*std::fmax(1.0, std::fabs(exp));
        if (!ok) ++fails;
        std::printf("  %-40s got %.10g  exp %.10g  %s\n", nm, got, exp, ok ? "OK" : "FAIL");
    };
    auto flag = [&](const char* nm, bool ok) { if (!ok) ++fails; std::printf("  %-40s %s\n", nm, ok ? "OK" : "FAIL"); };

    const int nb = 6;   // boundary faces; faceCell = identity so the adjacent-cell U is indexed by face
    std::vector<scalar> cfx = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
    std::vector<scalar> cfy = {0.01, -0.02, 0.03, -0.04, 0.05, -0.06};
    std::vector<scalar> cfz(nb, 0.7);
    std::vector<scalar> ux(nb, 1.0);
    std::vector<scalar> uy = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6};
    std::vector<scalar> uz(nb, 0.0);
    std::vector<label>  fc(nb); for (int i = 0; i < nb; ++i) fc[i] = i;

    DeviceBuffer<scalar> CfX, CfY, CfZ, Ux, Uy, Uz, refX, refY, refZ;
    DeviceBuffer<label>  faceCell;
    CfX.copyFrom(cfx); CfY.copyFrom(cfy); CfZ.copyFrom(cfz);
    Ux.copyFrom(ux); Uy.copyFrom(uy); Uz.copyFrom(uz);
    faceCell.copyFrom(fc);
    const scalar SENT = -999.0;
    refX.copyFrom(std::vector<scalar>(nb, SENT));
    refY.copyFrom(std::vector<scalar>(nb, SENT));
    refZ.copyFrom(std::vector<scalar>(nb, SENT));

    // A snippet exercising position (fx), time (t), and the adjacent-cell U (cy): out = (2*fx + t, cy, fz).
    std::printf("NVRTC compile of a per-face coded-BC snippet:\n");
    CodedBcKernel k;
    try { k = compileCodedVectorBc("unit", "out = make3(2.0*fx + t, cy, fz);"); }
    catch (const std::exception& e) { std::printf("  compile threw: %s\n", e.what()); return 1; }
    flag("compiled + module loaded (k.valid())", k.valid());

    const scalar t = 3.0;
    launchCodedVectorBc(k, 0, nb, t, CfX, CfY, CfZ, Ux, Uy, Uz, faceCell, refX, refY, refZ);
    cudaDeviceSynchronize();
    auto rx = refX.host(), ry = refY.host(), rz = refZ.host();
    std::printf("device kernel output vs host formula (all %d faces, offset 0):\n", nb);
    for (int i = 0; i < nb; ++i) {
        char nm[40];
        std::snprintf(nm, sizeof nm, "face %d refX = 2*fx+t", i); chk(nm, rx[i], 2.0*cfx[i] + t, 1e-12);
        std::snprintf(nm, sizeof nm, "face %d refY = cy (adj U_y)", i); chk(nm, ry[i], uy[i], 1e-12);
        std::snprintf(nm, sizeof nm, "face %d refZ = fz", i); chk(nm, rz[i], cfz[i], 1e-12);
    }

    // offset indexing: launch over [2, 2+3) -> writes only faces 2,3,4; faces 0,1,5 keep the sentinel.
    refX.copyFrom(std::vector<scalar>(nb, SENT));
    launchCodedVectorBc(k, 2, 3, t, CfX, CfY, CfZ, Ux, Uy, Uz, faceCell, refX, refY, refZ);
    cudaDeviceSynchronize();
    rx = refX.host();
    std::printf("offset/range indexing (launch [2,5), rest stays sentinel):\n");
    chk("face 2 written", rx[2], 2.0*cfx[2] + t, 1e-12);
    chk("face 4 written", rx[4], 2.0*cfx[4] + t, 1e-12);
    chk("face 0 untouched (sentinel)", rx[0], SENT, 1e-12);
    chk("face 5 untouched (sentinel)", rx[5], SENT, 1e-12);

    // a bad snippet must throw (the NVRTC compile-log path) so users get a real error, not silent wrong results.
    bool threw = false;
    try { compileCodedVectorBc("bad", "out = this_is_not_valid_cuda;"); } catch (const std::exception&) { threw = true; }
    flag("bad snippet throws NVRTC compile error", threw);

    std::printf("\n%s: %d failure(s)\n", fails ? "FAIL" : "PASS", fails);
    return fails ? 1 : 0;
}
