// GPU offload (#7): thread-block clusters + DSM, the foundational brick. Validate the cluster+DSM reductions
// (dot, sumMag) against the existing global-memory deviceDot/deviceSumMag (which are validated to the CPU
// oracle), and against a host reference. The cross-block combine runs through distributed shared memory, not
// global memory, proving the cluster launch + DSM map_shared_rank + cluster.sync() mechanism is correct.
#include "device_buffer.cuh"
#include "device_blas.cuh"
#include "device_cluster.cuh"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point a, Clock::time_point b) { return std::chrono::duration<double, std::milli>(b - a).count(); }

int main() {
    if (!deviceClusterSupported()) { std::printf("cluster launch NOT supported on this device\n"); return 1; }

    for (int n : {1000, 12225, 48900, 200000}) {
        std::vector<scalar> hx(n), hy(n);
        for (int i = 0; i < n; ++i) { hx[i] = std::sin(0.001 * i + 0.3) * 2.0 - 0.5; hy[i] = std::cos(0.002 * i) * 1.5; }
        DeviceBuffer<scalar> x(hx), y(hy);

        // host references (double accumulation -> the "true" values)
        long double rdot = 0, rmag = 0; for (int i = 0; i < n; ++i) { rdot += (long double)hx[i] * hy[i]; rmag += std::fabs(hx[i]); }

        const scalar gDot = deviceDot(x, y), gMag = deviceSumMag(x);
        const scalar cDot = deviceClusterDot(x, y), cMag = deviceClusterSumMag(x);

        auto rel = [](scalar a, long double b) { return b != 0 ? std::fabs((long double)a - b) / std::fabs(b) : std::fabs((long double)a); };
        const scalar eDot = rel(cDot, rdot), eMag = rel(cMag, rmag);
        const scalar eVsG = std::fmax(rel(cDot, gDot), rel(cMag, gMag));

        std::printf("n=%7d  dot cluster %.10e vs host %.10Le (rel %.2e) | sumMag rel %.2e | vs deviceDot/SumMag %.2e\n",
                    n, cDot, rdot, eDot, eMag, eVsG);
        if (!(eDot < 1e-10 && eMag < 1e-10)) { std::printf("FAIL (n=%d)\n", n); return 1; }
    }

    // timing (informational): cluster+DSM reduction vs the global reduction at the validation mesh size
    const int n = 48900; std::vector<scalar> hx(n, 0.7), hy(n, 1.1); DeviceBuffer<scalar> x(hx), y(hy);
    const int R = 2000; volatile scalar sink = 0;
    deviceDot(x, y); deviceClusterDot(x, y); cudaDeviceSynchronize();
    auto t0 = Clock::now(); for (int r = 0; r < R; ++r) sink += deviceDot(x, y);       cudaDeviceSynchronize(); const double tG = ms(t0, Clock::now());
    t0 = Clock::now();      for (int r = 0; r < R; ++r) sink += deviceClusterDot(x, y); cudaDeviceSynchronize(); const double tC = ms(t0, Clock::now());
    std::printf("timing (n=%d, %d dots): deviceDot %.2f us/op | clusterDot %.2f us/op\n", n, R, tG/R*1e3, tC/R*1e3);

    std::printf("PASS\n");
    return 0;
}
