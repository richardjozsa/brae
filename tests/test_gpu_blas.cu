// GPU offload G0: the device foundation. Validate DeviceBuffer round-trip and the BLAS-1 device kernels
// (axpy / scale / dot) against the CPU baseline (the oracle). Elementwise ops match to ~machine epsilon
// (FMA), the reduction to machine precision (FP non-associativity).
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_blas.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

int main() {
    const int n = 100000;
    std::vector<scalar> x(n), y(n);
    for (int i = 0; i < n; ++i) { x[i] = std::sin(0.3 * i + 1.0); y[i] = std::cos(0.17 * i) - 0.5; }
    const scalar a = 1.37;

    // (0) DeviceBuffer H2D -> D2H round-trip is exact.
    DeviceBuffer<scalar> dx(x), dy(y);
    const std::vector<scalar> xback = dx.host();
    scalar maxRt = 0; for (int i = 0; i < n; ++i) maxRt = std::fmax(maxRt, std::fabs(xback[i] - x[i]));

    // (1) axpy: y += a*x.
    std::vector<scalar> yref = y; for (int i = 0; i < n; ++i) yref[i] += a * x[i];
    deviceAxpy(a, dx, dy);
    const std::vector<scalar> ygpu = dy.host();
    scalar nA = 0, dA = 0; for (int i = 0; i < n; ++i) { nA = std::fmax(nA, std::fabs(ygpu[i] - yref[i])); dA = std::fmax(dA, std::fabs(yref[i])); }
    const scalar axpyErr = nA / dA;

    // (2) dot.
    scalar dref = 0; for (int i = 0; i < n; ++i) dref += x[i] * x[i];
    const scalar dgpu = deviceDot(dx, dx);
    const scalar dotErr = std::fabs(dgpu - dref) / std::fabs(dref);

    // (3) scale: x *= a.
    std::vector<scalar> xref = x; for (int i = 0; i < n; ++i) xref[i] *= a;
    deviceScale(dx, a);
    const std::vector<scalar> xs = dx.host();
    scalar nS = 0, dS = 0; for (int i = 0; i < n; ++i) { nS = std::fmax(nS, std::fabs(xs[i] - xref[i])); dS = std::fmax(dS, std::fabs(xref[i])); }
    const scalar scaleErr = nS / dS;

    std::printf("GPU G0 BLAS-1 (n=%d):\n", n);
    std::printf("  (0) DeviceBuffer round-trip : %.3e\n", maxRt);
    std::printf("  (1) axpy  vs CPU            : %.3e\n", axpyErr);
    std::printf("  (2) dot   vs CPU            : %.3e\n", dotErr);
    std::printf("  (3) scale vs CPU           : %.3e\n", scaleErr);
    const bool pass = maxRt == 0.0 && axpyErr < 1e-14 && dotErr < 1e-13 && scaleErr < 1e-15;
    std::printf("%s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
