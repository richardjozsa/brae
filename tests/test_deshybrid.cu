// DEShybrid: the per-face blend of a low-dissipation and an upwind-biased convection scheme.
//
// OF's scheme (TurbulenceModels/schemes/DEShybrid) computes a DES sensor per cell,
//     S     = sqrt(2)*mag(symm(gradU)),  Omega = sqrt(2)*mag(skew(gradU)),  tau0 = L0/U0
//     B     = CH3*Omega*max(S,Omega) / max(0.5*(S^2+Omega^2), (OmegaLim/tau0)^2)
//     g     = tanh(B^4)
//     K     = max(sqrt(0.5*(S^2+Omega^2)), 0.1/tau0)
//     lTurb = sqrt(max((max(nut, min((Cs*delta)^2*S, nutLim*nut)) + nu)/(0.09^1.5*K), 0))
//     A     = CH2*max(0, CDES*delta/max(lTurb*g, SMALL*L0) - 0.5)
//     sigma = max(sigmaMax*tanh(A^CH1), sigmaMin)
// interpolates it to faces, and uses it to weight the two sub-schemes. CH1=3, CH2=1, CH3=2, Cs=0.18 are
// fixed in OF's constructor and are not readable from the fvSchemes entry.
//
// LEG 1 IS THE SENSOR'S DEFINING BEHAVIOUR. In IRROTATIONAL flow Omega = 0, so B = 0, g = tanh(0) = 0,
// and A blows up against the SMALL floor -- sigma saturates at sigmaMax, i.e. fully upwind. That is the
// whole design intent: no resolved turbulence, no low-dissipation scheme. It is also the one place the
// formula is analytic, so it pins the sensor without re-deriving it.
//
// Leg 1 alone is not enough, and this was checked: computing skew() as symm() leaves leg 1 passing
// (Omega picks up the strain, B is non-zero, and A still saturates) while leg 2 fails by 2.4e-01. The
// two legs cover different halves of the sensor.
//
// LEG 3 IS THE MORE USEFUL HALF. brae realises the blend as a deferred correction on top of its upwind
// matrix, so the face correction is (1-bf)*(linear - upwind) + bf*(linearUpwind - upwind). At sigma = 0
// that must reproduce brae's OWN deviceLinearCorr exactly, and at sigma = 1 its deviceLinearUpwindCorr --
// two independently validated implementations. Anything wrong with the face arithmetic, the upwind side
// selection or the interpolation of sigma breaks one of those two identities.
#include "box_mesh.cuh"
#include "device_deshybrid.cuh"
#include "device_simple.cuh"
#include "device_mesh.cuh"
#include "device_buffer.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

scalar worst(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar d = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) d = std::fmax(d, std::fabs(a[i] - b[i]));
    return d;
}
scalar peak(const std::vector<scalar>& a)
{
    scalar d = 0;
    for (scalar v : a) d = std::fmax(d, std::fabs(v));
    return d;
}

// OF's sigma, re-derived here rather than called, so leg 2 is a real comparison.
scalar sigmaHost(const std::vector<scalar>& t, scalar V, scalar nut, scalar nu, const DesHybridCoeffs& co)
{
    scalar ss = 0, ww = 0;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            const scalar s = 0.5*(t[i*3+j] + t[j*3+i]);
            const scalar w = 0.5*(t[i*3+j] - t[j*3+i]);
            ss += s*s; ww += w*w;
        }
    const scalar S = std::sqrt(2*ss), Om = std::sqrt(2*ww);
    const scalar tau0 = co.L0/co.U0, d = std::cbrt(V);
    const scalar half = 0.5*(S*S + Om*Om), oLim = co.OmegaLim/tau0;
    const scalar B = co.CH3*Om*std::fmax(S, Om)/std::fmax(half, oLim*oLim);
    const scalar g = std::tanh(B*B*B*B);
    const scalar K = std::fmax(std::sqrt(half), 0.1/tau0);
    const scalar Csd = co.Cs*d;
    const scalar nuE = std::fmax(nut, std::fmin(Csd*Csd*S, co.nutLim*nut)) + nu;
    const scalar lT = std::sqrt(std::fmax(nuE/(std::pow(0.09, 1.5)*K), 0.0));
    const scalar A = co.CH2*std::fmax(0.0, co.CDES*d/std::fmax(lT*g, 1e-15*co.L0) - 0.5);
    return std::fmax(co.sigmaMax*std::tanh(std::pow(A, co.CH1)), co.sigmaMin);
}
}   // namespace

int main()
{
    const label N = 5;
    PrimitiveMesh m = boxtest::boxMesh(N, N, N);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const int nC = (int)m.nCells();
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const DesHybridCoeffs co;   // OF/tutorial defaults except CDES, which the entry gives; 0.65 here too

    // ---- 1. irrotational flow -> sigma == sigmaMax (fully upwind) ----
    {
        // pure strain, zero rotation: a symmetric gradient
        std::vector<scalar> t(9, 0.0);
        t[0] = 3.0; t[4] = -1.0; t[8] = -2.0;
        std::vector<scalar> gh(9*nC);
        for (int q = 0; q < 9; ++q) for (int c = 0; c < nC; ++c) gh[q*nC + c] = t[q];
        // nut must sit in the sensor's ACTIVE range, or sigma saturates at 1 whatever Omega is and the
        // leg passes for any implementation. Checked: at nut = 1e-4 a version computing skew as symm
        // still returns 1 here; at nut = 0.05 it returns 8.3e-04, so the saturation below is a real
        // statement about Omega rather than about the magnitudes.
        DeviceBuffer<scalar> gradU, nut, sigma;
        gradU.copyFrom(gh);
        nut.copyFrom(std::vector<scalar>(nC, 0.05));
        deviceDesHybridSigma(nC, gradU, dm.V, nut, 1e-5, co, sigma);
        const std::vector<scalar> s = sigma.host();
        std::printf("  irrotational (Omega = 0): sigma = %.10f  (sigmaMax = %.4g)\n", (double)s[0], (double)co.sigmaMax);
        if (std::fabs(s[0] - co.sigmaMax) > 1e-12)
        {
            std::printf("  FAIL with no rotation the sensor must saturate at sigmaMax: B = 0 so g = 0, and A\n"
                        "       is then bounded only by the SMALL floor. Fully upwind is the design intent.\n");
            ++failures;
        }
    }

    // ---- 2. a general gradient: match the formula, re-derived on the host ----
    {
        std::vector<std::vector<scalar>> cells(nC, std::vector<scalar>(9, 0.0));
        for (int c = 0; c < nC; ++c)
            for (int q = 0; q < 9; ++q) cells[c][q] = 0.4*std::sin(0.7*c + 1.3*q) + 0.2*q;
        std::vector<scalar> gh(9*nC);
        for (int q = 0; q < 9; ++q) for (int c = 0; c < nC; ++c) gh[q*nC + c] = cells[c][q];
        // nut has to put lTurb*g in the neighbourhood of CDES*delta, or A saturates tanh and sigma is 1
        // everywhere -- which is what the first draft of this leg did, and its vacuity guard caught.
        // With unit cells (delta = 1) and K = O(1), lTurb ~ sqrt(nut/0.027), so nut ~ 0.05 is the knee.
        std::vector<scalar> nuth(nC);
        for (int c = 0; c < nC; ++c) nuth[c] = 0.004 + 0.30*(scalar(c)/scalar(nC));
        DeviceBuffer<scalar> gradU, nut, sigma;
        gradU.copyFrom(gh); nut.copyFrom(nuth);
        deviceDesHybridSigma(nC, gradU, dm.V, nut, 1e-5, co, sigma);
        const std::vector<scalar> got = sigma.host();
        const std::vector<scalar> Vh = dm.V.host();
        scalar w = 0, lo = 1, hi = 0;
        for (int c = 0; c < nC; ++c)
        {
            const scalar want = sigmaHost(cells[c], Vh[c], nuth[c], 1e-5, co);
            w = std::fmax(w, std::fabs(got[c] - want));
            lo = std::fmin(lo, got[c]); hi = std::fmax(hi, got[c]);
        }
        std::printf("  general gradient: max|device - host| = %.3e   (sigma spans %.4f .. %.4f)\n",
                    (double)w, (double)lo, (double)hi);
        if (w > 1e-12)
        { std::printf("  FAIL the device sensor does not reproduce OF's formula\n"); ++failures; }
        // VACUITY GUARD: if sigma were saturated everywhere the comparison would be 1 == 1.
        if (hi - lo < 1e-6)
        {
            std::printf("  FAIL vacuous: sigma is constant over the whole fixture, so leg 2 compares a\n"
                        "       single saturated value and says nothing about the formula\n");
            ++failures;
        }
    }

    // ---- 3. the blend collapses to each sub-scheme at its endpoints ----
    {
        std::vector<scalar> fh(nC), phih(dm.nInternalFaces), gxh(nC), gyh(nC), gzh(nC);
        for (int c = 0; c < nC; ++c)
        {
            fh[c]  = 2.0 + std::sin(0.8*c);
            gxh[c] = 0.5 + 0.1*c; gyh[c] = -0.3 - 0.05*c; gzh[c] = 0.2*std::cos(0.4*c);
        }
        for (int f = 0; f < dm.nInternalFaces; ++f) phih[f] = (f % 3 ? 1.0 : -1.0) * (0.2 + 0.01*f);
        DeviceBuffer<scalar> fd, phi, gx, gy, gz, sigma, des, lin, lu;
        fd.copyFrom(fh); phi.copyFrom(phih); gx.copyFrom(gxh); gy.copyFrom(gyh); gz.copyFrom(gzh);

        deviceLinearCorr(dm, phi, fd, lin);
        deviceLinearUpwindCorr(dm, phi, gx, gy, gz, lu);

        sigma.copyFrom(std::vector<scalar>(nC, 0.0));
        deviceDesHybridCorr(dm, phi, sigma, fd, gx, gy, gz, des);
        const scalar d0 = worst(des.host(), lin.host());
        sigma.copyFrom(std::vector<scalar>(nC, 1.0));
        deviceDesHybridCorr(dm, phi, sigma, fd, gx, gy, gz, des);
        const scalar d1 = worst(des.host(), lu.host());
        std::printf("  sigma = 0 vs deviceLinearCorr      : %.3e   (term peak %.3e)\n",
                    (double)d0, (double)peak(lin.host()));
        std::printf("  sigma = 1 vs deviceLinearUpwindCorr: %.3e   (term peak %.3e)\n",
                    (double)d1, (double)peak(lu.host()));
        if (d0 > 1e-12)
        { std::printf("  FAIL at sigma = 0 the blend must BE the linear correction\n"); ++failures; }
        if (d1 > 1e-12)
        { std::printf("  FAIL at sigma = 1 the blend must BE the linearUpwind correction\n"); ++failures; }
        if (peak(lin.host()) < 1e-9 || peak(lu.host()) < 1e-9)
        { std::printf("  FAIL vacuous: a reference correction is ~0, so the identities are 0 == 0\n"); ++failures; }
        // ...and the two sub-schemes must actually DIFFER, or both identities hold trivially.
        if (worst(lin.host(), lu.host()) < 1e-9)
        {
            std::printf("  FAIL vacuous: linear and linearUpwind agree on this fixture, so sigma = 0 and\n"
                        "       sigma = 1 are the same test\n");
            ++failures;
        }

        // ---- 4. and it is LINEAR in between: sigma = 0.5 is the midpoint ----
        sigma.copyFrom(std::vector<scalar>(nC, 0.5));
        deviceDesHybridCorr(dm, phi, sigma, fd, gx, gy, gz, des);
        const std::vector<scalar> h = des.host(), a = lin.host(), b = lu.host();
        scalar w = 0;
        for (int c = 0; c < nC; ++c) w = std::fmax(w, std::fabs(h[c] - 0.5*(a[c] + b[c])));
        std::printf("  sigma = 0.5 vs the midpoint        : %.3e\n", (double)w);
        if (w > 1e-12)
        { std::printf("  FAIL the blend is not linear in sigma\n"); ++failures; }
    }

    std::printf("deshybrid: %d failures\n", failures);
    return failures ? 1 : 0;
}
