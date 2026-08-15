// Uf, the face velocity a moving-mesh pimpleFoam carries beside phi, and the reason it cannot be
// shortcut on a mesh whose face areas change.
//
// fvc::correctUf (fvcMeshPhi.C):
//     Uf  = fvc::interpolate(U);
//     n   = Sf/magSf;
//     Uf += n*(phi/magSf - (n & Uf));
// and EulerDdtScheme::fvcDdtUfCorr then reads  phiUf0 = mesh.Sf() & Uf.oldTime().
//
// THE DEFINING PROPERTY (leg 1): after correctUf, Sf & Uf == phi, exactly. That is the whole point of
// the normal-component replacement -- the face velocity is made to carry the conservative flux, so the
// tangential part comes from the interpolation and the normal part from phi. Everything else about Uf
// follows from it.
//
// WHY IT IS NOT phi + mesh.phi() (leg 2). brae used to reconstruct phiUf0 as phi.oldTime() + mesh.phi(),
// which is exact -- verified against OpenFOAM to 2.7e-10 -- for as long as Sf is the SAME at both time
// levels. OF dots the CURRENT Sf into the STORED Uf, so the moment a face's area changes the two part
// company. Leg 2 is that divergence, at a scale that matters: a cyclicACMI rescales its coupled areas
// from the overlap mask every step, and on the moving oscillatingInletACMI2D the shortcut was exact at
// step 1 and 5.1e-03 out by step 10, all of it on the interface columns. Carrying the real Uf took that
// case to 9.7e-07 against stock OpenFOAM.
//
// Leg 3 is the tangential half: correctUf must leave the tangential velocity ALONE. A version that just
// set Uf = n*phi/magSf would satisfy leg 1 perfectly and be wrong.
#include "device_ddt.cuh"
#include "device_buffer.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

void near(const char* what, scalar got, scalar want, scalar tol)
{
    if (std::fabs(got - want) <= tol) return;
    std::printf("  FAIL %s: got %.12g, want %.12g (tol %.1e)\n", what, (double)got, (double)want, (double)tol);
    ++failures;
}

}   // namespace

int main()
{
    // Three faces with deliberately awkward, non-axis-aligned normals, so a formula that only worked for
    // a Cartesian face would not survive. magSf is |Sf| for each.
    const std::vector<scalar> Sfx = { 3.0, -1.0,  0.6};
    const std::vector<scalar> Sfy = { 4.0,  2.0, -0.8};
    const std::vector<scalar> Sfz = { 0.0,  2.0,  0.0};
    std::vector<scalar> magSf(3);
    for (int f = 0; f < 3; ++f) magSf[f] = std::sqrt(Sfx[f]*Sfx[f] + Sfy[f]*Sfy[f] + Sfz[f]*Sfz[f]);

    // interp(U) going in, and a flux that deliberately does NOT match it (that is the interesting case:
    // phi carries the Rhie-Chow correction the interpolation does not know about).
    const std::vector<scalar> ux0 = { 1.0,  0.5, -2.0};
    const std::vector<scalar> uy0 = {-0.5,  1.5,  1.0};
    const std::vector<scalar> uz0 = { 0.25, 0.0,  3.0};
    const std::vector<scalar> phi = { 7.0, -3.0,  0.9};

    DeviceBuffer<scalar> dSfx, dSfy, dSfz, dMag, dPhi, ufx, ufy, ufz;
    dSfx.copyFrom(Sfx); dSfy.copyFrom(Sfy); dSfz.copyFrom(Sfz);
    dMag.copyFrom(magSf); dPhi.copyFrom(phi);
    ufx.copyFrom(ux0); ufy.copyFrom(uy0); ufz.copyFrom(uz0);

    deviceCorrectUf(3, nullptr, dSfx, dSfy, dSfz, dMag, dPhi, ufx, ufy, ufz);
    const std::vector<scalar> fx = ufx.host(), fy = ufy.host(), fz = ufz.host();

    // ---- 1. Sf & Uf == phi, exactly ----
    {
        DeviceBuffer<scalar> dot;
        deviceDotSf(3, nullptr, dSfx, dSfy, dSfz, ufx, ufy, ufz, dot);
        const std::vector<scalar> d = dot.host();
        for (int f = 0; f < 3; ++f)
        {
            std::printf("  face %d: Sf & Uf = %+.12g   phi = %+.12g\n", f, (double)d[f], (double)phi[f]);
            near("Sf & Uf == phi", d[f], phi[f], scalar(1e-12));
        }
        // VACUITY GUARD: if the input already satisfied it, correctUf could be a no-op and still pass.
        scalar worstIn = 0;
        for (int f = 0; f < 3; ++f)
            worstIn = std::fmax(worstIn, std::fabs(Sfx[f]*ux0[f] + Sfy[f]*uy0[f] + Sfz[f]*uz0[f] - phi[f]));
        if (worstIn < scalar(1e-3))
        {
            std::printf("  FAIL vacuous: interp(U) already carried phi (worst %.3e), so correctUf had\n"
                        "       nothing to correct and doing nothing would pass\n", (double)worstIn);
            ++failures;
        }
        else std::printf("  (input was off by up to %.3g, so the correction is doing real work)\n", (double)worstIn);
    }

    // ---- 3. the TANGENTIAL component is untouched ----
    for (int f = 0; f < 3; ++f)
    {
        const scalar nx = Sfx[f]/magSf[f], ny = Sfy[f]/magSf[f], nz = Sfz[f]/magSf[f];
        const scalar dotIn  = nx*ux0[f] + ny*uy0[f] + nz*uz0[f];
        const scalar dotOut = nx*fx[f]  + ny*fy[f]  + nz*fz[f];
        // tangential part = u - n(n.u); it must be identical before and after
        const scalar tInX = ux0[f] - nx*dotIn,  tOutX = fx[f] - nx*dotOut;
        const scalar tInY = uy0[f] - ny*dotIn,  tOutY = fy[f] - ny*dotOut;
        const scalar tInZ = uz0[f] - nz*dotIn,  tOutZ = fz[f] - nz*dotOut;
        near("tangential x preserved", tOutX, tInX, scalar(1e-12));
        near("tangential y preserved", tOutY, tInY, scalar(1e-12));
        near("tangential z preserved", tOutZ, tInZ, scalar(1e-12));
    }
    std::printf("  tangential velocity preserved on all 3 faces (only the normal part is replaced)\n");

    // ---- 2. the shortcut breaks when Sf changes; Sf & Uf does not ----
    // Same stored Uf, but the face area has since changed -- an ACMI mask rescaling its coupled areas is
    // exactly this. phiUf0 must follow the NEW area; phi.oldTime() (fixed at the old one) cannot.
    {
        const scalar scale = 0.4;   // the coupled area shrank to 40%: a partially uncovered ACMI face
        std::vector<scalar> Sfx2(3), Sfy2(3), Sfz2(3);
        for (int f = 0; f < 3; ++f) { Sfx2[f] = scale*Sfx[f]; Sfy2[f] = scale*Sfy[f]; Sfz2[f] = scale*Sfz[f]; }
        DeviceBuffer<scalar> d2x, d2y, d2z, dot2;
        d2x.copyFrom(Sfx2); d2y.copyFrom(Sfy2); d2z.copyFrom(Sfz2);
        deviceDotSf(3, nullptr, d2x, d2y, d2z, ufx, ufy, ufz, dot2);
        const std::vector<scalar> d = dot2.host();
        for (int f = 0; f < 3; ++f)
        {
            near("phiUf0 follows the CURRENT area", d[f], scale*phi[f], scalar(1e-12));
            // ...and that is NOT what the old flux says, which is the whole reason Uf is carried
            if (std::fabs(d[f] - phi[f]) < scalar(1e-6))
            {
                std::printf("  FAIL face %d: phiUf0 came back equal to the OLD flux (%.6g); with the area\n"
                            "       changed it must not, or the shortcut would have been sufficient\n",
                            f, (double)phi[f]);
                ++failures;
            }
        }
        std::printf("  area x%.2g: phiUf0 tracks it (%.6g vs the old flux %.6g on face 0)\n",
                    (double)scale, (double)d[0], (double)phi[0]);
    }

    // ---- a zero-area face must be left alone, not divided by ----
    {
        DeviceBuffer<scalar> zx, zy, zz, zm, zp, ax, ay, az;
        zx.copyFrom(std::vector<scalar>{0.0}); zy.copyFrom(std::vector<scalar>{0.0});
        zz.copyFrom(std::vector<scalar>{0.0}); zm.copyFrom(std::vector<scalar>{0.0});
        zp.copyFrom(std::vector<scalar>{0.0});
        ax.copyFrom(std::vector<scalar>{1.5}); ay.copyFrom(std::vector<scalar>{-2.5}); az.copyFrom(std::vector<scalar>{0.5});
        deviceCorrectUf(1, nullptr, zx, zy, zz, zm, zp, ax, ay, az);
        const scalar gx = ax.host()[0], gy = ay.host()[0], gz = az.host()[0];
        std::printf("  zero-area face: Uf = (%.4g %.4g %.4g), unchanged and finite\n",
                    (double)gx, (double)gy, (double)gz);
        near("zero-area face x untouched", gx, scalar(1.5), scalar(0));
        near("zero-area face y untouched", gy, scalar(-2.5), scalar(0));
        if (!std::isfinite((double)gx) || !std::isfinite((double)gy) || !std::isfinite((double)gz))
        { std::printf("  FAIL zero-area face produced a non-finite Uf (divided by magSf)\n"); ++failures; }
    }

    std::printf("correct_uf: %d failures\n", failures);
    return failures ? 1 : 0;
}
