// div(phi,U) Gauss limitedLinearV k -- OpenFOAM's VECTOR (NVDVTVDV) limited-linear convection, which the
// four pimpleFoam TJunction tutorials all name.
//
// OF LimitedScheme<vector, limitedLinearLimiter<NVDVTVDV>, limitFuncs::null>:
//     gradfV = U[N] - U[P];   gradf = gradfV & gradfV
//     gradcf = gradfV & (d & gradU[upwind]),   d = C[N] - C[P],  upwind strict on faceFlux > 0
//     r      = |gradcf| >= 1000|gradf| ? 2*1000*sign(gradcf)*sign(gradf) - 1 : 2*gradcf/gradf - 1
//     limiter = max(min(2/k * r, 1), 0)
//     w      = limiter*CDweight + (1 - limiter)*pos0(faceFlux)
// then fvmDiv: lower = -w*flux, upper = lower + flux, diag = -sum(off-diag).
// ONE limiter per face for the whole vector -- that is the entire difference from plain limitedLinear,
// which would reduce U to magSqr(U) first and get a different number.
//
// THE TWO LIMITS ARE THE REAL TEST. limiter is not a free parameter: it is pinned at both ends.
//   Leg 2: a LINEAR velocity field with its exact gradient gives gradcf == gradf exactly, so r == 1,
//          limiter == 1, and the scheme must collapse to plain central differencing -- bit for bit
//          against deviceDivCentralCoeffs, which knows nothing about limiters.
//   Leg 3: a zero gradient gives r == -1, limiter == 0, and it must collapse to pure upwind, again bit
//          for bit against a separate function.
// Leg 4 then shows a real field lands strictly BETWEEN those two, so legs 2 and 3 are not passing
// because the code has degenerated into one of them.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "solver_controls.cuh"
#include "scheme_parse.cuh"
#include "box_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       std::printf("  ok:   %s\n", what);
}

scalar maxAbs(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar e = 0;
    for (std::size_t i = 0; i < a.size(); ++i) e = std::max(e, std::fabs(a[i] - b[i]));
    return e;
}

// OF's formula, written straight from LimitedScheme/NVDVTVDV/limitedLinearLimiter, on the host.
void refCoeffs(
    const PrimitiveMesh& m, const FvGeometry& g,
    const std::vector<scalar>& w,            // CD weights (mesh)
    const std::vector<scalar>& phi,
    const std::vector<scalar> U[3],
    const std::vector<scalar>& gradU,        // packed, gradU[q*nC + c], q = 3i + j
    scalar twoByk,
    std::vector<scalar>& upper, std::vector<scalar>& lower, std::vector<scalar>& diag)
{
    const label nIf = m.nInternalFaces(), nC = m.nCells();
    const std::vector<vector>& C = g.C();
    upper.assign(nIf, 0.0); lower.assign(nIf, 0.0); diag.assign(nC, 0.0);
    for (label f = 0; f < nIf; ++f)
    {
        const label P = m.owner()[f], N = m.neighbour()[f];
        const scalar d[3] = {C[N].x - C[P].x, C[N].y - C[P].y, C[N].z - C[P].z};
        scalar gv[3];
        for (int j = 0; j < 3; ++j) gv[j] = U[j][N] - U[j][P];
        const scalar gradf = gv[0]*gv[0] + gv[1]*gv[1] + gv[2]*gv[2];
        const label up = (phi[f] > 0.0) ? P : N;
        scalar gradcf = 0;
        for (int j = 0; j < 3; ++j)
        {
            scalar dg = 0;
            for (int i = 0; i < 3; ++i) dg += d[i]*gradU[(3*i + j)*nC + up];
            gradcf += gv[j]*dg;
        }
        const auto sign = [](scalar s) { return (s >= 0.0) ? scalar(1) : scalar(-1); };
        const scalar r = (std::fabs(gradcf) >= 1000.0*std::fabs(gradf))
                       ? 2.0*1000.0*sign(gradcf)*sign(gradf) - 1.0
                       : 2.0*(gradcf/gradf) - 1.0;
        scalar lim = twoByk*r;
        lim = std::max(scalar(0), std::min(scalar(1), lim));
        const scalar W = lim*w[f] + (1.0 - lim)*((phi[f] >= 0.0) ? 1.0 : 0.0);
        lower[f] = -W*phi[f];
        upper[f] = lower[f] + phi[f];
        diag[P] -= lower[f];       // negSumDiag
        diag[N] -= upper[f];
    }
}

std::string writeSchemes(const std::string& divLine)
{
    const std::string dir = "test_limited_linear_v.case";
    std::system(("mkdir -p " + dir + "/system").c_str());
    std::ofstream f(dir + "/system/fvSchemes");
    f << "ddtSchemes { default Euler; }\n"
         "gradSchemes { default Gauss linear; }\n"
         "divSchemes { default none; " << divLine << " div((nuEff*dev2(T(grad(U))))) Gauss linear; }\n"
         "laplacianSchemes { default Gauss linear corrected; }\n"
         "interpolationSchemes { default linear; }\n"
         "snGradSchemes { default corrected; }\n";
    return dir;
}
} // namespace

int main()
{
    std::printf("== div(phi,U) limitedLinearV (OF NVDVTVDV) ==\n");

    PrimitiveMesh m = boxtest::boxMesh(6, 5, 4);
    FvGeometry g; g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const label nC = m.nCells(), nIf = m.nInternalFaces();

    // A flux with both signs (the upwind side must actually switch across the mesh).
    std::vector<scalar> phi(nIf);
    for (label f = 0; f < nIf; ++f)
    {
        const vector& S = g.Sf()[f];
        phi[f] = S.x*1.0 + S.y*0.35 - S.z*0.6;
    }
    DeviceBuffer<scalar> dPhi; dPhi.copyFrom(phi);

    std::vector<scalar> hW;  dm.w.copyTo(hW);
    const scalar twoByk = 2.0;   // k = 1, what every TJunction case asks for

    // ---- the field used for legs 1 and 4: a sharp shear layer, nothing like linear ------------------
    const std::vector<vector>& C = g.C();
    std::vector<scalar> U[3] = {std::vector<scalar>(nC), std::vector<scalar>(nC), std::vector<scalar>(nC)};
    std::vector<scalar> gradU(9*nC, 0.0);
    for (label c = 0; c < nC; ++c)
    {
        // Every component varies along every axis, so no internal face has a zero velocity JUMP.
        // A face with U[N] == U[P] is a real special case in OF (gradf == 0 takes the 1000-ratio
        // branch and comes out UNLIMITED), and leg 3 wants faces that are limited, not that.
        const scalar x = C[c].x, y = C[c].y, z = C[c].z;
        const scalar t = std::tanh(6.0*(x - 3.0));
        U[0][c] = t + 0.05*z;
        U[1][c] = 0.4*y*y + 0.07*x;
        U[2][c] = 0.1*x*y + 0.3*z;
        const scalar dt = 6.0*(1.0 - t*t);
        gradU[(3*0 + 0)*nC + c] = dt;          // dU0/dx
        gradU[(3*2 + 0)*nC + c] = 0.05;        // dU0/dz
        gradU[(3*1 + 1)*nC + c] = 0.8*y;       // dU1/dy
        gradU[(3*0 + 1)*nC + c] = 0.07;        // dU1/dx
        gradU[(3*0 + 2)*nC + c] = 0.1*y;       // dU2/dx
        gradU[(3*1 + 2)*nC + c] = 0.1*x;       // dU2/dy
        gradU[(3*2 + 2)*nC + c] = 0.3;         // dU2/dz
    }
    DeviceBuffer<scalar> dU[3], dG;
    for (int j = 0; j < 3; ++j) dU[j].copyFrom(U[j]);
    dG.copyFrom(gradU);

    DeviceBuffer<scalar> diag, up, lo;
    deviceDivLimitedVCoeffs(dm, dPhi, dU, dG, twoByk, diag, up, lo);
    std::vector<scalar> hDiag, hUp, hLo;
    diag.copyTo(hDiag); up.copyTo(hUp); lo.copyTo(hLo);

    // ---- Leg 1: against an independent host transcription of the OF formula -------------------------
    std::vector<scalar> rUp, rLo, rDiag;
    refCoeffs(m, g, hW, phi, U, gradU, twoByk, rUp, rLo, rDiag);
    check(maxAbs(hUp, rUp) < 1e-14 && maxAbs(hLo, rLo) < 1e-14, "kernel upper/lower == the host OF formula");
    check(maxAbs(hDiag, rDiag) < 1e-13, "kernel diag == negSumDiag of the host formula");

    // ---- Leg 2: a LINEAR field is unlimited -> plain central differencing ---------------------------
    // U = A.x with the exact (constant) gradient: gradfV = A.d and (d & gradU) = A.d, so gradcf == gradf
    // and r == 1 for EVERY face, whatever the flux sign. limiter == 1 -> the CD weights survive intact.
    {
        const scalar A[3][3] = {{ 0.7, -0.2, 0.9},     // A[j][i] = dU_j/dx_i
                                {-0.4,  1.1, 0.3},
                                { 0.25, 0.5,-0.8}};
        std::vector<scalar> Ul[3] = {std::vector<scalar>(nC), std::vector<scalar>(nC), std::vector<scalar>(nC)};
        std::vector<scalar> gl(9*nC, 0.0);
        for (label c = 0; c < nC; ++c)
        {
            const scalar xc[3] = {C[c].x, C[c].y, C[c].z};
            for (int j = 0; j < 3; ++j)
            {
                Ul[j][c] = A[j][0]*xc[0] + A[j][1]*xc[1] + A[j][2]*xc[2];
                for (int i = 0; i < 3; ++i) gl[(3*i + j)*nC + c] = A[j][i];
            }
        }
        DeviceBuffer<scalar> dUl[3], dGl;
        for (int j = 0; j < 3; ++j) dUl[j].copyFrom(Ul[j]);
        dGl.copyFrom(gl);
        DeviceBuffer<scalar> d2, u2, l2, dc, uc, lc;
        deviceDivLimitedVCoeffs(dm, dPhi, dUl, dGl, twoByk, d2, u2, l2);
        deviceDivCentralCoeffs(dm, dPhi, dc, uc, lc);
        std::vector<scalar> a, b, a2, b2;
        u2.copyTo(a); uc.copyTo(b); l2.copyTo(a2); lc.copyTo(b2);
        check(maxAbs(a, b) < 1e-12 && maxAbs(a2, b2) < 1e-12,
              "limit 1: a linear U with its exact gradient reproduces central differencing exactly");
    }

    // ---- Leg 3: a zero gradient is fully limited -> pure upwind -------------------------------------
    // gradcf == 0 with gradf != 0 gives r == -1, so limiter == 0 on every face.
    {
        DeviceBuffer<scalar> dZero; dZero.copyFrom(std::vector<scalar>(9*nC, 0.0));
        DeviceBuffer<scalar> d3, u3, l3, du, uu, lu;
        deviceDivLimitedVCoeffs(dm, dPhi, dU, dZero, twoByk, d3, u3, l3);
        deviceDivUpwindCoeffs(dm, dPhi, du, uu, lu);
        std::vector<scalar> a, b, a2, b2;
        u3.copyTo(a); uu.copyTo(b); l3.copyTo(a2); lu.copyTo(b2);
        check(maxAbs(a, b) < 1e-15 && maxAbs(a2, b2) < 1e-15,
              "limit 0: a zero gradient reproduces pure upwind exactly");
    }

    // ---- Leg 4: the real field sits strictly between the two limits ---------------------------------
    {
        DeviceBuffer<scalar> dc, uc, lc, du, uu, lu;
        deviceDivCentralCoeffs(dm, dPhi, dc, uc, lc);
        deviceDivUpwindCoeffs(dm, dPhi, du, uu, lu);
        std::vector<scalar> hc, hu;
        uc.copyTo(hc); uu.copyTo(hu);
        const scalar dCen = maxAbs(hUp, hc), dUpw = maxAbs(hUp, hu), spread = maxAbs(hc, hu);
        check(dCen > 0.05*spread && dUpw > 0.05*spread,
              "discrimination: the shear-layer field is neither pure central nor pure upwind");
        std::printf("        (|lim-central| = %.3g, |lim-upwind| = %.3g, |central-upwind| = %.3g)\n",
                    dCen, dUpw, spread);
        // and the limiter genuinely varies: some faces are AT each limit
        std::vector<scalar> rU, rL, rD;
        refCoeffs(m, g, hW, phi, U, gradU, twoByk, rU, rL, rD);
        int atCen = 0, atUpw = 0;
        for (label f = 0; f < nIf; ++f)
        {
            if (std::fabs(rU[f] - hc[f]) < 1e-14) ++atCen;
            if (std::fabs(rU[f] - hu[f]) < 1e-14) ++atUpw;
        }
        check(atCen > 0 && atUpw > 0, "the per-face limiter reaches BOTH 1 and 0 on this mesh");
        std::printf("        (%d/%d faces unlimited, %d/%d fully upwinded)\n", atCen, (int)nIf, atUpw, (int)nIf);
    }

    // ---- Leg 5: the scheme is actually selected, and a bad k is refused -----------------------------
    {
        DeviceSimpleControls ctl;
        parseFvSchemesControls(writeSchemes("div(phi,U) Gauss limitedLinearV 1;"), ctl);
        check(ctl.divULimitedV && std::fabs(ctl.divUTwoBykV - 2.0) < 1e-14,
              "parser: `Gauss limitedLinearV 1` selects the scheme with twoByk = 2");
        check(!ctl.divULinear && !ctl.linearUpwind, "parser: it does not also trip the linear/linearUpwind paths");
    }
    {
        DeviceSimpleControls ctl;
        parseFvSchemesControls(writeSchemes("div(phi,U) Gauss limitedLinearV 0.5;"), ctl);
        check(std::fabs(ctl.divUTwoBykV - 4.0) < 1e-14, "parser: k = 0.5 gives twoByk = 4");
    }
    {
        DeviceSimpleControls ctl;
        bool threw = false;
        try { parseFvSchemesControls(writeSchemes("div(phi,U) Gauss limitedLinearV 2;"), ctl); }
        catch (const std::exception&) { threw = true; }
        check(threw, "refusal: k outside [0,1] is rejected, as OF's limitedLinearLimiter does");
    }
    {
        DeviceSimpleControls ctl;
        parseFvSchemesControls(writeSchemes("div(phi,U) Gauss upwind;"), ctl);
        check(!ctl.divULimitedV, "negative control: `Gauss upwind` leaves the scheme off");
    }
    std::system("rm -rf test_limited_linear_v.case");

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
