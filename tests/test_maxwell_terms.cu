// The two momentum terms only the Maxwell model has, and the stress production that drives it.
//
// OF laminarModels::Maxwell::divDevRhoReff:
//     div(nuM*grad(U)) + div(sigma) - div(nu*dev2(T(grad(U)))) - laplacian(nu + nuM, U)
// The first two are Gauss divergences of a TENSOR field, and both are new here. The third and fourth
// brae already had.
//
// WHY A DIVERGENCE OF A GRADIENT IS NOT A LAPLACIAN. div(nuM*grad(U)) and laplacian(nuM,U) are the same
// operator on paper and different matrices in a finite-volume code: the first interpolates a cell
// gradient to the face (a WIDE stencil reaching i+/-2), the second uses the compact face snGrad. OF
// leans on exactly that difference -- it puts nuM in the implicit laplacian for stability and subtracts
// it back explicitly -- so the two do not cancel to zero, and a test that only checked "they cancel"
// would pass on an implementation that had both wrong.
//
// Leg 1 therefore pins the term against its ANALYTIC value on a field whose gradient is exact, where
// the wide stencil is exact too: a quadratic profile. Leg 2 does the same for div(sigma), whose answer
// is a pure geometric sum. Leg 3 pins the production tensor P against a hand-multiplied 3x3.
#include "device_maxwell.cuh"
#include "device_divdevreff.cuh"
#include "device_kepsilon.cuh"   // deviceGradU
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "box_mesh.cuh"
#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       std::printf("  ok:   %s\n", what);
}
} // namespace

int main()
{
    std::printf("== Maxwell momentum terms ==\n");

    // A channel-like box: 1 cell in x and z, many in y, so d/dy is the only direction that matters and
    // the answer is one-dimensional and hand-checkable.
    const label NY = 20;
    PrimitiveMesh m = boxtest::boxMesh(1, NY, 1);
    FvGeometry g; g.build(m);
    std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();
    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    const std::vector<vector>& C = g.C();

    // U = (a*y^2 + b*y + c, 0, 0). Its exact laplacian is (2a, 0, 0) -- constant, so a WIDE stencil
    // reproduces it exactly in the interior, which is what makes leg 1 an equality rather than an
    // order-of-accuracy statement.
    const scalar a = 0.7, b = -0.3, c0 = 0.15;
    GeometricField<vector> U;
    U.internal.resize(nC);
    for (label i = 0; i < nC; ++i)
    {
        const scalar y = C[i].y;
        U.internal[i] = vector{a*y*y + b*y + c0, 0, 0};
    }
    // fixedValue on every patch, set to the ANALYTIC value at the face centre: with the exact boundary
    // data the discrete answer is the analytic one, so any deviation is the operator, not the data.
    for (const FvPatch& p : fvp)
    {
        std::vector<vector> vals(p.size);
        for (label i = 0; i < p.size; ++i)
        {
            const scalar y = g.Cf()[p.start + i].y;
            vals[i] = vector{a*y*y + b*y + c0, 0, 0};
        }
        U.boundary.push_back(std::make_unique<FixedValuePatchField<vector>>(p, false, vector{}, vals));
    }
    U.evaluateBoundary();
    const DeviceVectorBoundary dbU = buildDeviceVectorBoundary(U, fvp, g);

    DeviceBuffer<scalar> Uk[3];
    for (int k = 0; k < 3; ++k)
    {
        std::vector<scalar> comp(nC);
        for (label i = 0; i < nC; ++i) comp[i] = component(U.internal[i], k);
        Uk[k].copyFrom(comp);
    }
    DeviceBuffer<scalar> gradU;
    deviceGradU(dm, dbU, Uk[0], Uk[1], Uk[2], gradU);

    // ---- Leg 1: div(nuM*grad(U)) == nuM * laplacian(U) = (2*a*nuM, 0, 0), times V ------------------
    {
        const scalar nuM = 1.3;
        DeviceBuffer<scalar> mX, mY, mZ;
        deviceDivNuMGradU(dm, dbU, Uk[0], Uk[1], Uk[2], gradU, nuM, mX, mY, mZ);
        std::vector<scalar> hx, hy, hz, V;
        mX.copyTo(hx); mY.copyTo(hy); mZ.copyTo(hz);
        dm.V.copyTo(V);
        // The kernels return V*fvc::div (brae's source convention), so divide it back out.
        scalar worst = 0, worstOff = 0, worstBnd = 0;
        for (label i = 0; i < nC; ++i)
        {
            const scalar e = std::fabs(hx[i]/V[i] - 2*a*nuM);
            // The two wall-adjacent cells are NOT expected to be exact: their wall face takes the
            // gaussGrad-corrected boundary gradient, whose normal part is the one-sided snGrad over half
            // a cell -- second-order data, first-order derivative. OpenFOAM's fvc::div(nuM*grad(U)) has
            // exactly the same error there, by the same formula, so the test pins the INTERIOR (where
            // the wide stencil is exact) and separately bounds the boundary cells.
            // TWO cells deep, not one: the wide stencil reaches i+/-2, so the wall cell's one-sided
            // gradient contaminates its neighbour as well. Measured 1.479 and 1.706 against 1.820.
            const bool nearBnd = (i < 2) || (i >= nC - 2);
            if (nearBnd) worstBnd = std::max(worstBnd, e);
            else         worst    = std::max(worst, e);
            worstOff = std::max(worstOff, std::fabs(hy[i]) + std::fabs(hz[i]));
        }
        check(worst < 1e-10, "div(nuM*grad(U)) == nuM*laplacian(U) exactly in the interior");
        check(worstBnd < 0.5*std::fabs(2*a*nuM), "...and the wall cells carry only the snGrad's one-sided error");
        std::printf("        (interior %.3g, wall cells %.3g, expected %.4f)\n",
                    (double)worst, (double)worstBnd, (double)(2*a*nuM));

        check(worstOff < 1e-12, "...and the other two components are identically zero");
        std::printf("        (worst |div - 2a*nuM| = %.3g, expected %.4f)\n", (double)worst, (double)(2*a*nuM));
    }

    // ---- Leg 2: div(sigma) for a LINEAR stress field -----------------------------------------------
    // sigma_xy = s*y (everything else zero) has div(sigma) = (d(sigma_yx)/dy, 0, 0) = (s, 0, 0), again
    // exactly representable. Note which index moves: OF's div(T)_j = d(T_ij)/dx_i, so it is the ROW
    // index that is differentiated -- transposing the tensor here would give zero and look plausible.
    {
        const scalar sc = 0.45;
        std::vector<GeometricField<scalar>> sig(6);
        for (int k = 0; k < 6; ++k)
        {
            sig[k].internal.assign(nC, 0.0);
            if (k == SXY) for (label i = 0; i < nC; ++i) sig[k].internal[i] = sc*C[i].y;
            for (const FvPatch& p : fvp)
            {
                std::vector<scalar> vals(p.size, 0.0);
                if (k == SXY) for (label i = 0; i < p.size; ++i) vals[i] = sc*g.Cf()[p.start + i].y;
                sig[k].boundary.push_back(std::make_unique<FixedValuePatchField<scalar>>(p, false, scalar(0), vals));
            }
            sig[k].evaluateBoundary();
        }
        DeviceBuffer<scalar> sCell[6], sBnd[6];
        for (int k = 0; k < 6; ++k)
        {
            sCell[k].copyFrom(sig[k].internal);
            const DeviceBoundary db = buildDeviceBoundary(sig[k], fvp, g);
            deviceBCValue(db, sCell[k], sBnd[k]);
        }
        DeviceBuffer<scalar> sX, sY, sZ;
        deviceDivSymmTensor(dm, sCell, sBnd, sX, sY, sZ);
        std::vector<scalar> hx, hy, hz, V;
        sX.copyTo(hx); sY.copyTo(hy); sZ.copyTo(hz);
        dm.V.copyTo(V);
        scalar worst = 0, worstOff = 0;
        for (label i = 0; i < nC; ++i)
        {
            worst    = std::max(worst,    std::fabs(hx[i]/V[i] - sc));
            worstOff = std::max(worstOff, std::fabs(hz[i]));
        }
        check(worst < 1e-10, "div(sigma) picks up d(sigma_xy)/dy in the x component");
        check(worstOff < 1e-12, "...and nothing appears in z");
        std::printf("        (worst |div - s| = %.3g, expected %.4f)\n", (double)worst, (double)sc);
    }

    // ---- Leg 3: the production tensor P against a hand-built 3x3 -----------------------------------
    {
        const scalar nuM = 2.0, rLambda = 0.25;
        const scalar S[6] = {1.0, -0.5, 0.25, 2.0, 0.75, -1.5};   // xx xy xz yy yz zz
        const scalar G[9] = {0.1, 0.2, 0.3,   // dU/dx : d(U_x)/dx d(U_y)/dx d(U_z)/dx
                             0.4, 0.5, 0.6,   // dU/dy
                             0.7, 0.8, 0.9};  // dU/dz
        DeviceBuffer<scalar> sig[6], gU, P[6];
        for (int k = 0; k < 6; ++k) sig[k].copyFrom(std::vector<scalar>{S[k]});
        {
            std::vector<scalar> gp(9);
            for (int q = 0; q < 9; ++q) gp[q] = G[q];
            gU.copyFrom(gp);
        }
        deviceMaxwellP(1, sig, gU, nuM, rLambda, P);

        // host reference, written as the matrices OF's expression means
        const scalar Sm[3][3] = {{S[0], S[1], S[2]}, {S[1], S[3], S[4]}, {S[2], S[4], S[5]}};
        scalar Gm[3][3];
        for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) Gm[i][j] = G[3*i + j];
        scalar SG[3][3];
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
            {
                scalar s = 0;
                for (int k = 0; k < 3; ++k) s += Sm[i][k]*Gm[k][j];
                SG[i][j] = s;
            }
        auto ref = [&](int i, int j) { return SG[i][j] + SG[j][i] - nuM*rLambda*(Gm[i][j] + Gm[j][i]); };
        const scalar want[6] = {ref(0,0), ref(0,1), ref(0,2), ref(1,1), ref(1,2), ref(2,2)};
        scalar worst = 0;
        for (int k = 0; k < 6; ++k)
        {
            std::vector<scalar> h; P[k].copyTo(h);
            worst = std::max(worst, std::fabs(h[0] - want[k]));
        }
        check(worst < 1e-13, "P == twoSymm(sigma & gradU) - nuM/lambda * twoSymm(gradU)");

        // Discrimination: with sigma = 0 the first term vanishes and P is pure strain -- so a P that
        // ignored sigma entirely would still pass a zero-sigma test. Check it does NOT here.
        DeviceBuffer<scalar> zero[6], P0[6];
        for (int k = 0; k < 6; ++k) zero[k].copyFrom(std::vector<scalar>{scalar(0)});
        deviceMaxwellP(1, zero, gU, nuM, rLambda, P0);
        std::vector<scalar> h0, h1;
        P0[SXY].copyTo(h0); P[SXY].copyTo(h1);
        check(std::fabs(h0[0] - h1[0]) > 1e-6, "vacuity guard: sigma actually enters P");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
