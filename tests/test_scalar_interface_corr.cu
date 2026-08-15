// The SCALAR forms of the two interface deferred corrections, and why they exist.
//
// k, epsilon, omega, nuTilda and he are transported by deviceSolveScalarTransport, and that path had NONE
// of the coupled-patch terms the momentum predictor has had all along:
//   - deviceGaussGrad sums internal + non-coupled boundary faces, so the gradient stopped at the interface
//   - cellLimitedGrad could not see the coupled neighbour (test_cell_limit_interface)
//   - linearUpwind's deferred correction skipped the coupled faces, which OF reaches through its
//     `if (pSfCorr.coupled())` boundary loop in linearUpwind::correction()
//   - the non-orthogonal laplacian correction skipped them too
//
// The correction kernels already existed for momentum, but their entry points take a component index and
// a three-buffer gradient and rotate the neighbour by forwardT. A scalar has one gradient and is NEVER
// transformed across a rotational interface, so it needs its own entry point. This test pins those new
// overloads to the old ones: on a NON-rotational interface, where rotation is the identity anyway, the
// scalar form must equal the vector form at comp = 0 fed the same gradient in all three slots. That is
// the exact equivalence the overload claims.
//
// WHAT IT DOES NOT CATCH, stated because it was checked: the component index. Feeding the same gradient
// to all three slots makes comp irrelevant by construction, so building the scalar form with comp = 1
// passes this test -- correctly, since for a scalar it IS irrelevant. What the legs do catch is the
// plumbing: swapping dNbr for dOwn in the scalar overload fails leg 1 by 8.3e-01 against a term of
// 5.0e-01.
//
// HONEST SCOPE. On pimpleFoam/RAS/oscillatingInletACMI2D these terms are inert -- the mesh is orthogonal
// (so the non-orth correction is identically zero) and the tutorial convects k/epsilon with `Gauss
// upwind` (so no linearUpwind gradient is built at all). Re-run with `div(phi,k) Gauss linearUpwind
// grad(k)` and a cellLimited grad(k)/grad(epsilon) they do engage, and there the fix moves nut's worst
// relative error from 1.27e-01 to 6.83e-02 -- but k and epsilon barely move, because that case carries a
// separate ~1.8e-02 error which is NOT at the interface (only 6.5% of its squared error is on the
// interface ring). So these terms are correct, not decisive; the k/epsilon gap has another cause.
#include "acmi_mesh.cuh"
#include "acmi_area_scaling.cuh"
#include "device_mesh.cuh"
#include "device_ami.cuh"
#include "device_interface.cuh"
#include "device_boundary.cuh"
#include "device_blas.cuh"
#include "cyclic_field.cuh"
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

}   // namespace

int main()
{
    PrimitiveMesh m = acmitest::twoBlockACMI(acmitest::ACMI_DY, /*withBlockage*/true, "ACMI1_blockage",
                                             /*frontBackEmpty*/true);
    FvGeometry g;
    std::vector<FvPatch> fvp;
    std::vector<AMIInterface> amis;
    buildGeometryPatchesAndAMI(m, g, fvp, amis);
    const label nC = m.nCells();

    DeviceMesh dm = buildDeviceMesh(m, g, fvp);
    DeviceAMI  ami = buildDeviceAMI(amis);
    if (ami.n <= 0)
    { std::printf("  FAIL vacuous: the fixture built no AMI faces\n"); return 1; }
    if (ami.rotational)
    { std::printf("  FAIL this fixture is rotational; the equivalence asserted here needs a plain one\n"); return 1; }

    // A NON-ZERO interface flux, alternating in sign so BOTH branches of the kernel run: phi >= 0 takes
    // the owner's gradient, phi < 0 the AMI-interpolated neighbour's. A freshly built AMI has phi = 0,
    // which would send every face down the owner branch and multiply the whole correction by zero.
    {
        std::vector<scalar> phi(ami.n);
        for (int i = 0; i < ami.n; ++i) phi[i] = (i % 2 ? -1.0 : 1.0) * (0.3 + 0.05*i);
        ami.phi.copyFrom(phi);
    }

    std::vector<scalar> f(nC);
    for (label c = 0; c < nC; ++c) f[c] = 2.0 + 4.0*g.C()[c].x + 1.3*g.C()[c].y*g.C()[c].y;
    GeometricField<scalar> ff = buildCyclicField<scalar>(f, fvp, {});
    ff.evaluateBoundary();
    const DeviceBoundary db = buildDeviceBoundary(ff, fvp, g);

    DeviceBuffer<scalar> fd, bv, gx, gy, gz, D;
    fd.copyFrom(f);
    deviceBCValue(db, fd, bv);
    deviceGaussGrad(dm, fd, bv, gx, gy, gz);
    interfaceAddGrad(ami, fd, dm.V, gx, gy, gz);
    D.copyFrom(std::vector<scalar>(nC, 0.7));

    // The same gradient in all three component slots, which is what the scalar overload does internally.
    DeviceBuffer<scalar> g3x[3], g3y[3], g3z[3];
    for (int k = 0; k < 3; ++k) { deviceCopy(g3x[k], gx); deviceCopy(g3y[k], gy); deviceCopy(g3z[k], gz); }

    // ---- 1. linearUpwind: scalar overload == vector overload at comp 0 ----
    {
        DeviceBuffer<scalar> a(nC), b(nC);
        cudaMemset(a.data(), 0, nC*sizeof(scalar));
        cudaMemset(b.data(), 0, nC*sizeof(scalar));
        interfaceAddLinUpwindCorr(ami, gx, gy, gz, a);        // the new scalar form
        interfaceAddLinUpwindCorr(ami, 0, g3x, g3y, g3z, b);  // the existing vector form
        const std::vector<scalar> ha = a.host(), hb = b.host();
        std::printf("  linearUpwind: |scalar - vector| = %.3e   (term peak %.3e)\n",
                    (double)worst(ha, hb), (double)peak(ha));
        if (worst(ha, hb) > 1e-14)
        { std::printf("  FAIL the scalar overload does not reproduce the vector one at comp 0\n"); ++failures; }
        if (peak(ha) < 1e-6)
        {
            std::printf("  FAIL vacuous: the correction is %.3e, indistinguishable from round-off, so the\n"
                        "       comparison above is 0 == 0 -- check that ami.phi is non-zero\n", (double)peak(ha));
            ++failures;
        }
    }

    // ---- 2. the non-orthogonal laplacian correction, same equivalence ----
    // THIS FIXTURE'S OWN corrVec IS ZERO. The two blocks meet on a plane with parallel faces, so the
    // geometric correction vector comes out at 7.9e-18 -- round-off. The first draft of this leg asserted
    // `peak > 0` and passed on exactly that, which is the vacuous pass its own guard was meant to catch.
    // Since what is under test is the KERNEL's arithmetic, not this mesh, inject a corrVec that exercises
    // it and require a real magnitude below.
    {
        std::vector<scalar> cvx(ami.n), cvy(ami.n), cvz(ami.n);
        for (int i = 0; i < ami.n; ++i)
        { cvx[i] = 0.11 + 0.03*i; cvy[i] = -0.07 - 0.02*i; cvz[i] = 0.05*(i % 3); }
        ami.corrVecX.copyFrom(cvx); ami.corrVecY.copyFrom(cvy); ami.corrVecZ.copyFrom(cvz);

        DeviceBuffer<scalar> a(nC), b(nC);
        cudaMemset(a.data(), 0, nC*sizeof(scalar));
        cudaMemset(b.data(), 0, nC*sizeof(scalar));
        interfaceAddLapCorr(ami, D, gx, gy, gz, a);
        interfaceAddLapCorr(ami, 0, D, g3x, g3y, g3z, b);
        const std::vector<scalar> ha = a.host(), hb = b.host();
        std::printf("  lapCorr     : |scalar - vector| = %.3e   (term peak %.3e)\n",
                    (double)worst(ha, hb), (double)peak(ha));
        if (worst(ha, hb) > 1e-14)
        { std::printf("  FAIL the scalar overload does not reproduce the vector one at comp 0\n"); ++failures; }
        if (peak(ha) < 1e-6)
        {
            std::printf("  FAIL vacuous: the term is %.3e, indistinguishable from round-off, so the\n"
                        "       equivalence above is 0 == 0 and proves nothing about the overload\n",
                        (double)peak(ha));
            ++failures;
        }
    }

    // ---- 3. the gradient itself must reach the interface ----
    {
        DeviceBuffer<scalar> px, py, pz;
        deviceGaussGrad(dm, fd, bv, px, py, pz);            // internal + non-coupled boundary only
        const std::vector<scalar> plain = px.host(), withIf = gx.host();
        const scalar w = worst(plain, withIf);
        std::printf("  gradient    : interface contribution moves grad_x by up to %.3e\n", (double)w);
        if (w < 1e-6)
        {
            std::printf("  FAIL interfaceAddGrad contributed nothing, so the scalar transport's gradient\n"
                        "       still stops at the interface\n");
            ++failures;
        }
    }

    std::printf("scalar_interface_corr: %d failures\n", failures);
    return failures ? 1 : 0;
}
