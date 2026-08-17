// CUDA vs the _cpp REFERENCE, stage by stage.
//
// Every existing GPU test compares the device against CPU code written inline in that same test. This one
// compares it against the _cpp reference components, which are themselves validated against OpenFOAM's own
// dumps (test_peqn_cpp, test_ueqn_cpp, test_simple_step_cpp). That closes the chain the rebuild is built
// around:
//
//     OpenFOAM  ->  _cpp reference  ->  CUDA
//
// and it is compared at STAGE granularity, so a disagreement names the stage rather than the iteration.
// Debugging from a final residual is what this replaces.
//
// Run: test_gpu_vs_cpp <caseDir> <timeDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "fv_patch_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "fv_matrix_ops.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_simple.cuh"
#include "UEqn_cpp.cuh"
#include "pEqn_cpp.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

static int g_fails = 0;

static void cmp(const std::vector<scalar>& gpu, const std::vector<scalar>& ref,
                const char* nm, scalar tol)
{
    scalar mx = 0, mg = 0;
    for (std::size_t i = 0; i < ref.size(); ++i)
    {
        mx = std::fmax(mx, std::fabs(gpu[i] - ref[i]));
        mg = std::fmax(mg, std::fabs(ref[i]));
    }
    const scalar rel = mg > 0 ? mx / mg : mx;
    const bool ok = rel <= tol;
    if (!ok) ++g_fails;
    std::printf("  %-30s n=%6zu rel=%.3e  %s\n", nm, ref.size(), rel, ok ? "OK" : "FAIL");
}

static void check(bool ok, const char* what)
{
    std::printf("  %-52s %s\n", what, ok ? "OK" : "FAIL");
    if (!ok) ++g_fails;
}

int main(int argc, char** argv)
{
    if (argc < 3) { std::printf("usage: %s <caseDir> <timeDir>\n", argv[0]); return 2; }
    const std::string caseDir = argv[1], t = argv[2];
    const scalar nu = 1e-5;

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    const label nC = m.nCells();

    GeometricField<vector> U =
        buildField<vector>(readField<vector>(caseDir + "/" + t + "/U"), fvp, nC);
    U.evaluateBoundary();
    GeometricField<scalar> p =
        buildField<scalar>(readField<scalar>(caseDir + "/" + t + "/p"), fvp, nC);
    p.evaluateBoundary();

    const FieldData<scalar> phiF = readField<scalar>(caseDir + "/" + t + "/phi");
    std::vector<std::vector<scalar>> phiBnd(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        phiBnd[pi].assign(fvp[pi].size, 0.0);
        for (const auto& b : phiF.boundary)
            if (b.name == fvp[pi].name && b.hasValue && (label)b.values.size() == fvp[pi].size)
                phiBnd[pi] = b.values;
    }

    const DeviceMesh dm = buildDeviceMesh(m, g, fvp);

    std::printf("test_gpu_vs_cpp:\n");

    // ---- the _cpp reference UEqn and its pressure stages ------------------------------------
    std::vector<scalar> nuEffC(nC, nu);
    std::vector<std::vector<scalar>> nuEffB(fvp.size());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi) nuEffB[pi].assign(fvp[pi].size, nu);

    cpu::MomentumInput mi;
    mi.phi = &phiF.internalField; mi.phiBnd = &phiBnd;
    mi.nuEff = &nuEffC;           mi.nuEffBnd = &nuEffB;
    mi.relaxU = 1.0;
    const FvVectorMatrix UEqn = cpu::assembleUEqn(U, mi, m, g, fvp);

    cpu::PressureInput pin;
    pin.pRefCell = -1;
    const cpu::PressureStages st = cpu::pressurePredictor(UEqn, U, p, pin, m, g, fvp);
    const FvScalarMatrix pEqn = cpu::assemblePEqn(st, p, pin, m, g, fvp);

    // ---- stage: rAU = 1/A(), via deviceReciprocalV ------------------------------------------
    // The device takes the already-folded diagonal (D = diag + cmptAv(internalCoeffs)) and divides by V.
    // Build that diagonal from the reference matrix so the comparison isolates the KERNEL, not the
    // assembly that feeds it -- otherwise a disagreement could come from either and name neither.
    {
        std::vector<scalar> D = UEqn.diag;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            for (label i = 0; i < fvp[pi].size; ++i)
                D[fvp[pi].faceCells[i]] += cmptAv(UEqn.internalCoeffs[pi][i]);

        DeviceBuffer<scalar> dD, drAU;
        dD.copyFrom(D);
        deviceReciprocalV(dm, dD, drAU);
        cmp(drAU.host(), st.rAU, "rAU  (deviceReciprocalV)", 1e-14);
    }

    // ---- stage: the pressure Laplacian coefficients -----------------------------------------
    {
        const SurfaceScalarField rAUf = fvc::interpolate(st.rAU, m, g, fvp);
        DeviceBuffer<scalar> dgamma;
        dgamma.copyFrom(rAUf.internal);
        DeviceBuffer<scalar> dDiag, dUp, dLo;
        deviceLaplacianCoeffs(dm, dgamma, dDiag, dUp, dLo);
        cmp(dUp.host(),   pEqn.upper, "pEqn upper (deviceLaplacian)", 1e-13);
        cmp(dLo.host(),   pEqn.lower, "pEqn lower (deviceLaplacian)", 1e-13);
        cmp(dDiag.host(), pEqn.diag,  "pEqn diag  (deviceLaplacian)", 1e-13);
    }

    // ---- stage: pEqn.flux() internal --------------------------------------------------------
    // faceH(p) = upper*p[nei] - lower*p[own]; the same expression the reference matrixFlux uses.
    {
        DeviceBuffer<scalar> dUp, dLo, dDiag, dp, dflux;
        dUp.copyFrom(pEqn.upper); dLo.copyFrom(pEqn.lower);
        dDiag.copyFrom(pEqn.diag); dp.copyFrom(p.internal);
        DeviceLduView A{};
        A.nCells = nC; A.nInternalFaces = m.nInternalFaces();
        A.diag = dDiag.data(); A.upper = dUp.data(); A.lower = dLo.data();
        A.owner = dm.owner.data(); A.nei = dm.nei.data();
        A.ownerStart = dm.ownerStart.data();
        A.losort = dm.losort.data(); A.losortStart = dm.losortStart.data();
        deviceMatrixFluxInternal(A, dp, dflux);

        const SurfaceScalarField ref = matrixFlux(pEqn, p.internal, m, fvp);
        cmp(dflux.host(), ref.internal, "pEqn.flux() (deviceMatrixFlux)", 1e-13);
    }

    // ---- stage: setReference ----------------------------------------------------------------
    // fvMatrix.C:1011-1023 DOUBLES the diagonal. Assert the kernel does the same thing the reference
    // does, on the same cell, and touches nothing else.
    {
        const label refCell = 7;
        const scalar refValue = 2.25;
        DeviceBuffer<scalar> dDiag, dSrc;
        dDiag.copyFrom(pEqn.diag); dSrc.copyFrom(pEqn.source);
        deviceSetReference(dDiag, dSrc, refCell, refValue);

        std::vector<scalar> rDiag = pEqn.diag, rSrc = pEqn.source;
        rSrc[refCell] += rDiag[refCell] * refValue;
        rDiag[refCell] += rDiag[refCell];

        cmp(dDiag.host(), rDiag, "setReference diag", 1e-15);
        cmp(dSrc.host(),  rSrc,  "setReference source", 1e-15);
        // Control: the reference cell must actually have changed, or both sides agree trivially.
        check(rDiag[refCell] != pEqn.diag[refCell], "the reference cell really changed (control)");
    }

    // ---- control: the comparison can detect a difference ------------------------------------
    // Perturb one reference value and require cmp to report a failure. Without this every OK above is
    // only evidence that the comparator ran.
    {
        std::vector<scalar> a = pEqn.diag, b = pEqn.diag;
        b[0] *= 1.001;
        scalar mx = 0, mg = 0;
        for (std::size_t i = 0; i < a.size(); ++i)
        {
            mx = std::fmax(mx, std::fabs(a[i] - b[i]));
            mg = std::fmax(mg, std::fabs(b[i]));
        }
        check((mg > 0 ? mx / mg : mx) > 1e-13, "a 0.1% perturbation is detected (control)");
    }

    std::printf("%s\n", g_fails == 0 ? "PASS" : "FAIL");
    return g_fails == 0 ? 0 : 1;
}
