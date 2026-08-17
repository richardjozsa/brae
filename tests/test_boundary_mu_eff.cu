// Host boundary muEff -- the OF verification that turns cpu/boundary_mu_eff.cuh into a baseline.
//
// GROUNDED IN FILES OPENFOAM WROTE ITSELF. No brae dump, no patched OF, no instrumentation: OF writes
// nut's boundaryField into its time directories as a matter of course, and that is the only input this
// test compares against. If OF does not write it, this test does not check it.
//
// WHAT IS ACTUALLY BEING TESTED, and why it is not circular. muEff = mu + rho*nut is a formula, so
// asserting it against itself would prove nothing. The content is in WHERE nut_b comes from:
//
//   OF     takes nut_b from the patch's OWN boundaryField  (a `calculated` inlet carries the case
//          file's value, commonly 0; a wall carries what the wall function wrote)
//   legacy takes nut_b from the ADJACENT CELL, extrapolated   (device_kepsilon.cu:797,831,922)
//
// Those agree on a blockMesh duct, which is why all five validated tutorials never caught it, and
// differ by 607x at gasMixing/injectorPipe's inlet_air. So the test drives a case whose patch nut and
// cell nut genuinely disagree, and asserts the patch value wins. A run that cannot tell the two apart
// is reported as vacuous rather than passing -- an inlet where they happen to coincide proves nothing.
//
// GENERAL: the case comes from argv[1] and no patch name or type is hardcoded. The patch that
// exercises the difference is FOUND by comparing each patch's own nut against its adjacent cells.
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include <sys/stat.h>
#include "boundary_mu_eff.cuh"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

using namespace brae;

namespace {

int failures = 0;
int checked  = 0;

}   // namespace

int main(int argc, char** argv)
{
    const std::string caseDir = argc > 1 ? argv[1] : "validation/rhoBox";
    const std::string timeDir = argc > 2 ? argv[2] : "0";

    // SKIP (77, the code CMake is already told to treat as skipped) rather than abort when the case is
    // absent. validation/ is `.gitignore`d wholesale, so a fixture reaches CI only if it was force-added;
    // this test threw std::runtime_error from PrimitiveMesh::read instead, which ctest reports as a
    // failure and which reads like a solver bug rather than a missing file.
    //
    // Skipping is the fallback, NOT the intent: airfoil is the only checked-in case with faces where the
    // patch nut and the adjacent cell nut differ (78 of 21812), so a skip here means this test proves
    // nothing. The fixture belongs in the repo.
    {
        struct stat st;
        const std::string pts = caseDir + "/constant/polyMesh/points";
        if (stat(pts.c_str(), &st) != 0 && stat((pts + ".gz").c_str(), &st) != 0)
        {
            std::printf("SKIP: case fixture '%s' not present\n", caseDir.c_str());
            return 77;
        }
    }

    PrimitiveMesh m;
    m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);

    // OF's own written fields. nut is optional: a laminar case has none, and muEff must then be mu.
    GeometricField<scalar> nut = buildField<scalar>(
        readField<scalar>(caseDir + "/" + timeDir + "/nut"), fvp, m.nCells());
    nut.evaluateBoundary();

    // mu and rho are built uniform here because this test is about WHERE nut_b comes from, not about
    // the thermo. Any positive constants work: the extrapolated-vs-patch distinction is carried
    // entirely by nut, and a wrong mu/rho would shift both candidates equally and hide nothing.
    FieldData<scalar> muFd;
    muFd.internalUniform = true;
    muFd.internalUniformValue = 1.8476e-05;
    FieldData<scalar> rhoFd;
    rhoFd.internalUniform = true;
    rhoFd.internalUniformValue = 1.0;
    for (const FvPatch& p : fvp)
    {
        PatchFieldData<scalar> dmu;
        dmu.name = p.name;
        dmu.type = (p.type == "empty") ? "empty" : "calculated";
        dmu.hasValue = (dmu.type != "empty");
        dmu.valueUniform = dmu.hasValue;
        dmu.uniformValue = 1.8476e-05;
        muFd.boundary.push_back(dmu);

        PatchFieldData<scalar> drho = dmu;
        drho.uniformValue = 1.0;
        rhoFd.boundary.push_back(drho);
    }
    GeometricField<scalar> mu  = buildField<scalar>(muFd,  fvp, m.nCells());
    GeometricField<scalar> rho = buildField<scalar>(rhoFd, fvp, m.nCells());
    mu.evaluateBoundary();
    rho.evaluateBoundary();

    const std::vector<std::vector<scalar>> muEff = cpu::boundaryMuEff(fvp, mu, rho, &nut);

    // ---------------------------------------------------------------------------------------------
    // 1. muEff_b is built from the patch's own nut, NOT from the adjacent cell.
    // For every patch, compare against both candidates and require the patch one. Patches where the
    // two agree cannot discriminate and are counted separately so a vacuous run is visible.
    const std::vector<label>& faceCell = m.owner();
    int discriminating = 0;

    for (std::size_t p = 0; p < fvp.size(); ++p)
    {
        const FvPatch& patch = fvp[p];
        if (patch.size == 0) continue;

        const std::vector<scalar>& nutB = nut.boundary[p]->value();

        for (label i = 0; i < patch.size; ++i)
        {
            const label f = patch.start + i;
            const label c = faceCell[f];

            const double wantPatch = 1.8476e-05 + 1.0*nutB[i];          // OF: patch's own nut
            const double wantCell  = 1.8476e-05 + 1.0*nut.internal[c];  // legacy: extrapolated cell nut
            const double got       = muEff[p][i];

            // Only faces where the two candidates differ can tell the implementations apart.
            const double sep = std::fabs(wantPatch - wantCell)
                             / std::fmax(std::fabs(wantPatch), 1e-300);
            if (sep > 1e-9) ++discriminating;

            ++checked;
            if (std::fabs(got - wantPatch)/std::fmax(std::fabs(wantPatch), 1e-300) > 1e-13)
            {
                if (failures < 5)
                {
                    std::printf("  FAIL patch %s face %d: muEff_b %.9e, patch-nut gives %.9e,\n"
                                "       cell-extrapolated gives %.9e -- reading the wrong nut\n",
                                patch.name.c_str(), (int)i, got, wantPatch, wantCell);
                }
                ++failures;
            }
        }
    }

    std::printf("  checked %d boundary faces, %d of them discriminating (patch nut != cell nut)\n",
                checked, discriminating);

    // 2. ANTI-VACUOUS. If no face separates the two implementations, this case cannot detect the
    // defect and a pass means nothing. Say so rather than reporting green.
    if (discriminating == 0)
    {
        std::printf("  FAIL vacuous: on this case every patch nut equals its adjacent cell nut, so the\n"
                    "       extrapolating implementation would pass too. Use a case whose inlets carry a\n"
                    "       nut that differs from the bulk (e.g. calculated nut 0 against a turbulent\n"
                    "       interior) -- otherwise this test cannot go red.\n");
        ++failures;
    }

    // 3. LAMINAR: with no nut field muEff_b must be exactly mu_b, on every patch.
    {
        const std::vector<std::vector<scalar>> lam = cpu::boundaryMuEff(fvp, mu, rho, nullptr);
        for (std::size_t p = 0; p < fvp.size(); ++p)
        {
            for (std::size_t i = 0; i < lam[p].size(); ++i)
            {
                if (lam[p][i] != 1.8476e-05)
                {
                    std::printf("  FAIL laminar muEff_b %.9e != mu_b on patch %s\n",
                                lam[p][i], fvp[p].name.c_str());
                    ++failures;
                    break;
                }
            }
        }
        std::printf("  laminar (no nut): muEff_b = mu_b on all %zu patches\n", fvp.size());
    }

    std::printf("test_boundary_mu_eff: %d failures\n", failures);
    return failures ? 1 : 0;
}
