// fanPressure, and the parser bug that finding it exposed.
//
// OF's fanPressureFvPatchScalarField::updateCoeffs measures the patch's volumetric flow rate, looks up the
// fan's pressure rise at that rate, and hands totalPressure a shifted p0:
//     dir         = -1 for `direction in`, +1 for `out`   (OF: 2*ffd - 1, with ffdIn = 0)
//     volFlowRate = dir*sum(phi_patch)
//     p0_eff      = p0 - dir*fanCurve(max(volFlowRate, 0))
// The face treatment is then plain totalPressure, which brae already had, so the model is: read the curve,
// shift p0 each step. On pimpleFoam/RAS/TJunctionFan the inlet settles at p = 42.6 against OpenFOAM's 42.2,
// well above the p0 of 30 -- i.e. the ~+13 rise off the curve is being applied by both.
//
// LEG 4 IS THE ONE THAT MATTERS BEYOND THIS BC. The field reader's skip-unknown-entry helper tracked
// parentheses but not braces, so a patch entry carrying a sub-DICTIONARY value --
//     fanCurve { type table; file "<constant>/FluxVsdP.dat"; }
// -- stopped at the ';' INSIDE the block, and the block's '}' was then taken for the patch's. Every later
// patch of that field vanished. It surfaced as "no boundaryField entry for patch outlet1", naming a patch
// that is plainly present and pointing nowhere near the fanCurve two entries above it. The polyMesh
// boundary parser had the identical bug for cyclicACMI's `scaleCoeffs` (test_boundary_subdict_skip); this
// is the same defect in the FIELD reader, and it will bite any future BC with a sub-dictionary option.
#include "fan_pressure.cuh"
#include "foam_field_reader.cuh"
#include "box_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

void near(const char* what, scalar got, scalar want, scalar tol = 1e-12)
{
    if (std::fabs(got - want) <= tol) return;
    std::printf("  FAIL %s: got %.10g, want %.10g\n", what, (double)got, (double)want);
    ++failures;
}
}   // namespace

int main()
{
    const char* tmp = std::getenv("TMPDIR");
    const std::string dir = std::string(tmp ? tmp : "/tmp") + "/brae_fan";
    std::system(("rm -rf " + dir + " && mkdir -p " + dir + "/constant && mkdir -p " + dir + "/0").c_str());

    // ---- 1. the curve: linear between points, CLAMPED outside (OF's `outOfBounds clamp` default) ----
    {
        DeviceSimpleSolver::FanPressure f;
        f.curve = {{0.0, 20.0}, {0.0023, 10.0}, {0.003, 5.0}};
        near("below range clamps to the first value", f.dp(-1.0), 20.0);
        near("at a knot",                             f.dp(0.0023), 10.0);
        near("midway between knots",                  f.dp(0.00115), 15.0);
        near("above range clamps to the last value",  f.dp(99.0), 5.0);
        std::printf("  curve: dp(-1)=%.4g dp(0.00115)=%.4g dp(99)=%.4g\n",
                    (double)f.dp(-1.0), (double)f.dp(0.00115), (double)f.dp(99.0));
    }

    // ---- the fixture: a p field whose inlet is a fanPressure with a FILE curve, and patches after it ----
    { std::ofstream o(dir + "/constant/FluxVsdP.dat");
      o << "(\n    (0      20)\n    (0.0023 10)\n    (0.003  5)\n);\n"; }
    PrimitiveMesh m = boxtest::boxMesh(2, 2, 1);
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);
    {
        std::ofstream o(dir + "/0/p");
        o << "FoamFile { version 2.0; format ascii; class volScalarField; object p; }\n"
          << "dimensions      [0 2 -2 0 0 0 0];\ninternalField   uniform 0;\nboundaryField\n{\n";
        bool first = true;
        for (const FvPatch& q : fvp)
        {
            o << "    " << q.name << "\n    {\n";
            if (first)
                o << "        type            fanPressure;\n"
                  << "        direction       in;\n"
                  << "        fanCurve\n        {\n            type table;\n"
                  << "            file \"<constant>/FluxVsdP.dat\";\n        }\n"
                  << "        p0              uniform 30;\n";
            else
                o << "        type            zeroGradient;\n";
            o << "    }\n";
            first = false;
        }
        o << "}\n";
    }

    // ---- 4. the field reader must survive the fanCurve sub-dictionary ----
    {
        const FieldData<scalar> fd = readField<scalar>(dir + "/0/p");
        std::printf("  field reader: %zu boundary entries parsed (mesh has %zu patches)\n",
                    fd.boundary.size(), fvp.size());
        if (fd.boundary.size() != fvp.size())
        {
            std::printf("  FAIL a patch entry with a sub-dictionary value swallowed the patches after it.\n"
                        "       The skip-unknown-entry helper must track braces, not just parentheses.\n");
            ++failures;
        }
        else
        {
            bool sawLast = false;
            for (const auto& b : fd.boundary) if (b.name == fvp.back().name) sawLast = true;
            if (!sawLast)
            { std::printf("  FAIL the LAST patch is missing, so the parse desynchronised\n"); ++failures; }
        }
    }

    // ---- 2/3. collectFanPressure: the sign, the file curve, and <constant> expansion ----
    {
        const std::vector<DeviceSimpleSolver::FanPressure> fans = collectFanPressure(dir, dir + "/0", fvp);
        std::printf("  collectFanPressure: %zu patch(es)", fans.size());
        if (!fans.empty())
            std::printf(", dir=%+.0f p0=%.4g, %zu curve points", (double)fans[0].dir, (double)fans[0].p0,
                        fans[0].curve.size());
        std::printf("\n");
        if (fans.size() != 1)
        { std::printf("  FAIL expected exactly one fanPressure patch\n"); ++failures; }
        else
        {
            near("direction `in` -> dir = -1", fans[0].dir, -1.0);
            near("p0", fans[0].p0, 30.0);
            if (fans[0].curve.size() != 3)
            {
                std::printf("  FAIL the file-based fanCurve was not read (<constant> expansion, or the\n"
                            "       table parse). brae's Function1 builds inline tables only, which is why\n"
                            "       this path exists at all.\n");
                ++failures;
            }
            else near("curve read from the file", fans[0].curve[1].second, 10.0);
            // ...and the shift OF applies: p0_eff = p0 - dir*dp, so `in` RAISES p0 by the fan rise.
            const scalar p0eff = fans[0].p0 - fans[0].dir * fans[0].dp(0.0);
            near("p0_eff at zero flow (30 + 20)", p0eff, 50.0);
        }
    }

    // ---- 5. `out` flips the sign; a bad direction is refused ----
    {
        std::string s;
        { std::ifstream in(dir + "/0/p"); s.assign((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()); }
        std::string sOut = s;
        sOut.replace(sOut.find("direction       in;"), std::string("direction       in;").size(), "direction       out;");
        { std::ofstream o(dir + "/0/p"); o << sOut; }
        const std::vector<DeviceSimpleSolver::FanPressure> f2 = collectFanPressure(dir, dir + "/0", fvp);
        if (f2.size() == 1) near("direction `out` -> dir = +1", f2[0].dir, 1.0);
        else { std::printf("  FAIL the `out` variant did not parse\n"); ++failures; }

        std::string sBad = s;
        sBad.replace(sBad.find("direction       in;"), std::string("direction       in;").size(), "direction       sideways;");
        { std::ofstream o(dir + "/0/p"); o << sBad; }
        bool threw = false;
        try { collectFanPressure(dir, dir + "/0", fvp); } catch (const std::exception&) { threw = true; }
        std::printf("  bad direction refused: %d\n", (int)threw);
        if (!threw)
        { std::printf("  FAIL an unrecognised direction must be refused, not defaulted -- the sign of the\n"
                      "       fan's contribution depends on it\n"); ++failures; }
    }

    std::printf("fan_pressure: %d failures\n", failures);
    return failures ? 1 : 0;
}
