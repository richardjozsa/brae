// E6: the universal gas constant must be resolved the way OpenFOAM resolves it, not compiled in.
//
// brae's RR was a literal, and it was wrong once already -- the CODATA-2018 8314.46261815324 against OF
// v2412's 8314.47006650545, an 8.958e-07 gap that reached rho in every compressible case. Matching the
// NUMBER fixed the default. It did not match the MECHANISM: OF builds RR = NA*k from `DimensionedConstants`
// in its etc/controlDict, and every entry there is overridable (defineDimensionedConstantWithDefault), so a
// site or user file changes the gas constant for every OF run on that machine while brae would carry on
// with its literal.
//
// The test that matters is therefore NOT "the default is right" -- a literal passes that. It is "an
// OVERRIDE moves brae's answer", which only a real resolver can pass. Each case below is run in a
// synthetic OF installation (WM_PROJECT_DIR pointed at a temp tree), never against the real one.
//
// foamRR() caches in a function-local static, so each expectation has to run in its own process. The test
// re-executes itself with a mode argument rather than pretending one process can observe several answers.

#include "foam_constants.cuh"
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

using namespace brae;

namespace {

// A minimal OF etc/controlDict carrying only what foamRR() reads. Deliberately includes a #codeStream
// directive: the REAL etc/controlDict has them, brae's dict reader refuses them on purpose, and the whole
// reason foamRR slices the DimensionedConstants block out textually is to survive that. If the slicing
// regressed to "parse the whole file", this entry is what would catch it.
void writeInstall(const std::string& root, const std::string& unitSet, const std::string& pcBody)
{
    std::filesystem::create_directories(root + "/etc");
    std::ofstream(root + "/etc/controlDict")
        << "FoamFile { version 2.0; format ascii; class dictionary; object controlDict; }\n"
        << "\n"
        << "// A directive brae's reader refuses -- present because OF's real file has these.\n"
        << "InfoSwitches { writePrecision 6; }\n"
        << "someEntry #codeStream { code #{ return 1; #}; };\n"
        << "\n"
        << "DimensionedConstants\n"
        << "{\n"
        << "    unitSet " << unitSet << ";\n"
        << "    " << unitSet << "Coeffs\n"
        << "    {\n"
        << "        physicoChemical\n"
        << "        {\n" << pcBody << "        }\n"
        << "    }\n"
        << "}\n";
}

const char* OF_DEFAULT_PC =
    "            mu mu [1 0 0 0 0 0 0] 1.66054e-27;\n"
    "            k  k  [1 2 -2 -1 0 0 0] 1.38065e-23;\n";

int fail(const char* what, double got, double want)
{
    std::printf("  FAIL %s: got %.15g, want %.15g\n", what, got, want);
    return 1;
}

}   // namespace

int main(int argc, char** argv)
{
    const std::string tmp = "/tmp/brae_foam_constants";
    const std::string mode = (argc > 1) ? argv[1] : "";

    // ---- child modes: one expectation per process, because foamRR() caches ----
    if (mode == "default")
    {
        // OF's own shipped values -> exactly the number OF prints for thermodynamic::RR.
        const double rr = static_cast<double>(foamRR());
        if (std::fabs(rr - 8314.47006650545) > 1e-9) return fail("shipped constants", rr, 8314.47006650545);
        return 0;
    }
    if (mode == "override")
    {
        // k doubled -> RR must double. A literal cannot pass this.
        const double rr = static_cast<double>(foamRR());
        const double want = 1e3 * 6.022141793e23 * 2.7613e-23;
        if (std::fabs(rr - want) > 1e-6 * want) return fail("overridden k", rr, want);
        return 0;
    }
    if (mode == "na")
    {
        // NA is NOT in OF's shipped file (it is a compiled default), but the same mechanism overrides it.
        const double rr = static_cast<double>(foamRR());
        const double want = 1e3 * 6.0e23 * 1.38065e-23;
        if (std::fabs(rr - want) > 1e-6 * want) return fail("overridden NA", rr, want);
        return 0;
    }
    if (mode == "uscs")
    {
        // USCS is a different constant set entirely; brae is SI throughout, so it must REFUSE, not
        // approximate -- every density would be wrong by a units factor.
        try
        {
            (void)foamRR();
        }
        catch (const std::exception&)
        {
            return 0;
        }
        std::printf("  FAIL unitSet USCS was accepted; brae is SI-only and must refuse\n");
        return 1;
    }
    if (mode == "unreachable")
    {
        // No OF install in sight -> fall back to OF v2412's value rather than failing.
        const double rr = static_cast<double>(foamRR());
        if (std::fabs(rr - 8314.47006650545) > 1e-9) return fail("fallback", rr, 8314.47006650545);
        return 0;
    }

    // ---- parent: build each synthetic install and run the child ----
    std::filesystem::remove_all(tmp);
    writeInstall(tmp + "/of_default", "SI", OF_DEFAULT_PC);
    writeInstall(tmp + "/of_override", "SI",
                 "            k  k  [1 2 -2 -1 0 0 0] 2.7613e-23;\n");
    writeInstall(tmp + "/of_na", "SI",
                 "            NA NA [0 0 0 0 -1 0 0] 6.0e23;\n"
                 "            k  k  [1 2 -2 -1 0 0 0] 1.38065e-23;\n");
    writeInstall(tmp + "/of_uscs", "USCS", OF_DEFAULT_PC);

    struct Case { const char* mode; const char* root; };
    const Case cases[] = {
        {"default",     "/of_default"},
        {"override",    "/of_override"},
        {"na",          "/of_na"},
        {"uscs",        "/of_uscs"},
        {"unreachable", "/does_not_exist"},
    };

    int failures = 0;
    for (const Case& c : cases)
    {
        // WM_PROJECT_VERSION is deliberately unset so the ~/.OpenFOAM/<ver> path is not consulted and the
        // test cannot be perturbed by whatever the developer has in their home directory.
        const std::string cmd =
            "WM_PROJECT_DIR=" + tmp + c.root + " WM_PROJECT_VERSION= " +
            std::string(argv[0]) + " " + c.mode;
        const int rc = std::system(cmd.c_str());
        const bool ok = (rc == 0);
        std::printf("  %-12s %s\n", c.mode, ok ? "OK" : "FAIL");
        if (!ok) failures++;
    }

    std::printf("foam_constants: %d failures over %zu cases\n", failures, sizeof(cases)/sizeof(cases[0]));
    return failures ? 1 : 0;
}
