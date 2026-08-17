// fvOptions: the modern `sources` key, and a miss that must be LOUD.
//
// OF's SemiImplicitSource::readCoeffs looks for `injectionRateSuSp` first -- kept for 2112 and earlier --
// and falls back to `sources`, which is what every current case writes. brae read only the legacy name,
// so pimpleFoam/laminar/planarPoiseuille (`sources { U ((5 0 0) 0); }`) parsed to NOTHING and the driver
// printed "No finite volume options present". That case's entire flow IS the source: the fluid starts at
// rest with no inlet, so the run produced a plausible, quiet, completely wrong answer.
//
// With the key read, brae matches OpenFOAM on it to 7.92e-08 in U over 200 steps (|U|max 4.94).
//
// Leg 2 is the more important one. A source whose numbers brae cannot find must be reported as
// unsupported, not skipped: silently dropping a body force is the exact failure the transient driver's
// blanket fvOptions refusal was standing in for, and that refusal has now been lifted.
//
// Leg 3 is the TIME WINDOW, which only a transient driver has to care about. OF's cellSetOption is active
// for timeStart <= t <= timeStart + duration; brae bakes all sources into one set of buffers and cannot
// switch one off mid-run, so the driver refuses a window it cannot honour. This asserts the window is
// carried out of the reader at all -- without it the driver has nothing to refuse on.
#include "fv_options.cuh"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

std::string mkCase(const std::string& base, const char* body)
{
    std::system(("rm -rf " + base + " && mkdir -p " + base + "/constant").c_str());
    std::ofstream o(base + "/constant/fvOptions");
    o << "FoamFile { version 2.0; format ascii; class dictionary; object fvOptions; }\n" << body;
    return base;
}
}   // namespace

int main()
{
    const char* tmp = std::getenv("TMPDIR");
    const std::string root = std::string(tmp ? tmp : "/tmp") + "/brae_fvo_key";
    const label nC = 8;
    const std::vector<scalar> V(nC, 0.5);

    // ---- 1. the modern `sources` spelling is read ----
    {
        const std::string c = mkCase(root + "/modern",
            "momentumSource\n{\n    type            vectorSemiImplicitSource;\n"
            "    selectionMode   all;\n    volumeMode      specific;\n"
            "    sources\n    {\n        U    ((5 0 0) 0);\n    }\n}\n");
        const FvOptionsData fo = readFvOptions(c, {}, V, nC);
        std::printf("  `sources`: count=%d hasMomentum=%d unsupported=%zu",
                    fo.count, (int)fo.hasMomentum, fo.unsupported.size());
        if (fo.hasMomentum && !fo.momSu[0].empty()) std::printf(" momSu[0][0]=%.6g", (double)fo.momSu[0][0]);
        std::printf("\n");
        if (!fo.unsupported.empty())
        { std::printf("  FAIL reported unsupported: %s\n", fo.unsupported[0].c_str()); ++failures; }
        if (fo.count != 1 || !fo.hasMomentum)
        {
            std::printf("  FAIL the `sources` sub-dictionary was not read -- this is the planarPoiseuille\n"
                        "       failure, where the whole body force vanished and the run stayed quiet\n");
            ++failures;
        }
        else if (fo.momSu[0].empty() || fo.momSu[0][0] <= 0)
        { std::printf("  FAIL the Su value did not reach momSu\n"); ++failures; }
    }

    // ---- 2. the legacy spelling still works (OF keeps it, so brae must too) ----
    {
        const std::string c = mkCase(root + "/legacy",
            "momentumSource\n{\n    type            vectorSemiImplicitSource;\n"
            "    selectionMode   all;\n    volumeMode      specific;\n"
            "    injectionRateSuSp\n    {\n        U    ((5 0 0) 0);\n    }\n}\n");
        const FvOptionsData fo = readFvOptions(c, {}, V, nC);
        std::printf("  `injectionRateSuSp`: count=%d hasMomentum=%d\n", fo.count, (int)fo.hasMomentum);
        if (fo.count != 1 || !fo.hasMomentum)
        { std::printf("  FAIL the 2112-and-earlier spelling regressed\n"); ++failures; }
    }

    // ---- 3. NEITHER key: loud, not silent ----
    {
        const std::string c = mkCase(root + "/neither",
            "momentumSource\n{\n    type            vectorSemiImplicitSource;\n"
            "    selectionMode   all;\n    volumeMode      specific;\n}\n");
        const FvOptionsData fo = readFvOptions(c, {}, V, nC);
        std::printf("  neither key: count=%d unsupported=%zu\n", fo.count, fo.unsupported.size());
        if (fo.unsupported.empty())
        {
            std::printf("  FAIL a source with no readable numbers was skipped in silence. The driver only\n"
                        "       refuses on `unsupported`, so this would run with the force missing.\n");
            ++failures;
        }
    }

    // ---- 4. the time window reaches the caller ----
    {
        const std::string c = mkCase(root + "/window",
            "momentumSource\n{\n    type            vectorSemiImplicitSource;\n"
            "    timeStart       0.5;\n    duration        2;\n"
            "    selectionMode   all;\n    volumeMode      specific;\n"
            "    sources\n    {\n        U    ((5 0 0) 0);\n    }\n}\n");
        const FvOptionsData fo = readFvOptions(c, {}, V, nC);
        std::printf("  window: %zu recorded", fo.windows.size());
        if (!fo.windows.empty())
            std::printf(" -> [%g, %g]", (double)fo.windows[0].start,
                        (double)(fo.windows[0].start + fo.windows[0].duration));
        std::printf("\n");
        if (fo.windows.size() != 1 || fo.windows[0].start != scalar(0.5) || fo.windows[0].duration != scalar(2))
        {
            std::printf("  FAIL timeStart/duration did not reach the caller, so a transient driver has\n"
                        "       nothing to refuse on and would apply the source outside its window\n");
            ++failures;
        }
        // and a source WITHOUT a window must record none -- OF's default timeStart_ = -1 is "always on",
        // which must not be mistaken for a window starting at -1.
        const std::string c2 = mkCase(root + "/nowindow",
            "momentumSource\n{\n    type            vectorSemiImplicitSource;\n"
            "    selectionMode   all;\n    volumeMode      specific;\n"
            "    sources\n    {\n        U    ((5 0 0) 0);\n    }\n}\n");
        const FvOptionsData fo2 = readFvOptions(c2, {}, V, nC);
        if (!fo2.windows.empty())
        {
            std::printf("  FAIL a source with no timeStart recorded a window, so an always-on source would\n"
                        "       be refused for a run it is perfectly valid over\n");
            ++failures;
        }
    }

    std::printf("fvoptions_sources_key: %d failures\n", failures);
    return failures ? 1 : 0;
}
