// relaxationFactors: which factor applies on the LAST outer corrector, and why that is not a tuning
// detail.
//
// Nearly every PIMPLE tutorial writes some form of
//     relaxationFactors { equations { U 0.8; ".*Final" 1; } }
// and it only does anything because OF looks the name up as a REGEX after appending "Final" on the final
// iteration (GeometricField::relax and fvMatrix::relax both go through psi.select(isFinalIteration()),
// and solution.C:341,383 match with keyType::REGEX). The final corrector therefore runs UNRELAXED.
//
// brae had no notion of a Final factor at all -- it applied the ordinary one on every corrector,
// including the last. That is not a slower convergence: the final corrector is what makes a PIMPLE step
// satisfy momentum and continuity together, so relaxing it leaves a residue the next step inherits.
//
// Two things make this worth its own file rather than a line in a sweep:
//
//   - It is a PARSE that is silent when wrong. A missing ".*Final" does not error; it just relaxes, and
//     the run still completes. brae's dict audit did flag `relaxationFactors/.*Final` as unread, which
//     is the only reason it was ever noticed.
//   - The regex is load-bearing. A literal-only lookup finds `U` and misses `UFinal` entirely, so the
//     feature silently degrades to the old behaviour. Leg 2 is aimed straight at that.
//
// Leg 4 pins OF's LEGACY (flat) form, which is not the modern one and not a simplification of it: a flat
// relaxationFactors dict becomes EQUATION relaxation in full, while only keys beginning `p` or `rho` are
// promoted to FIELD relaxation (solution.C:82-100).
#include "foam_dict.cuh"
#include "linear_solver_setup.cuh"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace brae;

namespace {
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       std::printf("  ok:   %s\n", what);
}

DeviceSimpleControls parse(const std::string& body)
{
    const std::string f = "test_relaxation_final.tmp";
    { std::ofstream o(f); o << "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n" << body; }
    const FoamDict d = readDict(f);
    DeviceSimpleControls c;
    readRelaxationFactors(d, c);
    std::filesystem::remove(f);
    return c;
}
} // namespace

int main()
{
    std::printf("== relaxationFactors: the final-corrector factor ==\n");

    // ---- Leg 1: the modern nested form, with an explicit Final entry --------------------------------
    {
        const DeviceSimpleControls c = parse(
            "relaxationFactors { equations { U 0.8; nuTilda 0.8; \".*Final\" 1; }\n"
            "                    fields    { p 0.4; pFinal 1; } }\n");
        check(c.relaxU == scalar(0.8),      "U relaxes at 0.8 on an ordinary corrector");
        check(c.relaxUFinal == scalar(1),   "...and at 1.0 on the final one, via the \".*Final\" regex");
        check(c.relaxP == scalar(0.4),      "p field-relaxes at 0.4 ordinarily");
        check(c.relaxPFinal == scalar(1),   "...and at 1.0 on the final corrector");
    }

    // ---- Leg 2: the regex is what does the work -----------------------------------------------------
    // A literal lookup finds `U` and misses `UFinal`, so the Final factor would silently fall back to
    // the ordinary one and the defect would be exactly as before. This leg fails on a literal-only
    // implementation and passes on OF's regex one -- nothing else distinguishes them.
    {
        const DeviceSimpleControls c = parse(
            "relaxationFactors { equations { \"U|nuTilda\" 0.7; \".*Final\" 1; } }\n");
        check(c.relaxU == scalar(0.7),    "a REGEX key (\"U|nuTilda\") supplies the ordinary factor");
        check(c.relaxUFinal == scalar(1), "a REGEX key (\".*Final\") supplies the final factor");
    }

    // ---- Leg 3: no Final entry -> NO relaxation, not the ordinary factor ----------------------------
    // The counter-intuitive half of OF's rule, and the one worth pinning. relax() is guarded by
    // `if (relaxField(name))`, so an unmatched "UFinal" means relax() is simply not called -- the final
    // corrector runs unrelaxed even though the case named no Final entry at all. Falling back to the
    // ordinary factor here would look reasonable and would relax a corrector OpenFOAM leaves alone.
    {
        const DeviceSimpleControls c = parse("relaxationFactors { equations { U 0.5; } fields { p 0.3; } }\n");
        check(c.relaxU == scalar(0.5),    "the ordinary factors are still read");
        check(c.relaxUFinal == scalar(1), "no UFinal entry -> the final corrector is UNRELAXED");
        check(c.relaxPFinal == scalar(1), "no pFinal entry -> likewise for p");
    }

    // ---- Leg 4: OF's LEGACY flat form ---------------------------------------------------------------
    // Flat means EQUATION relaxation for everything, and FIELD relaxation only for keys starting p/rho.
    // This is the form LES/vortexShed ships, which is how the whole defect surfaced.
    {
        const DeviceSimpleControls c = parse(
            "relaxationFactors { nuTilda 0.8; U 0.8; p 0.8; \".*Final\" 1.0; }\n");
        check(c.relaxU == scalar(0.8),      "flat form: U takes equation relaxation");
        check(c.relaxUFinal == scalar(1),   "flat form: \".*Final\" still reaches UFinal");
        check(c.relaxP == scalar(0.8),      "flat form: p IS promoted to field relaxation (starts with p)");
        check(c.relaxPFinal == scalar(1),   "flat form: and pFinal comes out unrelaxed");
    }

    // ---- Leg 5: the p/rho filter on the legacy form --------------------------------------------------
    // `".*"` in a FLAT dict must not become a field factor: OF's legacy branch only copies keys
    // beginning p or rho into fieldRelaxDict_, so a catch-all relaxes equations and leaves the pressure
    // field alone. Without the filter brae would field-relax p at 0.7 where OF does not relax it at all.
    {
        const DeviceSimpleControls c = parse("relaxationFactors { \".*\" 0.7; }\n");
        check(c.relaxU == scalar(0.7),   "flat \".*\" relaxes the U EQUATION");
        check(c.relaxP == scalar(1),     "...but does NOT field-relax p");
        check(c.relaxPFinal == scalar(1), "...nor pFinal");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
