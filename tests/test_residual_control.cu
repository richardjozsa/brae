// An EMPTY residualControl must never report convergence.
//
// OF simpleControl::criteriaSatisfied (simpleControl.C:49-):
//     if (residualControl_.empty()) { return false; }
//     bool achieved = true;
//     bool checked  = false;    // "safety that some checks were indeed performed"
//     ... return achieved && checked;
//
// brae's test was `hasRC && ok(p) && ok(U)`, where ok() returns TRUE for a field the dict does not list
// (target -1, "not a criterion" -- which is correct on its own). So `residualControl { }` made every
// check vacuously pass: the run stopped after ONE iteration, printed "SIMPLE solution converged in 1
// iterations", and wrote a full set of plausible-looking fields. That is worse than hanging, because
// nothing about the output says the solve never happened. It was found while trying to run a 60-iteration
// comparison against OF that kept coming back with a single time directory.
//
// The rule needs BOTH halves. Dropping only the empty-dict check still lets `residualControl { foo 1e-5; }`
// -- a dict naming a field brae does not check -- converge instantly, which is exactly what OF's `checked`
// flag exists to prevent. So the cases below cover an empty dict, an unrelated-field dict, and the normal
// path, and the negative control asserts a real criterion is still honoured.

#include "residual_control.cuh"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace brae;

namespace {

int failures = 0;

void expect(const char* what, bool got, bool want)
{
    if (got == want) return;
    std::printf("  FAIL %s: got %s, want %s\n", what, got ? "converged" : "not converged",
                want ? "converged" : "not converged");
    failures++;
}

const FoamDict* rcOf(const FoamDict& fvSolution)
{
    const FoamDict* simple = fvSolution.subDict("SIMPLE");
    return simple ? simple->subDict("residualControl") : nullptr;
}

FoamDict write(const std::string& dir, const std::string& body)
{
    std::filesystem::create_directories(dir);
    {
        std::ofstream f(dir + "/fvSolution");
        f << "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
          << "solvers { }\n"
          << "SIMPLE\n{\n" << body << "\n}\n";
    }
    return readDict(dir + "/fvSolution");
}

// One convergence test, exactly as the drivers do it: residuals for p and Ux against their targets.
bool converged(const FoamDict* dict, scalar resP, scalar resU)
{
    ResidualControl rc(dict);
    rc.beginIteration();
    const bool achieved = rc.ok(resP, "p") && rc.ok(resU, "U");
    return rc.converged(achieved);
}

}   // namespace

int main()
{
    const std::string base = "/tmp/brae_residual_control";
    std::filesystem::remove_all(base);

    // 1. EMPTY residualControl: never converges, however small the residuals.
    {
        const FoamDict fv = write(base + "/empty", "    residualControl\n    {\n    }");
        expect("empty residualControl with tiny residuals", converged(rcOf(fv), 1e-30, 1e-30), false);
    }

    // 2. No residualControl at all: same answer, by a different route (no dict rather than empty dict).
    {
        const FoamDict fv = write(base + "/none", "    nNonOrthogonalCorrectors 0;");
        expect("absent residualControl", converged(rcOf(fv), 1e-30, 1e-30), false);
    }

    // 3. A dict that lists only fields this check does not look at. OF's `checked` flag is what stops
    //    this; without it the unlisted-field rule alone makes it converge on iteration one.
    {
        const FoamDict fv = write(base + "/other", "    residualControl\n    {\n        nuTilda 1e-5;\n    }");
        expect("residualControl naming only unchecked fields", converged(rcOf(fv), 1e-30, 1e-30), false);
    }

    // 4. Normal path: a real criterion, met -> converged.
    {
        const FoamDict fv = write(base + "/real", "    residualControl\n    {\n        p 1e-2;\n        U 1e-3;\n    }");
        expect("both criteria met", converged(rcOf(fv), 1e-4, 1e-5), true);
        // NEGATIVE CONTROL: the criteria must still be able to FAIL. Without this, "return false always"
        // would pass cases 1-3 and this test would assert nothing about the normal path.
        expect("p above its target", converged(rcOf(fv), 1e-1, 1e-5), false);
        expect("U above its target", converged(rcOf(fv), 1e-4, 1e-2), false);
    }

    // 5. A partially-specified dict is still honoured: p listed, U not. The unlisted field is not a
    //    criterion (OF), but the listed one is a performed check, so convergence is allowed.
    {
        const FoamDict fv = write(base + "/partial", "    residualControl\n    {\n        p 1e-2;\n    }");
        expect("only p listed, p met", converged(rcOf(fv), 1e-4, 1e30), true);
        expect("only p listed, p not met", converged(rcOf(fv), 1e-1, 1e-30), false);
    }

    if (failures == 0) std::printf("  OK   empty/absent/irrelevant residualControl never converges; real criteria honoured\n");
    std::printf("residual_control: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
