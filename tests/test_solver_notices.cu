// E2/E3: brae must SAY when it runs a different linear solver than the case asked for.
//
// Found by dict_audit, which reported solvers/p/solver, solvers/p/smoother and
// solvers/(U|h|e)/preconditioner as read off disk by nobody. brae runs AMG-PCG for pressure and
// Jacobi-preconditioned BiCGStab elsewhere, whatever the dict says.
//
// This NOTICES rather than refuses, and the distinction is the whole design decision. A substituted
// linear solver is not a wrong answer: it solves the same linear system to the same tolerance, so the
// converged SIMPLE result is unchanged. What changes is the iteration count, the cost, and -- at the loose
// per-step relTol that SIMPLE uses (0.01 on p is the norm) -- the intermediate fields, because two solvers
// stop at different points. Someone diffing brae's "Solving for p" against OF's has a right to know.
//
// The negative control is the point of this test. Asserting "the notice appears" is satisfied by a
// function that prints unconditionally; what has to hold is that a dict asking for EXACTLY what brae runs
// stays silent. Both directions are checked below.

#include "linear_solver_setup.cuh"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <unistd.h>

using namespace brae;

namespace {

int failures = 0;

void check(const char* what, bool ok, const std::string& detail = "")
{
    if (ok) return;
    std::printf("  FAIL %s%s\n", what, detail.empty() ? "" : ("  [" + detail + "]").c_str());
    failures++;
}

// Run `fn` with stderr captured to a file, and return what it wrote. The notices go to stderr through
// fprintf, so this reads the real output rather than a test-only hook that could drift from it.
template <typename F>
std::string captureStderr(const std::string& tmpFile, F&& fn)
{
    std::fflush(stderr);
    const int saved = dup(fileno(stderr));
    FILE* redirected = std::freopen(tmpFile.c_str(), "w", stderr);
    (void)redirected;
    fn();
    std::fflush(stderr);
    dup2(saved, fileno(stderr));
    close(saved);
    std::ifstream in(tmpFile);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

std::string writeFvSolution(const std::string& dir, const std::string& solversBody)
{
    std::filesystem::create_directories(dir + "/system");
    std::ofstream(dir + "/system/fvSolution")
        << "FoamFile { version 2.0; format ascii; class dictionary; object fvSolution; }\n"
        << "solvers\n{\n" << solversBody << "}\n"
        << "SIMPLE { nNonOrthogonalCorrectors 0; }\n"
        << "relaxationFactors { equations { U 0.7; } }\n";
    return dir;
}

bool has(const std::string& hay, const std::string& needle)
{
    return hay.find(needle) != std::string::npos;
}

}   // namespace

int main()
{
    const std::string tmp = "/tmp/brae_solver_notices";
    std::filesystem::remove_all(tmp);

    // ---- POSITIVE: a stock-tutorial fvSolution, none of whose choices brae actually runs ----
    {
        const std::string dir = writeFvSolution(tmp + "/asks",
            "    p { solver GAMG; smoother DICGaussSeidel; tolerance 1e-10; relTol 0.01; }\n"
            "    \"(U|h|e)\" { solver PBiCGStab; preconditioner DILU; tolerance 1e-10; relTol 0.1; }\n");
        const FoamDict fv = readDict(dir + "/system/fvSolution");

        const std::string out = captureStderr(tmp + "/asks.err", [&]
        {
            DeviceSimpleControls ctl;
            ctl.turbulent = false;
            readLinearSolverControls(fv, "epsilon", ctl);
            readEnergySolverControls(fv, /*internalEnergy=*/true);
        });

        check("p solver substitution is reported", has(out, "solvers/p solver") && has(out, "GAMG") && has(out, "AMG-PCG"), out);
        check("p smoother is reported ignored", has(out, "solvers/p smoother") && has(out, "DICGaussSeidel"), out);
        check("U solver substitution is reported", has(out, "solvers/U solver") && has(out, "PBiCGStab"), out);
        check("U preconditioner is reported", has(out, "solvers/U preconditioner") && has(out, "DILU"), out);
        check("energy preconditioner is reported", has(out, "solvers/e preconditioner") && has(out, "DILU"), out);
        // The wording must carry the reason, not just the fact -- a bare "ignored" would read as a bug.
        check("the notice explains the consequence", has(out, "iteration count and cost differ"), out);
    }

    // ---- NEGATIVE CONTROL: ask for exactly what brae runs -> silence ----
    //
    // notice() de-duplicates by (kind, subject, detail) in a process-wide set, so this MUST use different
    // field names than the positive case above, or the silence would just be de-duplication.
    {
        const std::string dir = writeFvSolution(tmp + "/matches",
            "    k { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }\n"
            "    omega { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-8; relTol 0.1; }\n");
        const FoamDict fv = readDict(dir + "/system/fvSolution");

        const std::string out = captureStderr(tmp + "/matches.err", [&]
        {
            DeviceSimpleControls ctl;
            ctl.turbulent = true;
            ctl.sa = false;
            readLinearSolverControls(fv, "omega", ctl);
        });

        check("smoothSolver+symGaussSeidel on k produces no notice", !has(out, "solvers/k"), out);
        check("smoothSolver+symGaussSeidel on omega produces no notice", !has(out, "solvers/omega"), out);
    }

    // ---- NEGATIVE CONTROL 2: no `solvers` entries at all -> nothing to report ----
    {
        const std::string dir = writeFvSolution(tmp + "/empty", "");
        const FoamDict fv = readDict(dir + "/system/fvSolution");
        const std::string out = captureStderr(tmp + "/empty.err", [&]
        {
            DeviceSimpleControls ctl;
            ctl.turbulent = false;
            readLinearSolverControls(fv, "epsilon", ctl);
        });
        check("an fvSolution with no solver entries is silent", !has(out, "NOTICE"), out);
    }

    std::printf("solver_notices: %d failures\n", failures);
    return failures ? 1 : 0;
}
