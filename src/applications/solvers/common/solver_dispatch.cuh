#pragma once
// Which brae solver owns a case -- the one place that maps an OpenFOAM case to a brae executable.
//
// `brae` is the single command a user types; the case itself names the solver it wants, in controlDict's
// `application` entry, exactly as OpenFOAM does. This header holds that registry: one row per solver brae
// implements. Adding a solver (rhoSimpleFoam, potentialFoam, ...) is one row plus its executable -- no new
// branching in the drivers.
//
// Selection, in order:
//   1. controlDict `application` names a registered solver -> that solver runs it (in-process if it IS this
//      executable, otherwise exec the sibling binary with the same argv).
//   2. controlDict `application` names something brae does not implement -> stop and say so, listing what brae
//      does have. brae never guesses a solver: a wrong one is a silently wrong answer.
//   3. controlDict has no `application` (hand-written cases) -> fall back to the case's fvSchemes ddtSchemes,
//      which still says steady (steadyState) or transient, and pick the registered solver of that kind.
// A case whose `application` disagrees with its ddtSchemes (application simpleFoam + Euler, say) is a case
// error, not something to paper over, so that stops too.
#include "foam_dict.cuh"   // readDict, readFileExpanded
#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>   // getenv/setenv: the one-hand-over-per-run guard
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>   // execv/execvp: hand the case to the solver that owns it

namespace brae {

struct BraeSolver
{
    const char* application;   // OF controlDict `application` this row claims
    const char* exe;           // brae executable that runs it
    bool        transient;     // marches in time (ddtSchemes.default != steadyState)
    const char* what;          // one-line description, shown when a case asks for a solver brae lacks
};

// The registry. Every row has the same shape, including the steady one: `brae` is the launcher AND the steady
// solver's executable, which is a fact about today's build, not a rule the dispatcher should know. It finds out
// by comparing the chosen row's executable with the binary that is actually running (selfExecutableName below)
// -- so the steady case costs no extra process, and the day the steady solver moves into its own binary, only
// this table changes.
inline const std::vector<BraeSolver>& braeSolvers()
{
    static const std::vector<BraeSolver> reg = {
        {"simpleFoam", "brae",            false, "steady incompressible, RAS/laminar"},
        {"pimpleFoam", "brae_pimpleFoam", true,  "transient incompressible, URANS/DES/LES/laminar"},
    };
    return reg;
}

// Filename of the running binary, "" if it cannot be determined (then nothing matches and the chosen solver is
// exec'd, which is the safe direction -- the loop guard in execSolver catches a misconfiguration).
inline std::string selfExecutableName()
{
    std::error_code ec;
    const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", ec);
    return ec ? std::string() : self.filename().string();
}

// ddtSchemes.default as written in system/fvSchemes, "" if the case has no readable ddtSchemes block.
// fvSchemes is not a plain key/value dict (multi-token values, $-vars), so scan the text, bounded by the
// block's own braces -- an empty ddtSchemes must not pick up the next block's `default`.
inline std::string readDdtSchemeWord(const std::string& fvSchemesPath)
{
    std::string text;
    try { text = readFileExpanded(fvSchemesPath); } catch (...) { return ""; }
    const std::size_t blk = text.find("ddtSchemes");
    if (blk == std::string::npos) return "";
    const std::size_t open = text.find('{', blk);
    if (open == std::string::npos) return "";
    const std::size_t close = text.find('}', open);
    std::size_t d = text.find("default", open);
    if (d == std::string::npos || (close != std::string::npos && d > close)) return "";
    d += 7;
    while (d < text.size() && std::isspace(static_cast<unsigned char>(text[d]))) ++d;
    std::string w;
    while (d < text.size() && !std::isspace(static_cast<unsigned char>(text[d])) && text[d] != ';') w += text[d++];
    return w;
}

// The solvers brae has, formatted for an error message.
inline std::string braeSolverList()
{
    std::string s;
    for (const BraeSolver& e : braeSolvers())
    {
        s += "\n    ";
        s += e.application;
        s.append(std::max<std::size_t>(1, 16 - std::strlen(e.application)), ' ');
        s += e.what;
    }
    return s;
}

// Replace this process with the solver that owns the case, forwarding argv unchanged (both drivers take
// `-case <dir>` and a positional case dir). Prefer the binary next to this one, so a build tree runs its own
// solvers and an install runs the installed set; fall back to PATH. Never returns on success.
[[noreturn]] inline void execSolver(const BraeSolver& s, int argc, char** argv, const std::string& why)
{
    // One hand-over per run. If the exec'd solver dispatches again we are in a loop (a renamed binary, or a
    // registry row pointing at the wrong executable), so stop with something readable instead of forking forever.
    if (const char* from = std::getenv("BRAE_DISPATCHED_FROM"))
        throw std::runtime_error(
            std::string("solver dispatch loop: already handed over from '") + from + "' and now asked for '" +
            s.exe + "'. Check that " + s.exe + " is the binary the registry names.");
    setenv("BRAE_DISPATCHED_FROM", s.application, 1);

    std::string exe = s.exe;
    std::error_code ec;
    const std::filesystem::path self = std::filesystem::read_symlink("/proc/self/exe", ec);
    if (!ec)
    {
        const std::filesystem::path sibling = self.parent_path() / s.exe;
        if (std::filesystem::exists(sibling, ec)) exe = sibling.string();
    }
    std::fprintf(stderr, "brae: %s -> %s (%s)\n", why.c_str(), s.application, s.exe);
    std::vector<char*> av;
    av.push_back(const_cast<char*>(exe.c_str()));
    for (int i = 1; i < argc; ++i) av.push_back(argv[i]);
    av.push_back(nullptr);
    execv(exe.c_str(), av.data());        // the sibling/installed path
    execvp(s.exe, av.data());             // else whatever is on PATH
    throw std::runtime_error(
        std::string("cannot start '") + s.exe + "', the brae solver for " + s.application +
        ". Build it with: cmake --build build --target " + s.exe);
}

// Route the case to its solver. Returns only when the running binary IS the chosen solver; otherwise it has
// already exec'd the right one, or thrown because brae does not have it.
inline void dispatchSolver(const std::string& caseDir, int argc, char** argv)
{
    const FoamDict controlDict = readDict(caseDir + "/system/controlDict");
    const std::string application = controlDict.wordOr("application", "");
    const std::string ddt = readDdtSchemeWord(caseDir + "/system/fvSchemes");
    const bool transientCase = !ddt.empty() && ddt != "steadyState";

    const BraeSolver* chosen = nullptr;
    std::string why;
    if (!application.empty())
    {
        for (const BraeSolver& e : braeSolvers())
            if (application == e.application) chosen = &e;
        if (!chosen)
            throw std::runtime_error(
                "controlDict application '" + application + "' is not a solver brae implements yet."
                "\n  brae runs:" + braeSolverList() +
                "\n  brae stops rather than run a case with the wrong solver.");
        // The case names a solver but its ddtSchemes says the other thing -- one of the two is wrong, and
        // guessing which would silently produce the wrong answer.
        if (!ddt.empty() && chosen->transient != transientCase)
            throw std::runtime_error(
                "controlDict application '" + application + "' is " +
                (chosen->transient ? "transient" : "steady") + ", but fvSchemes ddtSchemes.default is '" + ddt +
                "'. Fix one of the two: steady solvers need steadyState, transient solvers need "
                "Euler|backward|CrankNicolson.");
        why = "controlDict application " + application;
    }
    else
    {
        // No `application` entry: the ddt scheme still says steady or transient, so pick that kind of solver.
        for (const BraeSolver& e : braeSolvers())
            if (e.transient == transientCase) { chosen = &e; break; }
        if (!chosen)
            throw std::runtime_error(
                "system/controlDict has no `application` entry and fvSchemes ddtSchemes.default '" + ddt +
                "' matches no brae solver."
                "\n  brae runs:" + braeSolverList());
        why = "no controlDict application, ddtSchemes.default = " + (ddt.empty() ? std::string("(none)") : ddt);
    }

    if (selfExecutableName() == chosen->exe) return;   // already the right binary: solve in this process
    execSolver(*chosen, argc, argv, why);              // does not return
}

}  // namespace brae
