// Focused SIMPLE residualControl contract tests. The sweep below mirrors gpuSimpleFoam's ordering:
// p initial residual, one vector-U criterion over valid components, then every solved turbulence scalar.
#include "force_history.cuh"
#include "residual_control.cuh"
#include <array>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <utility>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

void expect(const char* what, bool got, bool want)
{
    if (got == want) return;
    std::printf("  FAIL %s: got %s, want %s\n", what, got ? "true" : "false", want ? "true" : "false");
    failures++;
}

void expectCount(const char* what, int got, int want)
{
    if (got == want) return;
    std::printf("  FAIL %s: got %d, want %d\n", what, got, want);
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

struct SolveResidual
{
    scalar initial = 0;
};

bool runSweep(ResidualControl& rc, const SolveResidual& p, const SolveResidual& ux,
              const SolveResidual& uy, const SolveResidual& uz,
              const std::vector<std::pair<std::string, SolveResidual>>& turbulence,
              const std::array<bool, 3>& validU)
{
    rc.beginIteration();
    bool achieved = true;
    achieved = rc.ok(p.initial, "p") && achieved;
    achieved = rc.okVelocity(ux.initial, uy.initial, uz.initial, validU) && achieved;
    for (const auto& entry : turbulence)
        achieved = rc.ok(entry.second.initial, entry.first) && achieved;
    return rc.converged(achieved);
}

bool metadataContains(const std::string& path, const std::string& value)
{
    std::ifstream in(path);
    std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return contents.find(value) != std::string::npos;
}

}   // namespace

int main()
{
    const std::string base = "/tmp/brae_residual_control";
    std::filesystem::remove_all(base);
    const std::array<bool, 3> allU = {true, true, true};

    // No dictionary and an empty dictionary are both non-converging; active is not equivalent to non-empty.
    {
        const FoamDict fv = write(base + "/none", "    nNonOrthogonalCorrectors 0;");
        ResidualControl rc(rcOf(fv));
        expect("no residualControl", runSweep(rc, {1e-30}, {1e-30}, {1e-30}, {1e-30}, {}, allU), false);
    }
    {
        const FoamDict fv = write(base + "/empty", "    residualControl\n    {\n    }");
        ResidualControl rc(rcOf(fv));
        expect("empty residualControl", runSweep(rc, {1e-30}, {1e-30}, {1e-30}, {1e-30}, {}, allU), false);
    }

    // A scalar criterion is sufficient; unlisted solved fields are not criteria.
    {
        const FoamDict fv = write(base + "/p-only", "    residualControl\n    {\n        p 1e-2;\n    }");
        ResidualControl rc(rcOf(fv));
        expect("p only passes", runSweep(rc, {1e-4}, {1e30}, {1e30}, {1e30}, {}, allU), true);
        expect("p only fails", runSweep(rc, {1e-1}, {1e-30}, {1e-30}, {1e-30}, {}, allU), false);
    }

    // U is the maximum of all three solved components in 3D, not Ux alone.
    {
        const FoamDict fv = write(base + "/u-3d", "    residualControl\n    {\n        U 1e-5;\n    }");
        ResidualControl rc(rcOf(fv));
        expect("3D U passes when all components pass",
               runSweep(rc, {1}, {1e-6}, {2e-6}, {3e-6}, {}, allU), true);
        expect("3D Uy failure blocks U convergence",
               runSweep(rc, {1}, {1e-6}, {2e-5}, {3e-6}, {}, allU), false);
    }

    // Empty-patch geometry supplies the valid-component mask. The unsolved normal direction is ignored;
    // wedge geometry is not mistaken for an empty direction and remains available for swirl.
    {
        FvPatch empty;
        empty.type = "empty";
        empty.size = 2;
        empty.nf = {{0, 0, 1}, {0, 0, 1}};
        const auto valid2D = ResidualControl::validVelocityComponents({empty});
        expect("2D empty patch keeps Ux", valid2D[0], true);
        expect("2D empty patch keeps Uy", valid2D[1], true);
        expect("2D empty patch removes Uz", valid2D[2], false);

        const FoamDict fv = write(base + "/u-2d", "    residualControl\n    {\n        U 1e-5;\n    }");
        ResidualControl rc(rcOf(fv));
        expect("2D U ignores unsolved Uz",
               runSweep(rc, {1}, {1e-6}, {2e-6}, {1e30}, {}, valid2D), true);

        FvPatch wedge;
        wedge.type = "wedge";
        wedge.size = 2;
        wedge.nf = {{1, 0, 0}, {1, 0, 0}};
        const auto validWedge = ResidualControl::validVelocityComponents({wedge});
        expect("wedge does not remove a solved component", validWedge == allU, true);
    }

    // Every matched criterion participates, including turbulence transport fields.
    {
        const FoamDict fv = write(base + "/all", "    residualControl\n    {\n"
                                  "        p 1e-4;\n        U 1e-5;\n        k 1e-5;\n        omega 1e-5;\n    }");
        ResidualControl rc(rcOf(fv));
        const std::vector<std::pair<std::string, SolveResidual>> ko = {
            {"k", {1e-6}}, {"omega", {2e-6}}
        };
        expect("p U k omega pass together",
               runSweep(rc, {1e-5}, {1e-6}, {2e-6}, {3e-6}, ko, allU), true);
        expect("one failing turbulence field blocks convergence",
               runSweep(rc, {1e-5}, {1e-6}, {2e-6}, {3e-6},
                        {{"k", {2e-5}}, {"omega", {2e-6}}}, allU), false);
    }

    // The same report path covers the alternative turbulence scalar names used by the SA and k-epsilon models.
    {
        const FoamDict fv = write(base + "/other-turbulence", "    residualControl\n    {\n"
                                  "        epsilon 1e-5;\n        nuTilda 1e-5;\n    }");
        ResidualControl rc(rcOf(fv));
        expect("epsilon and nuTilda pass",
               runSweep(rc, {1}, {1}, {1}, {1},
                        {{"epsilon", {1e-6}}, {"nuTilda", {2e-6}}}, allU), true);
        expect("nuTilda failure blocks convergence",
               runSweep(rc, {1}, {1}, {1}, {1},
                        {{"epsilon", {1e-6}}, {"nuTilda", {2e-5}}}, allU), false);
    }

    // Regex keys are one dictionary entry but can consume both solved fields.
    {
        const FoamDict fv = write(base + "/regex", "    residualControl\n    {\n        \"(k|omega)\" 1e-5;\n    }");
        const FoamDict* dict = rcOf(fv);
        ResidualControl rc(dict);
        expect("regex k and omega pass",
               runSweep(rc, {1}, {1}, {1}, {1},
                        {{"k", {1e-6}}, {"omega", {2e-6}}}, allU), true);
        expect("regex omega failure blocks convergence",
               runSweep(rc, {1}, {1}, {1}, {1},
                        {{"k", {1e-6}}, {"omega", {2e-5}}}, allU), false);
        expect("one regex entry is consumed", dict->queried.count("(k|omega)") == 1, true);
    }

    // Unknown-only input has no performed check and cannot converge.
    {
        const FoamDict fv = write(base + "/unknown", "    residualControl\n    {\n        pressureResidual 1e-5;\n    }");
        const FoamDict* dict = rcOf(fv);
        ResidualControl rc(dict);
        expect("unknown field only", runSweep(rc, {1e-30}, {1e-30}, {1e-30}, {1e-30}, {}, allU), false);
        expect("unknown field was not falsely consumed", dict->queried.count("pressureResidual") == 0, true);
    }

    // A failing early criterion must not short-circuit later lookups: all four valid entries are checked/read.
    {
        const FoamDict fv = write(base + "/no-short-circuit", "    residualControl\n    {\n"
                                  "        p 1e-4;\n        U 1e-5;\n        k 1e-5;\n        omega 1e-5;\n    }");
        const FoamDict* dict = rcOf(fv);
        ResidualControl rc(dict);
        const bool got = runSweep(rc, {1e-2}, {1e-6}, {2e-6}, {3e-6},
                                  {{"k", {1e-6}}, {"omega", {2e-6}}}, allU);
        expect("early p failure blocks convergence", got, false);
        expectCount("later criteria are still checked", rc.checked(), 4);
        expect("p consumed", dict->queried.count("p") == 1, true);
        expect("U consumed", dict->queried.count("U") == 1, true);
        expect("k consumed", dict->queried.count("k") == 1, true);
        expect("omega consumed", dict->queried.count("omega") == 1, true);
    }

    // The sweep consumes the solver's initial residual fields for convergence checks.
    {
        const FoamDict fv = write(base + "/initial", "    residualControl\n    {\n        p 1e-5;\n        U 1e-5;\n    }");
        ResidualControl rc(rcOf(fv));
        expect("reported initial residual controls convergence",
               runSweep(rc, {1e-6}, {2e-6}, {3e-6}, {4e-6}, {}, allU), true);
    }

    // ForceHistoryWriter enforces one sample per completed iteration for both terminal reasons and persists
    // the converged reason that gpuSimpleFoam supplies when residualControl stops the run.
    {
        ForceCoeffsConfig cfg;
        cfg.name = "converged";
        ForceHistoryWriter history(base + "/force", cfg, 0);
        history.sample(1, 1, 0, ForceResult{}, ForceCoeffs{});
        history.sample(2, 2, 0, ForceResult{}, ForceCoeffs{});
        history.finish("converged", 2);
        expect("converged force-history stopping reason",
               metadataContains(history.metadataPath(), "stopping_reason=converged"), true);
        expect("converged force-history sample count",
               metadataContains(history.metadataPath(), "sample_count=2")
               && metadataContains(history.metadataPath(), "completed_iterations=2"), true);
    }

    if (failures == 0) std::printf("residual_control: all focused semantics passed\n");
    std::printf("residual_control: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
