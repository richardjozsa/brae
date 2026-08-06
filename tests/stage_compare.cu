// stage_compare -- compare one SIMPLE stage between brae and OpenFOAM, in dependency order.
//
// Phase 0 of docs/rhosimplefoam-restage-plan.md. SIMPLE is segregated, so an error in one stage shows up
// DOWNSTREAM and never upstream:
//
//     rAU/rAtU -> HbyA -> phiHbyA -> phid -> p -> U -> he -> turbulence
//
// Walking that order localises a defect to a single stage. The squareBend investigation did it by hand and
// landed on HbyA: +5.11% on internal faces while its boundary was exactly -0.5 (= the case's massFlowRate),
// with phid inheriting precisely that ratio. This turns that afternoon into a command.
//
// It reports more than an L2 norm on purpose. "The fields differ by 5%" does not say whether the error is
// at walls, at the inlet, or interior -- and on a 112k-cell mesh that distinction is most of the diagnosis.
// So it also prints where the worst cells are and how the error is distributed.
//
//   stage_compare <ofTimeDir> <braeDumpDir> <stage> [tol]
//
//     stage_compare /tmp/of/1 /tmp/braedump rAU
//
// OF side reads an OpenFOAM field (written by tools/dumpPEqn as stage_<name>); brae side reads the plain
// text stage_dump.cuh writes. Only ONE side has to be a real OpenFOAM field, which keeps the instrument
// simple -- an instrument that needs debugging is worse than none.

#include "foam_field_reader.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

using namespace brae;

namespace {

struct Field
{
    std::vector<scalar> v;      // flattened: scalars n, vectors 3n
    int comps = 1;
    std::size_t n() const { return comps ? v.size() / static_cast<std::size_t>(comps) : 0; }
};

// brae side: "n <count> <comps>" then one entry per line.
Field readBrae(const std::string& path)
{
    Field f;
    std::ifstream in(path);
    if (!in) return f;
    std::string tag;
    std::size_t n = 0;
    in >> tag >> n >> f.comps;
    if (tag != "n") { f.comps = 0; return f; }
    f.v.reserve(n * static_cast<std::size_t>(f.comps));
    scalar x;
    while (in >> x) f.v.push_back(x);
    return f;
}

// OF side: a real OpenFOAM field. readField already resolves includes/macros, and handles both
// volScalarField and surfaceScalarField -- a surface field's internalField is the internal-face list,
// which is exactly what brae's face dumps hold.
Field readOF(const std::string& path, int comps)
{
    Field f;
    f.comps = comps;
    std::ifstream probe(path);
    if (!probe) { f.comps = 0; return f; }
    if (comps == 1)
    {
        const FieldData<scalar> d = readField<scalar>(path);
        f.v = d.internalField;
    }
    else
    {
        const FieldData<vector> d = readField<vector>(path);
        f.v.reserve(d.internalField.size() * 3);
        for (const vector& u : d.internalField) { f.v.push_back(u.x); f.v.push_back(u.y); f.v.push_back(u.z); }
    }
    return f;
}

}   // namespace

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::printf("usage: stage_compare <ofTimeDir> <braeDumpDir> <stage> [tol]\n"
                    "  stages: rAU rAtU HbyA fluxHbyA phiHbyA0 phid phidB\n");
        return 2;
    }
    const std::string ofDir = argv[1], brDir = argv[2], stage = argv[3];
    const scalar tol = (argc > 4) ? std::atof(argv[4]) : scalar(1e-6);

    // Vector stages, by name. Reading a vector field as a scalar aborts in the token stream rather than
    // silently mis-comparing, which is the right failure -- but the list has to be kept current.
    const bool isVector = (stage == "HbyA" || stage == "Upred" || stage == "U");
    const Field br = readBrae(brDir + "/stage_" + stage);
    if (br.comps == 0 || br.v.empty())
    {
        std::printf("  FAIL brae dump %s/stage_%s missing or empty -- was BRAE_DUMP_STAGE set?\n",
                    brDir.c_str(), stage.c_str());
        return 1;
    }
    // fluxHbyA has no OF counterpart of its own: OF's equivalent is phiHbyA0 before the rho weighting,
    // which it does not write separately. Compared against phiHbyA0 only when asked for explicitly.
    // OF writes the momentum stages as stage_<name> (added by tools/dumpPEqn) but phid/phiHbyA0 under
    // their own OpenFOAM names, because those are real fields in pcEqn.H rather than harness additions.
    // Try both rather than making the caller remember which is which.
    Field of = readOF(ofDir + "/stage_" + stage, isVector ? 3 : 1);
    if (of.comps == 0 || of.v.empty()) of = readOF(ofDir + "/" + stage, isVector ? 3 : 1);
    if (of.comps == 0 || of.v.empty())
    {
        std::printf("  FAIL OF field %s/stage_%s missing -- was the case run with tools/dumpPEqn?\n",
                    ofDir.c_str(), stage.c_str());
        return 1;
    }

    const std::size_t n = std::min(of.n(), br.n());
    if (of.n() != br.n())
        std::printf("  NOTE sizes differ: OF %zu, brae %zu -- comparing the first %zu\n", of.n(), br.n(), n);
    if (n == 0) { std::printf("  FAIL nothing to compare\n"); return 1; }

    // Per-entry magnitude difference, so a vector is judged on the vector and not component by component.
    std::vector<std::pair<scalar, std::size_t>> err;
    err.reserve(n);
    scalar num = 0, den = 0;
    for (std::size_t i = 0; i < n; ++i)
    {
        scalar d2 = 0, o2 = 0;
        for (int c = 0; c < of.comps; ++c)
        {
            const scalar a = of.v[i * of.comps + c], b = br.v[i * br.comps + c];
            d2 += (a - b) * (a - b);
            o2 += a * a;
        }
        num += d2;
        den += o2;
        err.emplace_back(std::sqrt(d2), i);
    }
    const scalar l2 = (den > 0) ? std::sqrt(num / den) : std::sqrt(num);

    std::printf("  stage %-10s n=%zu  L2rel %.4e  tol %.0e  %s\n",
                stage.c_str(), n, (double)l2, (double)tol, (l2 <= tol) ? "OK" : "FAIL");

    if (l2 > tol)
    {
        // WHERE the error lives is most of the diagnosis. A defect concentrated in a contiguous index
        // range is usually one patch's adjacent cells; one spread evenly is a scheme or coefficient.
        std::sort(err.begin(), err.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
        std::printf("       worst entries:");
        for (std::size_t k = 0; k < std::min<std::size_t>(6, err.size()); ++k)
            std::printf(" [%zu]=%.3e", err[k].second, (double)err[k].first);
        std::printf("\n");

        scalar tot = 0;
        for (const auto& e : err) tot += e.first;
        scalar top = 0;
        std::size_t k = 0;
        for (; k < err.size() && top < 0.5 * tot; ++k) top += err[k].first;
        std::printf("       half the total error is carried by %zu of %zu entries (%.1f%%)"
                    " -- concentrated means one patch or region; spread means a scheme.\n",
                    k, n, 100.0 * (double)k / (double)n);
    }
    return (l2 <= tol) ? 0 : 1;
}
