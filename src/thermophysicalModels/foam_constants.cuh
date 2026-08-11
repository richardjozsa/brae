#pragma once
// foam_constants.cuh -- OpenFOAM's universal gas constant, resolved the way OpenFOAM resolves it.
//
// E6. brae had RR compiled in. That was already wrong once: brae carried the CODATA-2018
// 8314.46261815324 while OF v2412 computes 8314.47006650545, an 8.958e-07 relative gap that propagated to
// R, psi and rho in every compressible case (see thermo_parse.cuh for how that was found). Matching the
// NUMBER fixed the default; it did not match the MECHANISM, and the mechanism is the part a user can move
// under brae's feet.
//
// OF builds RR = NA*k from `DimensionedConstants` in its etc/controlDict (physicoChemicalConstants.C:44-52,
// via defineDimensionedConstantWithDefault, so every entry is overridable). Those files are merged in
// findEtcFiles order -- project, then site, then user -- with LATER files winning, so a user's
// ~/.OpenFOAM/<ver>/controlDict silently changes the gas constant for every OF run on that machine while
// brae would carry on with its compiled value.
//
// Two things make this awkward and are handled explicitly rather than ignored:
//
//  1. `unitSet USCS` selects an entirely different constant set. brae is SI throughout, so that is refused
//     rather than approximated -- every density in the run would be wrong by a units factor.
//
//  2. OF's etc/controlDict contains #codeStream directives, which brae's reader refuses on purpose (it
//     would have to mis-parse them into a wrong value). So the DimensionedConstants block is SLICED OUT
//     textually by brace matching and only that slice is parsed. Reading the whole file is not an option
//     and pretending the file does not exist is how E6 got here.
#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "brae_notice.cuh"
#include <cstdlib>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {

// OF v2412's compiled defaults, MEASURED from OF itself (a utility printing the constants), not taken
// from a data sheet. NA is not listed in the shipped etc/controlDict; k is.
inline constexpr scalar foamNAdefault = 6.022141793e23;
inline constexpr scalar foamKdefault  = 1.38065e-23;

// RR [J/(kmol K)] = 1e3 * NA * k. With the defaults above this is exactly 8314.47006650545, which is what
// OF prints for Foam::constant::thermodynamic::RR.
inline constexpr scalar foamRRdefault = 8314.47006650545;

// Standard temperature [K], the default reference point of OF's SENSIBLE enthalpy
// (hConstThermo: Hs = Cp*(T - Tref) + Href, with Tref defaulting to Tstd). Shipped in the same
// DimensionedConstants block as k, under `standard`, so it is overridable the same way.
inline constexpr scalar foamTstdDefault = 298.15;

namespace detail {

// The `Name { ... }` block, brace-matched, or empty. Textual because the surrounding file cannot be parsed.
inline std::string sliceBlock(const std::string& text, const std::string& name)
{
    std::size_t p = 0;
    for (;;)
    {
        p = text.find(name, p);
        if (p == std::string::npos) return {};
        // Must be a whole word, else "DimensionedConstantsFoo" would match.
        const bool lb = (p == 0 || !(std::isalnum((unsigned char)text[p-1]) || text[p-1] == '_'));
        const std::size_t e = p + name.size();
        const bool rb = (e >= text.size() || !(std::isalnum((unsigned char)text[e]) || text[e] == '_'));
        if (lb && rb) break;
        p = e;
    }
    const std::size_t open = text.find('{', p);
    if (open == std::string::npos) return {};
    int depth = 0;
    for (std::size_t i = open; i < text.size(); ++i)
    {
        if (text[i] == '{') ++depth;
        else if (text[i] == '}' && --depth == 0) return text.substr(p, i - p + 1);
    }
    return {};
}

// OF's findEtcFiles order, merged with LATER winning. Only the locations that can actually carry a
// DimensionedConstants override are consulted.
inline std::vector<std::string> etcControlDicts()
{
    std::vector<std::string> out;
    const char* proj = std::getenv("WM_PROJECT_DIR");
    if (proj && *proj) out.push_back(std::string(proj) + "/etc/controlDict");
    const char* home = std::getenv("HOME");
    const char* ver  = std::getenv("WM_PROJECT_VERSION");
    if (home && *home && ver && *ver)
        out.push_back(std::string(home) + "/.OpenFOAM/" + ver + "/controlDict");
    return out;
}

// Walk every etc/controlDict that can carry a DimensionedConstants override, LATER winning, and hand the
// callback that file's `<unitSet>Coeffs` dict. Shared so RR and Tstd read the SAME block in the SAME
// order -- a second copy of this loop would be a second place for the override order to drift.
template <class Fn>
void scanDimensionedConstants(Fn&& fn)
{
    for (const std::string& path : etcControlDicts())
    {
        std::ifstream in(path);
        if (!in) continue;
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const std::string block = sliceBlock(text, "DimensionedConstants");
        if (block.empty()) continue;

        // TokenStream reads a path, so the slice is materialised. Cheap, once per run, and it keeps
        // the token reader untouched. Named with the pid so concurrent ctest jobs cannot collide.
        FoamDict dc;
        const std::string tmpPath =
            (std::filesystem::temp_directory_path() /
             ("brae_dimconst_" + std::to_string(static_cast<long>(::getpid())) + ".dict")).string();
        try
        {
            { std::ofstream out(tmpPath); out << block << "\n"; }
            dc = readDict(tmpPath);
            std::error_code ec;
            std::filesystem::remove(tmpPath, ec);
        }
        catch (const std::exception&)
        {
            std::error_code ec;
            std::filesystem::remove(tmpPath, ec);
            noticeIgnored("OpenFOAM DimensionedConstants",
                          path + " -- could not be parsed; using brae's built-in OF v2412 constants");
            continue;
        }

        const FoamDict* d = dc.subDict("DimensionedConstants");
        if (!d) continue;

        const std::string unitSet = d->wordOr("unitSet", "SI");
        if (unitSet != "SI")
        {
            throw std::runtime_error(
                "brae: OpenFOAM's " + path + " selects `unitSet " + unitSet + "`. brae's thermophysics "
                "is SI throughout, so every density in the run would be wrong by a units factor. brae "
                "refuses rather than silently computing in the wrong unit system.");
        }
        if (const FoamDict* si = d->subDict(unitSet + "Coeffs")) fn(path, *si);
    }
}

}   // namespace detail

// The gas constant this run should use. Cached: the answer cannot change during a run, and the notice
// below must be emitted once, not per patch.
inline scalar foamRR()
{
    static const scalar cached = []() -> scalar
    {
        scalar NA = foamNAdefault, k = foamKdefault;
        std::string from;

        detail::scanDimensionedConstants(
            [&](const std::string& path, const FoamDict& si)
            {
                const FoamDict* pc = si.subDict("physicoChemical");
                if (!pc) return;
                NA = pc->scalarOr("NA", NA);
                k  = pc->scalarOr("k", k);
                from = path;
            });

        const scalar rr = 1.0e3 * NA * k;
        // Only worth a line when it actually differs from what brae would have assumed.
        if (std::fabs(rr - foamRRdefault) > 1e-9 * foamRRdefault)
        {
            noticeApplied("universal gas constant",
                          "RR = " + std::to_string((double)rr) + " J/(kmol K) from " +
                          (from.empty() ? std::string("brae's built-in constants") : from) +
                          " (NA = " + std::to_string((double)NA) + ", k = " + std::to_string((double)k) +
                          "); brae's built-in OF v2412 value is 8314.47006650545. R = RR/molWeight, so "
                          "this scales rho and psi in every compressible case.");
        }
        return rr;
    }();
    return cached;
}

// Standard temperature [K]. OF's hConstThermo takes it as the DEFAULT for `Tref`, the point about which
// the sensible enthalpy is measured: Hs = Cp*(T - Tref) + Href. It is not a cosmetic offset -- see
// hConstTToHe in thermo_model.cuh for why an offset in he cannot be cancelled out of the energy equation.
inline scalar foamTstd()
{
    static const scalar cached = []() -> scalar
    {
        scalar Tstd = foamTstdDefault;
        std::string from;

        detail::scanDimensionedConstants(
            [&](const std::string& path, const FoamDict& si)
            {
                const FoamDict* st = si.subDict("standard");
                if (!st) return;
                const scalar v = st->scalarOr("Tstd", Tstd);
                if (v != Tstd) { Tstd = v; from = path; }
            });

        if (!from.empty())
        {
            noticeApplied("standard temperature",
                          "Tstd = " + std::to_string((double)Tstd) + " K from " + from +
                          " (brae's built-in OF v2412 value is 298.15). It is the default reference "
                          "temperature of the sensible enthalpy, so it shifts he in every compressible case.");
        }
        return Tstd;
    }();
    return cached;
}

}   // namespace brae
