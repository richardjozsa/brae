// fvSchemes entries must be resolved WITHIN their own sub-dictionary (group C: C1-C4).
//
// The parser used to sniff keywords across a flat token stream: any statement containing "Gauss" was a
// candidate laplacian/snGrad entry, and the four energy-related div keys all wrote the same flags. Both
// are silent scheme substitutions -- the run converges to a plausible answer computed with a
// discretisation the case never asked for.
//
// The trigger is not exotic. aerofoilNACA0012, a stock OF v2412 tutorial, uses the standard macro idiom:
//
//     gradSchemes { limited  cellLimited Gauss linear 1;  grad(U) $limited;  grad(k) $limited; ... }
//     divSchemes  { div(phi,U)  bounded Gauss linearUpwind limited;  ... }
//
// where "limited" NAMES a gradient scheme. Both statements matched hasWord(ln, "limited") and set
// ctl.nonOrth, so a case whose laplacianSchemes AND snGradSchemes both say `orthogonal` ran WITH
// non-orthogonal correction. Measured on a laplacian-orthogonal case: nonOrth 0 -> 1.
//
// Each case below is paired with its negative control, because "the flag is off" is only evidence if the
// same parser turns it ON for the input that genuinely asks for it.

#include "scheme_parse.cuh"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>

using namespace brae;

namespace {

int failures = 0;

void check(const char* what, bool ok, const std::string& got, const std::string& want)
{
    if (ok) return;
    std::printf("  FAIL %s: got %s, want %s\n", what, got.c_str(), want.c_str());
    failures++;
}

void checkFlag(const char* what, bool got, bool want)
{
    check(what, got == want, got ? "true" : "false", want ? "true" : "false");
}

void checkNum(const char* what, scalar got, scalar want)
{
    check(what, std::fabs(got - want) < 1e-12, std::to_string((double)got), std::to_string((double)want));
}

// Write an fvSchemes into a throwaway case and parse it.
DeviceSimpleControls parse(const std::string& dir, const std::string& body)
{
    std::filesystem::create_directories(dir + "/system");
    std::ofstream(dir + "/system/fvSchemes") << body;
    DeviceSimpleControls ctl;
    parseFvSchemesControls(dir, ctl);
    return ctl;
}

const char* LAP_ORTHOGONAL =
    "laplacianSchemes { default Gauss linear orthogonal; }\n"
    "snGradSchemes    { default orthogonal; }\n"
    "interpolationSchemes { default linear; }\n";

const char* LAP_CORRECTED =
    "laplacianSchemes { default Gauss linear corrected; }\n"
    "snGradSchemes    { default corrected; }\n"
    "interpolationSchemes { default linear; }\n";

}   // namespace

int main()
{
    const std::string tmp = "/tmp/brae_scheme_blocks";
    std::filesystem::remove_all(tmp);

    // ---- C3: a gradScheme NAMED "limited", referenced from a div line, must not touch the laplacian ----
    {
        const std::string body =
            std::string("ddtSchemes { default steadyState; }\n")
            + "gradSchemes { default Gauss linear;\n"
              "              limited cellLimited Gauss linear 1;\n"
              "              grad(U) cellLimited Gauss linear 1; }\n"
              "divSchemes  { default none;\n"
              "              div(phi,U) bounded Gauss linearUpwind limited; }\n"
            + LAP_ORTHOGONAL;
        const DeviceSimpleControls c = parse(tmp + "/c3_orthogonal", body);
        checkFlag("C3 nonOrth off when laplacian+snGrad are orthogonal", c.nonOrth, false);
        // Negative control: the SAME div/grad lines with a corrected laplacian must still set it, so the
        // fix cannot be "nonOrth is never set from anywhere".
        const DeviceSimpleControls d = parse(tmp + "/c3_corrected",
            body.substr(0, body.size() - std::string(LAP_ORTHOGONAL).size()) + LAP_CORRECTED);
        checkFlag("C3 nonOrth ON when the laplacian IS corrected", d.nonOrth, true);
        // And the grad limiter must still be read from the gradSchemes line it belongs to.
        checkNum("C3 grad(U) cellLimited still parsed", c.gradULimitK, 1.0);
    }

    // ---- C3b: `limited 0.33` inside laplacianSchemes is the REAL limited-snGrad and must be honoured ----
    {
        const std::string body =
            std::string("gradSchemes { default Gauss linear; }\n")
            + "divSchemes  { default none; div(phi,U) bounded Gauss upwind; }\n"
              "laplacianSchemes { default Gauss linear limited corrected 0.33; }\n"
              "snGradSchemes    { default limited corrected 0.33; }\n"
              "interpolationSchemes { default linear; }\n";
        const DeviceSimpleControls c = parse(tmp + "/c3b", body);
        checkFlag("C3b limited snGrad sets nonOrth", c.nonOrth, true);
        checkNum("C3b limited coefficient", c.nonOrthLimit, 0.33);
    }

    // ---- C4: div(phi,e) and div(phi,K) are separate entries and must not share a slot ----
    {
        const std::string body =
            std::string("gradSchemes { default Gauss linear; }\n")
            + "divSchemes { default none;\n"
              "             div(phi,U)   bounded Gauss upwind;\n"
              "             div(phi,e)   bounded Gauss linearUpwind limited;\n"
              "             div(phi,K)   bounded Gauss upwind;\n"
              "             div(phi,Ekp) bounded Gauss upwind; }\n"
            + LAP_ORTHOGONAL;
        const DeviceSimpleControls c = parse(tmp + "/c4_split", body);
        checkFlag("C4 energy honours linearUpwind", c.luHe, true);
        checkFlag("C4 kinetic term stays upwind", c.luKin, false);
        checkFlag("C4 kinetic entry was seen", c.foundKinScheme, true);
        // Negative control: the reverse assignment must flip both, so the test cannot pass on a parser
        // that simply hardcodes luKin = false.
        const std::string swapped =
            std::string("gradSchemes { default Gauss linear; }\n")
            + "divSchemes { default none;\n"
              "             div(phi,U)   bounded Gauss upwind;\n"
              "             div(phi,e)   bounded Gauss upwind;\n"
              "             div(phi,K)   bounded Gauss linearUpwind limited;\n"
              "             div(phi,Ekp) bounded Gauss linearUpwind limited; }\n"
            + LAP_ORTHOGONAL;
        const DeviceSimpleControls d = parse(tmp + "/c4_swapped", swapped);
        checkFlag("C4 swapped: energy upwind", d.luHe, false);
        checkFlag("C4 swapped: kinetic linearUpwind", d.luKin, true);
        // No explicit K/Ekp entry -> inherit the energy scheme, and SAY so via foundKinScheme.
        const std::string none =
            std::string("gradSchemes { default Gauss linear; }\n")
            + "divSchemes { default none;\n"
              "             div(phi,U) bounded Gauss upwind;\n"
              "             div(phi,e) bounded Gauss linearUpwind limited; }\n"
            + LAP_ORTHOGONAL;
        const DeviceSimpleControls e = parse(tmp + "/c4_none", none);
        checkFlag("C4 absent kinetic entry inherits energy", e.luKin, true);
        checkFlag("C4 absent kinetic entry is flagged", e.foundKinScheme, false);
    }

    // ---- C2: cellLimited on the TURBULENCE and ENERGY gradients, not just grad(U) ----
    {
        const std::string body =
            std::string("gradSchemes { default Gauss linear;\n")
            + "              grad(U)     cellLimited Gauss linear 1;\n"
              "              grad(k)     cellLimited Gauss linear 0.5;\n"
              "              grad(omega) cellLimited Gauss linear 0.5; }\n"
              "divSchemes { default none; div(phi,U) bounded Gauss upwind; }\n"
            + LAP_ORTHOGONAL;
        const DeviceSimpleControls c = parse(tmp + "/c2", body);
        checkNum("C2 grad(U) limiter", c.gradULimitK, 1.0);
        checkNum("C2 grad(k)/grad(omega) limiter", c.gradKLimitK, 0.5);
        checkNum("C2 grad(h|e) unlisted stays unlimited", c.gradHeLimitK, 0.0);
        // Negative control: with no cellLimited anywhere, every limiter must be 0.
        const std::string plain =
            std::string("gradSchemes { default Gauss linear; }\n")
            + "divSchemes { default none; div(phi,U) bounded Gauss upwind; }\n"
            + LAP_ORTHOGONAL;
        const DeviceSimpleControls d = parse(tmp + "/c2_plain", plain);
        checkNum("C2 negative control grad(U)", d.gradULimitK, 0.0);
        checkNum("C2 negative control grad(k)", d.gradKLimitK, 0.0);
    }

    // ---- C1: interpolationSchemes was never parsed; a non-linear default must be refused ----
    {
        const std::string body =
            std::string("gradSchemes { default Gauss linear; }\n")
            + "divSchemes { default none; div(phi,U) bounded Gauss upwind; }\n"
              "laplacianSchemes { default Gauss linear orthogonal; }\n"
              "snGradSchemes    { default orthogonal; }\n"
              "interpolationSchemes { default midPoint; }\n";
        bool threw = false;
        try { parse(tmp + "/c1", body); }
        catch (const std::exception&) { threw = true; }
        checkFlag("C1 non-linear interpolation default is refused", threw, true);
        // Negative control: `linear` (OF's default, and what brae actually does) must still be accepted.
        bool threwLinear = false;
        try
        {
            parse(tmp + "/c1_linear",
                  std::string("gradSchemes { default Gauss linear; }\n")
                  + "divSchemes { default none; div(phi,U) bounded Gauss upwind; }\n"
                  + LAP_ORTHOGONAL);
        }
        catch (const std::exception&) { threwLinear = true; }
        checkFlag("C1 linear interpolation is accepted", threwLinear, false);
    }

    std::printf("scheme_blocks: %d failures\n", failures);
    return failures ? 1 : 0;
}
