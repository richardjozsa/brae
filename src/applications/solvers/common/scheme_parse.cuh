#pragma once
// Shared fvSchemes div/laplacian/grad scheme parse -> DeviceSimpleControls flags. Used by BOTH the steady
// (gpuSimpleFoam) and transient (gpuPimpleFoam) drivers so the scheme detection lives in ONE place. Reads
// system/fvSchemes ($-expanded): div(phi,U) bounded/linearUpwind[V]/LUST + div(phi,{k,epsilon,omega,nuTilda})
// limitedLinear/linearUpwind, laplacian/snGrad corrected/limited (non-orth), grad(U) cellLimited. Extracted
// verbatim from gpuSimpleFoam; throws (OF-style) on a missing/unsupported div(phi,U) scheme.
#include "solver_controls.cuh"
#include "foam_dict.cuh"    // readFileExpanded
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <string>

namespace brae {

// Fill ctl's convection/laplacian/grad scheme flags from caseDir/system/fvSchemes.
inline void parseFvSchemesControls(const std::string& caseDir, DeviceSimpleControls& ctl)
{
            const std::string schemesText = readFileExpanded(caseDir + "/system/fvSchemes");   // $var expanded ($turbulence)
            std::istringstream fsch(schemesText);
            std::string ln;
            bool foundDivU = false;   // require an EXPLICIT div(phi,U); brae does not resolve the divSchemes 'default'
            bool warnedLeastSq = false, warnedCellMD = false;   // #14: warn-once on grad schemes brae approximates
            auto hasWord = [](const std::string& s, const std::string& w)   // whole-word match (so "uncorrected" != "corrected")
            {
                for (std::size_t p = s.find(w); p != std::string::npos; p = s.find(w, p + 1))
                {
                    const bool lb = (p == 0 || !std::isalpha((unsigned char)s[p-1]));
                    const bool rb = (p + w.size() >= s.size() || !std::isalpha((unsigned char)s[p+w.size()]));
                    if (lb && rb) return true;
                }
                return false;
            };
            // "limitedLinear <k_>" on a div line -> twoByk = 2/max(k_,SMALL); returns 0 if the scheme is absent.
            auto limitedTwoByk = [](const std::string& s) -> scalar
            {
                const std::size_t p = s.find("limitedLinear");
                if (p == std::string::npos) return 0.0;
                scalar kc = 1.0;
                std::sscanf(s.c_str() + p + 13, "%lf", &kc);   // coefficient after "limitedLinear"
                return 2.0 / std::max(kc, (scalar)1e-30);
            };
            // the interpolation-scheme word after "Gauss" on a div(phi,*) line (OF: "[bounded] Gauss <scheme> [args]").
            auto divSchemeWord = [](const std::string& s) -> std::string
            {
                const std::size_t g = s.find("Gauss");
                if (g == std::string::npos) return std::string();
                const char* p = s.c_str() + g + 5;
                while (*p && std::isspace((unsigned char)*p)) ++p;
                std::string w;
                while (*p && !std::isspace((unsigned char)*p) && *p != ';') { w += *p; ++p; }
                return w;
            };
            // OF-faithful fail-fast (mirrors surfaceInterpolationScheme::New "Unknown discretisation scheme ... Valid
            // schemes are : (...)" + exit(FatalIOError)): throw on a scheme cf models nothing close to; warn loudly on
            // one cf only APPROXIMATES (so the route is detected, never silently covered). `ok` = exact; `approx` = warned.
            auto checkDiv = [&](
                const std::string& s,
                const char* field,
                std::initializer_list<const char*> ok,
                std::initializer_list<const char*> approx)
            {
                const std::string w = divSchemeWord(s);
                for (const char* o : ok)
                    if (w == o) return;
                for (const char* o : approx)
                    if (w == o)
                    {
                        std::fprintf(stderr, "brae WARNING: div(phi,%s) 'Gauss %s' has no exact cf kernel -- run as a near-equivalent "
                                     "(NOT OF-bit-identical). Set the scheme to an exact one to avoid this.\n", field, w.c_str());
                        return;
                    }
                std::string valid;
                for (const char* o : ok)     (valid += " ") += o;
                for (const char* o : approx) (valid += " ") += o;
                throw std::runtime_error(std::string("brae: unknown/unsupported div(phi,") + field +
                    ") scheme 'Gauss " + w + "'; cf supports : (" + valid + " )");
            };
            while (std::getline(fsch, ln))
            {
                if (ln.find("div(phi,U)") != std::string::npos)
                {
                    foundDivU = true;
                    checkDiv(ln, "U", {"upwind", "linearUpwind", "linearUpwindV", "LUST"}, {"limitedLinear", "limitedLinearV"});
                    if (ln.find("bounded") != std::string::npos)      ctl.bounded = true;
                    if (ln.find("linearUpwind") != std::string::npos) ctl.linearUpwind = true;   // linearUpwindV contains this -> upwind matrix + gradients
                    if (ln.find("linearUpwindV") != std::string::npos) ctl.linearUpwindV = true; // + OF vector direction limiter
                    if (ln.find("LUST") != std::string::npos)         ctl.lust = true;   // 0.75 linear + 0.25 linearUpwind
                }
                // grad(U) cellLimited Gauss linear <k> (OF cellLimitedGrad<minmod>): k is the first number after
                // "cellLimited" (the basicScheme between has no digits). 0 = unlimited. cellMDLimited not yet handled.
                if (ln.find("grad(U)") != std::string::npos && hasWord(ln, "cellLimited"))
                {
                    ctl.gradULimitK = 1.0;
                    const char* s = ln.c_str() + ln.find("cellLimited") + 11;
                    while (*s && !(std::isdigit((unsigned char)*s) || *s == '.')) ++s;
                    scalar kc;
                    if (std::sscanf(s, "%lf", &kc) == 1) ctl.gradULimitK = kc;
                }
                // #14: brae computes gradients via Gauss linear only. These are valid ALTERNATIVE discretisations
                // (not wrong answers), so warn-once rather than fail -- the user should know brae is approximating.
                if (!warnedLeastSq && ln.find("leastSquares") != std::string::npos)
                { warnedLeastSq = true; std::fprintf(stderr, "brae WARNING: gradScheme 'leastSquares' is approximated as Gauss linear (differs on skewed meshes)\n"); }
                if (!warnedCellMD && ln.find("cellMDLimited") != std::string::npos)
                { warnedCellMD = true; std::fprintf(stderr, "brae WARNING: grad limiter 'cellMDLimited' is not applied (runs unlimited)\n"); }
                if (std::getenv("BRAE_SCHEME_DEBUG") && ln.find("div(phi,") != std::string::npos) std::fprintf(stderr, "[scheme] %s\n", ln.c_str());
                if (ln.find("div(phi,k)") != std::string::npos)
                {
                    checkDiv(ln, "k", {"upwind", "linearUpwind", "limitedLinear"}, {});
                    const scalar t = limitedTwoByk(ln);
                    if (t > 0.0) { ctl.limitedK = true; ctl.twoBykK = t; }
                    if (ln.find("linearUpwind") != std::string::npos) ctl.luK = true;
                }
                if (ln.find("div(phi,epsilon)") != std::string::npos || ln.find("div(phi,omega)") != std::string::npos)
                {
                    checkDiv(ln, "epsilon|omega", {"upwind", "linearUpwind", "limitedLinear"}, {});
                    const scalar t = limitedTwoByk(ln);
                    if (t > 0.0) { ctl.limitedEps = true; ctl.twoBykEps = t; }   // 2nd turb scalar (eps|omega)
                    if (ln.find("linearUpwind") != std::string::npos) ctl.luEps = true;
                }
                // Energy: OF names the field "h" for sensibleEnthalpy and "e" for sensibleInternalEnergy,
                // and the kinetic term "K" or "Ekp" to match. Any of them sets the same flags.
                if (ln.find("div(phi,h)") != std::string::npos || ln.find("div(phi,e)") != std::string::npos
                 || ln.find("div(phi,K)") != std::string::npos || ln.find("div(phi,Ekp)") != std::string::npos)
                {
                    checkDiv(ln, "h|e|K|Ekp", {"upwind", "linearUpwind", "limitedLinear"}, {});
                    const scalar t = limitedTwoByk(ln);
                    if (t > 0.0) { ctl.limitedHe = true; ctl.twoBykHe = t; }
                    if (ln.find("linearUpwind") != std::string::npos) ctl.luHe = true;
                }
                if (ln.find("div(phi,nuTilda)") != std::string::npos)   // SA: nuTilda uses the k-slot scheme flags
                {
                    checkDiv(ln, "nuTilda", {"upwind", "linearUpwind", "limitedLinear"}, {});
                    const scalar t = limitedTwoByk(ln);
                    if (t > 0.0) { ctl.limitedK = true; ctl.twoBykK = t; }
                    if (ln.find("linearUpwind") != std::string::npos) ctl.luK = true;
                }
                if (ln.find("Gauss") != std::string::npos || ln.find("snGrad") != std::string::npos ||
                    ln.find("laplacian") != std::string::npos || ln.find("default") != std::string::npos)
                {
                    if (hasWord(ln, "corrected")) ctl.nonOrth = true;     // unlimited non-orth correction (psi = 1)
                    // OF fv::limitedSnGrad "limited [<correctedScheme>] <psi>" (psi in [0,1]): non-orth correction
                    // capped per-face. hasWord avoids matching "unlimited" and "limitedLinear" (a div scheme); the coeff
                    // is the next numeric token after "limited" (skip an optional scheme word like "corrected").
                    if (hasWord(ln, "limited"))
                    {
                        ctl.nonOrth = true;
                        scalar psi = 1.0;
                        const char* s = ln.c_str() + ln.find("limited") + 7;
                        while (*s && !(std::isdigit((unsigned char)*s) || *s == '.')) ++s;   // skip to the coefficient
                        if (std::sscanf(s, "%lf", &psi) == 1) ctl.nonOrthLimit = psi;
                    }
                }
            }
            if (!schemesText.empty() && !foundDivU)
                throw std::runtime_error("fvSchemes: no explicit div(phi,U) scheme. brae does not resolve the"
                    " divSchemes 'default' for momentum convection (it would silently run first-order upwind)."
                    " Add e.g. 'div(phi,U)  bounded Gauss linearUpwind grad(U);'.");
}

}  // namespace brae
