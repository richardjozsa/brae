#pragma once
// Shared fvSchemes div/laplacian/grad scheme parse -> DeviceSimpleControls flags. Used by BOTH the steady
// (gpuSimpleFoam) and transient (gpuPimpleFoam) drivers so the scheme detection lives in ONE place. Reads
// system/fvSchemes ($-expanded): div(phi,U) bounded/linearUpwind[V]/LUST + div(phi,{k,epsilon,omega,nuTilda})
// limitedLinear/linearUpwind, laplacian/snGrad corrected/limited (non-orth), grad(U) cellLimited. Extracted
// verbatim from gpuSimpleFoam; throws (OF-style) on a missing/unsupported div(phi,U) scheme.
#include "solver_controls.cuh"
#include "foam_dict.cuh"    // readFileExpanded
#include "brae_notice.cuh"
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <map>
#include <sstream>
#include <utility>
#include <vector>
#include <stdexcept>
#include <string>

namespace brae {

// Fill ctl's convection/laplacian/grad scheme flags from caseDir/system/fvSchemes.
inline void parseFvSchemesControls(const std::string& caseDir, DeviceSimpleControls& ctl)
{
            std::string schemesText = readFileExpanded(caseDir + "/system/fvSchemes");   // $var expanded ($turbulence)
            // Statements are ';'-terminated and each one belongs to a SUB-DICTIONARY. Both halves matter.
            //
            // Splitting on ';' (not newlines) is what stops one entry's flags leaking into the next: OF does
            // not care about layout, so "div(phi,U) upwind; div(phi,e) linearUpwind;" on one line is two
            // different schemes, and matching per LINE once gave div(phi,U) the energy equation's scheme.
            //
            // Tracking the enclosing sub-dictionary is what stops a rule firing on a statement from a
            // DIFFERENT group. The rules used to sniff keywords across a flat token stream, so any statement
            // containing "Gauss" was treated as a candidate laplacian/snGrad entry. aerofoilNACA0012 uses OF's
            // standard macro idiom:
            //     gradSchemes { limited  cellLimited Gauss linear 1;  grad(U) $limited; ... }
            //     divSchemes  { div(phi,U)  bounded Gauss linearUpwind limited; ... }
            // where "limited" is the NAME of a gradient scheme and has nothing to do with the laplacian. Both
            // statements matched `hasWord(ln, "limited")` and set ctl.nonOrth, so a case whose laplacianSchemes
            // and snGradSchemes both say `orthogonal` ran WITH non-orthogonal correction. Measured on a
            // laplacian-orthogonal case: nonOrth 0 -> 1 purely from the divSchemes line.
            struct Stmt { std::string block, text; };
            std::vector<Stmt> stmts;
            {
                std::string buf, cur;
                std::vector<std::string> stack;
                auto lastWord = [](const std::string& b)
                {
                    std::size_t e = b.size();
                    while (e > 0 && std::isspace((unsigned char)b[e-1])) --e;
                    std::size_t st = e;
                    while (st > 0 && !std::isspace((unsigned char)b[st-1]) && b[st-1] != '{' && b[st-1] != '}') --st;
                    return b.substr(st, e - st);
                };
                for (char c : schemesText)
                {
                    if (c == '{')      { stack.push_back(lastWord(buf)); buf.clear(); }
                    else if (c == '}') { if (!stack.empty()) stack.pop_back(); buf.clear(); }
                    else if (c == ';') { stmts.push_back({stack.empty() ? std::string() : stack.front(), buf}); buf.clear(); }
                    else                 buf += c;
                }
            }
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
            // The cellLimited coefficient of every NAMED gradSchemes entry, keyed by its name.
            //
            // OF's linearUpwind takes the gradient scheme as an ARGUMENT and looks that name up in
            // gradSchemes (linearUpwind.H: gradSchemeName_(schemeData) -> mesh.gradScheme(name)), so
            //     gradSchemes { limited cellLimited Gauss linear 1; }
            //     divSchemes  { div(phi,e) bounded Gauss linearUpwind limited; }
            // gives the energy's deferred correction a CELL-LIMITED gradient even though the case has no
            // `grad(e)` entry at all. brae keyed the limiter off `grad(e)`/`grad(h)` alone, found nothing,
            // and ran unlimited. Measured on aerofoilNACA0012: forcing OF to use an unlimited gradient
            // reproduces brae's first-iteration T[233.71, 301.64] against OF's own T[297.95, 298.01].
            std::map<std::string, scalar> gradLimitByName;
            for (const Stmt& gs : stmts)
            {
                if (gs.block != "gradSchemes") continue;
                const char* p = gs.text.c_str();
                while (*p && std::isspace((unsigned char)*p)) ++p;
                std::string key;
                while (*p && !std::isspace((unsigned char)*p)) key += *p++;
                if (key.empty()) continue;
                scalar kc = 0.0;                                     // not cellLimited -> unlimited
                if (hasWord(gs.text, "cellLimited"))
                {
                    kc = 1.0;
                    const char* c = gs.text.c_str() + gs.text.find("cellLimited") + 11;
                    while (*c && !(std::isdigit((unsigned char)*c) || *c == '.')) ++c;
                    if (std::sscanf(c, "%lf", &kc) != 1) kc = 1.0;
                }
                gradLimitByName[key] = kc;
            }
            // The cellLimited coefficient linearUpwind on THIS div statement will use: the word after
            // "linearUpwind" resolved through the table above, else the gradSchemes `default`. Returns
            // -1 when the statement is not linearUpwind at all, so a caller can keep its own fallback.
            auto luGradLimit = [&](const std::string& s) -> scalar
            {
                const std::size_t p = s.find("linearUpwind");
                if (p == std::string::npos) return -1.0;
                const char* c = s.c_str() + p + 12;
                while (*c && (std::isalpha((unsigned char)*c)))  ++c;   // skip the V of linearUpwindV
                while (*c && std::isspace((unsigned char)*c))    ++c;
                std::string name;
                while (*c && !std::isspace((unsigned char)*c) && *c != ';') name += *c++;
                if (name.empty()) name = "default";
                const auto it = gradLimitByName.find(name);
                if (it != gradLimitByName.end()) return it->second;
                const auto d = gradLimitByName.find("default");
                return d != gradLimitByName.end() ? d->second : 0.0;
            };
            for (const Stmt& st : stmts)
            {
                const std::string& ln = st.text;
                const bool inDiv    = (st.block == "divSchemes");
                const bool inGrad   = (st.block == "gradSchemes");
                const bool inLap    = (st.block == "laplacianSchemes" || st.block == "snGradSchemes");
                const bool inInterp = (st.block == "interpolationSchemes");
                (void)inInterp;
                if (inDiv && ln.find("div(phi,U)") != std::string::npos)
                {
                    foundDivU = true;
                    checkDiv(ln, "U", {"upwind", "linearUpwind", "linearUpwindV", "LUST"}, {"limitedLinear", "limitedLinearV"});
                    if (ln.find("bounded") != std::string::npos)      ctl.bounded = true;
                    if (ln.find("linearUpwind") != std::string::npos) ctl.linearUpwind = true;   // linearUpwindV contains this -> upwind matrix + gradients
                    if (ln.find("linearUpwindV") != std::string::npos) ctl.linearUpwindV = true; // + OF vector direction limiter
                    if (ln.find("LUST") != std::string::npos)         ctl.lust = true;   // 0.75 linear + 0.25 linearUpwind
                    { const scalar g = luGradLimit(ln); if (g >= 0.0) ctl.gradULimitK = g; }   // EXPERIMENT
                }
                // grad(U) cellLimited Gauss linear <k> (OF cellLimitedGrad<minmod>): k is the first number after
                // "cellLimited" (the basicScheme between has no digits). 0 = unlimited. cellMDLimited not yet handled.
                if (inGrad && ln.find("grad(U)") != std::string::npos && hasWord(ln, "cellLimited"))
                {
                    ctl.gradULimitK = 1.0;
                    const char* s = ln.c_str() + ln.find("cellLimited") + 11;
                    while (*s && !(std::isdigit((unsigned char)*s) || *s == '.')) ++s;
                    scalar kc;
                    if (std::sscanf(s, "%lf", &kc) == 1) ctl.gradULimitK = kc;
                }
                // C2: the same cellLimited rule for the TURBULENCE and ENERGY gradients. Previously only
                // grad(U) was scanned, so `grad(k) cellLimited Gauss linear 1` was read and discarded.
                if (inGrad && hasWord(ln, "cellLimited"))
                {
                    scalar kc = 1.0;
                    const char* c = ln.c_str() + ln.find("cellLimited") + 11;
                    while (*c && !(std::isdigit((unsigned char)*c) || *c == '.')) ++c;
                    if (std::sscanf(c, "%lf", &kc) != 1) kc = 1.0;
                    if (ln.find("grad(k)") != std::string::npos || ln.find("grad(omega)") != std::string::npos
                     || ln.find("grad(epsilon)") != std::string::npos || ln.find("grad(nuTilda)") != std::string::npos)
                        ctl.gradKLimitK = kc;
                    if (ln.find("grad(h)") != std::string::npos || ln.find("grad(e)") != std::string::npos)
                        ctl.gradHeLimitK = kc;
                }
                // #14: brae computes gradients via Gauss linear only. These are valid ALTERNATIVE discretisations
                // (not wrong answers), so warn-once rather than fail -- the user should know brae is approximating.
                if (inGrad && !warnedLeastSq && ln.find("leastSquares") != std::string::npos)
                { warnedLeastSq = true; std::fprintf(stderr, "brae WARNING: gradScheme 'leastSquares' is approximated as Gauss linear (differs on skewed meshes)\n"); }
                if (inGrad && !warnedCellMD && ln.find("cellMDLimited") != std::string::npos)
                { warnedCellMD = true; std::fprintf(stderr, "brae WARNING: grad limiter 'cellMDLimited' is not applied (runs unlimited)\n"); }
                if (std::getenv("BRAE_SCHEME_DEBUG") && ln.find("div(phi,") != std::string::npos) std::fprintf(stderr, "[scheme] %s\n", ln.c_str());
                if (inDiv && ln.find("div(phi,k)") != std::string::npos)
                {
                    checkDiv(ln, "k", {"upwind", "linearUpwind", "limitedLinear"}, {});
                    const scalar t = limitedTwoByk(ln);
                    if (t > 0.0) { ctl.limitedK = true; ctl.twoBykK = t; }
                    if (ln.find("linearUpwind") != std::string::npos) ctl.luK = true;
                    if (hasWord(ln, "bounded")) ctl.boundedK = true;
                    { const scalar g = luGradLimit(ln); if (g >= 0.0) ctl.gradKLimitK = g; }   // EXPERIMENT
                }
                if (inDiv && (ln.find("div(phi,epsilon)") != std::string::npos || ln.find("div(phi,omega)") != std::string::npos))
                {
                    checkDiv(ln, "epsilon|omega", {"upwind", "linearUpwind", "limitedLinear"}, {});
                    const scalar t = limitedTwoByk(ln);
                    if (t > 0.0) { ctl.limitedEps = true; ctl.twoBykEps = t; }   // 2nd turb scalar (eps|omega)
                    if (ln.find("linearUpwind") != std::string::npos) ctl.luEps = true;
                    if (hasWord(ln, "bounded")) ctl.boundedEps = true;
                    { const scalar g = luGradLimit(ln); if (g >= 0.0) ctl.gradKLimitK = g; }   // EXPERIMENT
                }
                // Energy: OF names the field "h" for sensibleEnthalpy and "e" for sensibleInternalEnergy,
                // and the kinetic term "K" or "Ekp" to match. Any of them sets the same flags.
                // Energy: OF names the field "h" for sensibleEnthalpy and "e" for sensibleInternalEnergy.
                if (inDiv && (ln.find("div(phi,h)") != std::string::npos || ln.find("div(phi,e)") != std::string::npos))
                {
                    checkDiv(ln, "h|e", {"upwind", "linearUpwind", "limitedLinear"}, {});
                    const scalar t = limitedTwoByk(ln);
                    if (t > 0.0) { ctl.limitedHe = true; ctl.twoBykHe = t; }
                    if (ln.find("linearUpwind") != std::string::npos) ctl.luHe = true;
                    if (hasWord(ln, "bounded")) ctl.boundedHe = true;
                    const scalar g = luGradLimit(ln);   // the gradient linearUpwind NAMES, not grad(e)
                    if (g >= 0.0) ctl.gradHeLimitK = g;
                }
                // The KINETIC term, named "K" alongside h and "Ekp" alongside e. Its own fvSchemes entry and
                // its own slots -- see DeviceSimpleControls::luKin for why sharing the He slots was wrong.
                if (inDiv && (ln.find("div(phi,K)") != std::string::npos || ln.find("div(phi,Ekp)") != std::string::npos))
                {
                    checkDiv(ln, "K|Ekp", {"upwind", "linearUpwind", "limitedLinear"}, {});
                    ctl.foundKinScheme = true;
                    const scalar t = limitedTwoByk(ln);
                    if (t > 0.0) { ctl.limitedKin = true; ctl.twoBykKin = t; }
                    if (ln.find("linearUpwind") != std::string::npos) ctl.luKin = true;
                    if (hasWord(ln, "bounded")) ctl.boundedKin = true;
                    const scalar g = luGradLimit(ln);
                    if (g >= 0.0) ctl.gradKinLimitK = g;
                }
                if (inDiv && ln.find("div(phi,nuTilda)") != std::string::npos)   // SA: nuTilda uses the k-slot scheme flags
                {
                    checkDiv(ln, "nuTilda", {"upwind", "linearUpwind", "limitedLinear"}, {});
                    const scalar t = limitedTwoByk(ln);
                    if (t > 0.0) { ctl.limitedK = true; ctl.twoBykK = t; }
                    if (ln.find("linearUpwind") != std::string::npos) ctl.luK = true;
                    if (hasWord(ln, "bounded")) ctl.boundedK = true;   // SA: nuTilda uses the k slot
                }
                if (inLap)
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
            // No explicit div(phi,K|Ekp): OF would fall through to the divSchemes `default`. brae keeps its
            // previous behaviour (the energy scheme) rather than inventing one, but says so, because that is
            // an assumption and not what the file asked for.
            if (!ctl.foundKinScheme)
            {
                ctl.boundedKin = ctl.boundedHe;
                ctl.limitedKin = ctl.limitedHe;
                ctl.luKin      = ctl.luHe;
                ctl.twoBykKin  = ctl.twoBykHe;
                ctl.gradKinLimitK = ctl.gradHeLimitK;
            }
            // C1: interpolationSchemes was never parsed at all -- zero hits in src/. brae interpolates
            // linearly everywhere, which IS OF's default, so a `default linear` case was right by accident.
            // Anything else silently ran linear instead, which is a different discretisation, so it is
            // refused rather than noticed (the project rule: notice when brae does LESS than asked,
            // throw when the answer would be WRONG).
            for (const Stmt& st : stmts)
            {
                if (st.block != "interpolationSchemes") continue;
                const std::string& ln = st.text;
                std::istringstream is(ln);
                std::string key, scheme;
                is >> key >> scheme;
                if (key.empty() || scheme.empty() || scheme == "linear") continue;
                if (key == "default")
                    throw std::runtime_error(
                        "brae: interpolationSchemes default is '" + scheme + "'; brae interpolates linearly "
                        "and would silently run 'linear' instead. Set it to linear, or use a case that does.");
                noticeIgnored("interpolationSchemes",
                              key + " " + scheme + " -- brae interpolates linearly; only the `default` entry is enforced");
            }
            if (!schemesText.empty() && !foundDivU)
                throw std::runtime_error("fvSchemes: no explicit div(phi,U) scheme. brae does not resolve the"
                    " divSchemes 'default' for momentum convection (it would silently run first-order upwind)."
                    " Add e.g. 'div(phi,U)  bounded Gauss linearUpwind grad(U);'.");

            // What brae CONCLUDED, not what the file said. BRAE_SCHEME_DEBUG used to echo the raw input
            // line -- which is exactly the thing that was ambiguous: a one-line divSchemes looked correct
            // in that echo while the flags leaked between entries, and div(phi,U) silently picked up the
            // energy scheme. A measurement that never states which group its input landed in cannot tell
            // "hypothesis wrong" from "input misread", and I read one as the other.
            if (std::getenv("BRAE_SCHEME_DEBUG"))
            {
                auto sname = [](bool lu, bool lim) { return lu ? "linearUpwind" : (lim ? "limitedLinear" : "upwind"); };
                std::fprintf(stderr,
                    "[scheme] RESOLVED  div(phi,U)=%s%s  div(phi,k|omega)=%s  div(phi,h|e)=%s  "
                    "div(phi,K|Ekp)=%s%s  nonOrth=%d gradLimit(U=%g,k|omega=%g,h|e=%g)\n",
                    ctl.linearUpwind ? "linearUpwind" : "upwind", ctl.linearUpwindV ? "V" : "",
                    sname(ctl.luK, ctl.limitedK), sname(ctl.luHe, ctl.limitedHe),
                    sname(ctl.luKin, ctl.limitedKin), ctl.foundKinScheme ? "" : "(inherited)",
                    (int)ctl.nonOrth, ctl.gradULimitK, ctl.gradKLimitK, ctl.gradHeLimitK);
            }
}

}  // namespace brae
