#pragma once
// residual_control.cuh -- SIMPLE residualControl convergence test, for every driver.
//
// OF's rule is in simpleControl::criteriaSatisfied() (simpleControl.C:49-):
//
//     if (residualControl_.empty()) { return false; }          // an empty dict NEVER converges
//     bool achieved = true;
//     bool checked  = false;    // "safety that some checks were indeed performed"
//     ...
//     return achieved && checked;
//
// brae had `converged = hasRC && ok(p) && ok(U)` where ok() returns TRUE for a field the dict does not
// list (target -1). So `residualControl { }` -- or a dict listing only fields brae does not check --
// made every test vacuously pass and the run stopped after ONE iteration reporting convergence. That is
// worse than a hang: it writes a plausible-looking field set and prints "converged".
//
// Both halves of OF's rule are needed. Dropping the empty check alone still lets `residualControl
// { foo 1e-5; }` converge instantly, which is exactly what `checked` exists to stop.

#include "cf_types.cuh"
#include <regex>
#include "foam_dict.cuh"
#include <string>

namespace brae {

class ResidualControl
{
public:
    explicit ResidualControl(const FoamDict* dict) : dict_(dict) {}

    bool active() const { return dict_ != nullptr; }

    // Target for a field, or -1 when the dict does not list it (OF: unlisted -> not a criterion).
    //
    // OF stores each residualControl key as a wordRe -- a word OR A REGEX (solutionControl::read:
    // `const word& fName = dEntry.keyword(); fd.name = fName.c_str();`, matched against the solved field
    // name). So `"(k|epsilon)" 1e-3;` is ONE entry covering two fields, and the stock tutorials use it:
    // angledDuct writes `"(k|epsilon)"`, aerofoilNACA0012 writes `"(k|omega|e)"`.
    //
    // An exact-name lookup finds neither, so those criteria were silently dropped and the run was gated
    // on p alone -- dict_audit reported residualControl/U, /e and /(k|epsilon) as never read. A case is
    // then declared converged on a subset of OF's criteria, which is the wrong answer in whichever
    // direction the missing field happens to lie.
    //
    // Exact first, then regex with LAST match winning -- the same rule findPatchEntry applies to patch
    // entries, so a regex-keyed case cannot resolve differently depending on which reader sees it.
    scalar target(const std::string& field) const
    {
        if (!dict_) return scalar(-1);
        if (const auto* t = dict_->find(field)) return dict_->scalarOr(field, scalar(-1));
        scalar hit = scalar(-1);
        for (const auto& lv : dict_->leaves)
        {
            const std::string& key = lv.first;
            if (key.find_first_of("()|*?[].^$") == std::string::npos) continue;   // a plain word, already tried
            try
            {
                if (std::regex_match(field, compileFoamRegex(key))) hit = dict_->scalarOr(key, scalar(-1));
            }
            catch (const std::regex_error&) { /* not a usable regex -> not a match, as OF treats it */ }
        }
        return hit;
    }

    // Test one field. An unlisted field is not a criterion and does NOT count as a performed check.
    bool ok(scalar residual, scalar tgt)
    {
        if (tgt < scalar(0)) return true;
        ++checked_;
        return residual < tgt;
    }

    bool ok(scalar residual, const std::string& field) { return ok(residual, target(field)); }

    // OF's `achieved && checked`, plus the empty-dict short circuit.
    bool converged(bool achieved) const { return active() && achieved && checked_ > 0; }

    // Per-iteration reset: `checked` counts one sweep, not the whole run.
    void beginIteration() { checked_ = 0; }

    int checked() const { return checked_; }

private:
    const FoamDict* dict_ = nullptr;
    int checked_ = 0;
};

}   // namespace brae
