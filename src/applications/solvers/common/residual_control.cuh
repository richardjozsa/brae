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
#include "foam_dict.cuh"
#include <string>

namespace brae {

class ResidualControl
{
public:
    explicit ResidualControl(const FoamDict* dict) : dict_(dict) {}

    bool active() const { return dict_ != nullptr; }

    // Target for a field, or -1 when the dict does not list it (OF: unlisted -> not a criterion).
    scalar target(const std::string& field) const
    {
        return dict_ ? dict_->scalarOr(field, scalar(-1)) : scalar(-1);
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
