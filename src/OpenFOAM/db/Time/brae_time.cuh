#pragma once
// brae::Time -- the OpenFOAM Time/functionObjectList lifecycle, owned centrally instead of per solver.
//
// WHY THIS EXISTS. In OpenFOAM, functionObjects are a member of Time (Time.C:469), and Time drives the
// whole lifecycle itself:
//
//     read()             TimeIO.C:443,484   on construction, and on every controlDict re-read
//     start()            Time.C:820         first time step
//     execute()          Time.C:797,825     every time step
//     end()              Time.C:801         final time step
//     adjustTimeStep()   Time.C:142         transient only
//
// rhoSimpleFoam.C contains NO functionObject code -- only `#include "postProcess.H"` and
// `runTime.write()`. A solver gets functionObjects by having a Time; it cannot forget them and it
// cannot implement them differently from another solver.
//
// brae had no such class. Each solver parsed `functions{}` itself, and in practice only ever looked for
// one hardcoded type (gpuSimpleFoam.cu:744 searches for `forceCoeffs` and nothing else). rhoSimpleFoam
// never parsed it at all, so on a case like gasMixing/injectorPipe -- which ships scalarTransport,
// sampling and abort -- every functionObject was silently skipped. Not refused, not warned: skipped.
// That is the bug class this file removes.
//
// WHAT THIS DELIBERATELY DOES NOT OWN. brae captures CUDA graphs INSIDE the linear solver
// (device_amg_vcycle.cu:315,332 and device_amg_pcg.cu:182), several layers below a time step, so the
// outer loop is free to be centralised without disturbing capture. The line drawn here is OF's:
//
//     PCG / AMG V-cycle          solver internals, graph-captured   -- NOT here
//     SIMPLE / PIMPLE iterations each solver, different shapes       -- NOT here
//     time advance, functionObjects, write cadence                   -- here
//
// Everything in this file is host code executing ONCE PER TIME STEP, against device work of
// milliseconds to seconds per iteration. It must never touch per-iteration device state: the moment it
// does, it becomes a synchronisation point in a loop that is deliberately asynchronous.

#include "cf_types.cuh"
#include "foam_dict.cuh"
#include "brae_notice.cuh"   // noticeIgnored: the established "never drop an input silently" channel
#include <memory>
#include <string>
#include <vector>

namespace brae {

// One functionObject. Mirrors OF's functionObject virtual interface: a solver never calls these, Time
// does. Returning false from execute()/write() reports failure without aborting the run, as in OF.
class FunctionObject
{
public:
    virtual ~FunctionObject() = default;

    // OF functionObject::name() -- the sub-dictionary key, used in logs and error messages.
    virtual const std::string& name() const = 0;

    // OF functionObject::execute(): called every time step, after the solver has advanced the fields.
    virtual bool execute() = 0;

    // OF functionObject::write(): called on write times only. Split from execute() because OF splits
    // them -- scalarTransport solves every step but writes on writeControl.
    virtual bool write() { return true; }

    // OF functionObject::end(): called once at the final time step.
    virtual bool end() { return true; }
};

// OF functionObjectList. Constructed from controlDict's `functions` entry, owned by Time.
//
// Unknown types are REPORTED, never skipped in silence -- that silence is exactly what let gasMixing's
// scalarTransport disappear. A type brae cannot build is named on stderr so the run states plainly
// which part of the case it is not honouring.
class FunctionObjectList
{
public:
    // OF functionObjectList::read(). Walks controlDict.functions and builds what it recognises.
    //
    // Anything it cannot build goes through noticeIgnored -- the SAME convention the rest of brae uses
    // for "never drop an input silently". Reporting here rather than in each solver is the entire point
    // of centralising: a solver that adopts Time cannot forget to report, in the same way an OF solver
    // cannot forget to run functionObjects. Returns the number not built.
    // `handledOutside` names types the CALLING SOLVER computes on its own, outside this lifecycle.
    // They are reported as APPROXIMATED rather than ignored, because that is what they are: brae's
    // forceCoeffs, for example, is a single post-run print, whereas OF's forceCoeffs is a per-time-step
    // functionObject (execute()/write(), forceCoeffs.H:547,550) that writes a coefficient HISTORY under
    // postProcessing/. Reporting those as "ignored" would be false; reporting nothing would hide a real
    // difference from OF. Neither is acceptable, so they get their own category.
    // The two "outside" lists exist because the SAME type can be faithful in one solver and an
    // approximation in another, and saying so accurately matters more than saying it uniformly:
    //
    //   pimpleFoam  forceCoeffs -> samples on the write cadence and writes
    //                              postProcessing/forceCoeffs/<time>/coefficient.dat, i.e. what OF does
    //   simpleFoam  forceCoeffs -> a single post-run print; OF's runs every step and writes a history
    //
    // Calling both "approximated" would understate pimpleFoam; calling both "applied" would overstate
    // simpleFoam. Each solver declares which it is.
    int read(const FoamDict* functions,
             const std::vector<std::string>& approximatedOutside = {},
             const std::vector<std::string>& appliedOutside = {})
    {
        objects_.clear();
        if (!functions) return 0;
        int missed = 0;
        auto listed = [](const std::vector<std::string>& v, const std::string& t)
        {
            for (const std::string& h : v) if (h == t) return true;
            return false;
        };
        for (const auto& s : functions->subs)
        {
            const std::string type = s.second.wordOr("type", "");
            std::unique_ptr<FunctionObject> fo = create(s.first, type, s.second);
            if (fo)
            {
                objects_.push_back(std::move(fo));
                continue;
            }
            if (listed(appliedOutside, type))
            {
                noticeApplied(
                    "functions/" + s.first,
                    "type '" + type + "' is implemented by the solver outside the functionObject "
                    "lifecycle, on OpenFOAM's own write cadence.");
                continue;
            }
            if (listed(approximatedOutside, type))
            {
                noticeApproximated(
                    "functions/" + s.first,
                    "type '" + type + "' is computed by the solver outside the functionObject "
                    "lifecycle, so it is reported once rather than per time step as OpenFOAM does.");
                continue;
            }
            ++missed;
            noticeIgnored(
                "functions/" + s.first,
                "type '" + type + "' is not implemented, so this functionObject is NOT run. "
                "OpenFOAM would execute it every time step.");
        }
        return missed;
    }

    bool start()   { return all(&FunctionObject::execute); }   // OF calls execute() on the first step
    bool execute() { return all(&FunctionObject::execute); }
    bool write()   { return all(&FunctionObject::write); }
    bool end()     { return all(&FunctionObject::end); }

    std::size_t size() const { return objects_.size(); }

private:
    // The registry. Empty for now BY DESIGN: this file is step 1 and adopts no solver, so it ships with
    // no concrete types. scalarTransport and forceCoeffs land here once the three solvers are migrated,
    // so they are written once rather than three times -- which is the duplication this class removes.
    std::unique_ptr<FunctionObject> create(
        const std::string& /*name*/, const std::string& /*type*/, const FoamDict& /*dict*/)
    {
        return nullptr;
    }

    bool all(bool (FunctionObject::*fn)())
    {
        bool ok = true;
        for (auto& o : objects_) ok = ((*o).*fn)() && ok;   // run every one, do not short-circuit
        return ok;
    }

    std::vector<std::unique_ptr<FunctionObject>> objects_;
};

}   // namespace brae
