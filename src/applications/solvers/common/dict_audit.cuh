#pragma once
// dict_audit.cuh -- report every dictionary entry brae READ OFF DISK AND THEN IGNORED.
//
// This exists because of what the rhoSimpleFoam audit kept finding. The dangerous class of defect in this
// project has never been "brae cannot parse X" -- that fails loudly and gets fixed. It is "brae parsed X,
// stored it, and then never used it", which converges to a plausible answer and says nothing. fvOptions
// (A2), the turbulent-inlet recompute (A3), the kinetic-term scheme (A4), the Function1 uniformValue (A5),
// tangentialVelocity (A9), interpolationSchemes (C1), grad(k) cellLimited (C2), the kinetic slot (C4),
// residualControl (C5), startFrom (C6) -- ten separate instances of one shape.
//
// Each was found by reading OF's source and noticing an entry brae never mentions. That does not scale and
// it does not generalise to the next solver. So: FoamDict records every key it is ASKED for, and this walks
// the parsed dict afterwards and names everything that was never asked. The complement is the gap list.
//
// It reports; it does not throw. An unread key is EVIDENCE, not proof -- some entries are legitimately for
// other tools (functionObject sub-entries, writeFormat), and a few are consumed by paths that ran before the
// audit point. Treating it as fatal would make it useless. Silence is the thing being fixed.
//
// On by default so it is visible during development. BRAE_DICT_AUDIT=0 silences it.
#include "foam_dict.cuh"
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <utility>
#include <vector>

namespace brae {

namespace detail {

// Entries that are genuinely not brae's to consume, so naming them would be noise that trains the reader
// to ignore the whole report. Kept deliberately short: when in doubt, REPORT it. A false positive costs a
// line of output; a false negative is the bug class this file exists to catch.
inline bool auditIgnorable(const std::string& key)
{
    static const char* skip[] = {
        "FoamFile", "version", "format", "class", "location", "object", "note", "arch",
        "writeFormat", "writePrecision", "writeCompression", "timeFormat", "timePrecision",
        "runTimeModifiable", "graphFormat", "libs", "functions", "DebugSwitches", "InfoSwitches",
        // Consumed by solver_dispatch on its own FoamDict instance before the driver ever starts, so it is
        // read by brae -- just not by this object. Listing it would be a permanent false positive.
        "application",
    };
    for (const char* s : skip)
        if (key == s) return true;
    return false;
}

inline void auditWalk(
    const FoamDict& d,
    const std::string& path,
    std::vector<std::string>& out)
{
    for (const auto& l : d.leaves)
    {
        if (auditIgnorable(l.first)) continue;
        if (d.queried.count(l.first)) continue;
        out.push_back(path + l.first);
    }
    for (const auto& s : d.subs)
    {
        if (auditIgnorable(s.first)) continue;
        if (!d.queried.count(s.first))
        {
            // A whole sub-dictionary nobody opened. Report it as one line rather than every leaf inside
            // it -- "brae never looked at PIMPLE" is the finding; its twelve entries are one fact.
            out.push_back(path + s.first + "/ (whole sub-dictionary never read)");
            continue;
        }
        auditWalk(s.second, path + s.first + "/", out);
    }
}

}   // namespace detail

// Name every entry in `d` that brae never looked up. `label` is the file, e.g. "system/fvSolution".
// `partial` marks a report taken from a run that STOPPED EARLY -- see DictAuditScope.
inline void auditDict(const FoamDict& d, const std::string& label, bool partial = false)
{
    if (const char* e = std::getenv("BRAE_DICT_AUDIT"))
        if (std::atoi(e) == 0) return;

    std::vector<std::string> unread;
    detail::auditWalk(d, "", unread);
    if (unread.empty()) return;

    std::fprintf(stderr, "brae NOTICE [unread] %s -- %zu entr%s brae never looked at%s:\n",
                 label.c_str(), unread.size(), unread.size() == 1 ? "y" : "ies",
                 partial ? " BEFORE THE RUN STOPPED" : "");
    for (const std::string& k : unread)
        std::fprintf(stderr, "brae NOTICE [unread]   %s\n", k.c_str());
    if (partial)
        std::fprintf(stderr,
                     "brae NOTICE [unread]   (PARTIAL: the run ended early, so an entry here may simply not have\n"
                     "brae NOTICE [unread]    been REACHED yet -- residualControl is read inside the iteration loop,\n"
                     "brae NOTICE [unread]    for instance. Compare against a case that runs to completion.)\n");
    else
        std::fprintf(stderr,
                     "brae NOTICE [unread]   (an entry here is EVIDENCE of an unimplemented input, not proof --\n"
                     "brae NOTICE [unread]    some belong to other tools. Set BRAE_DICT_AUDIT=0 to silence.)\n");
}

// E5: audit on EVERY exit, including a refusal.
//
// The audit used to sit at the end of the run, so any case brae refuses was never audited -- and the cases
// brae refuses are the biggest and most realistic ones. aerofoilNACA0012 and angledDuctExplicitFixedCoeff
// both stop on fvOptions (A2), so the two stock tutorials most worth auditing were precisely the two that
// never were. A scope guard fires on both paths: normal return and thrown exception.
//
// The report is marked PARTIAL on the exception path, and that distinction is not cosmetic. On an early
// stop, an unread entry may simply not have been REACHED -- residualControl is queried inside the
// iteration loop -- so "unread" stops meaning "unimplemented". Saying so is the difference between a
// gap list and a pile of false positives.
//
// LIFETIME: every registered dict must outlive this object, so declare it AFTER all of them. It holds
// pointers, not copies, because `queried` keeps filling right up to the moment of the report.
class DictAuditScope
{
public:
    void add(const FoamDict& d, std::string label) { entries_.push_back({&d, std::move(label)}); }

    ~DictAuditScope()
    {
        // Non-zero only when we are unwinding, i.e. brae threw. C++17's uncaught_exceptions() is the
        // supported way to tell a destructor which path it is on.
        const bool partial = std::uncaught_exceptions() > 0;
        for (const auto& e : entries_) auditDict(*e.first, e.second, partial);
    }

private:
    std::vector<std::pair<const FoamDict*, std::string>> entries_;
};

}   // namespace brae
