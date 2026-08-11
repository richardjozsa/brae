#pragma once
// stage_dump.cuh -- write one SIMPLE stage's field to a plain text file, for stage-by-stage comparison
// against OpenFOAM (cudafoam/rhosimplefoam-restage-plan.md, Phase 0).
//
// WHY THIS EXISTS. SIMPLE is segregated, so an error in one stage appears DOWNSTREAM and never upstream:
//
//     rAU/rAtU -> HbyA -> phiHbyA -> phid -> p -> U -> he -> turbulence
//
// Comparing in that order localises a defect to one stage instead of a whole solver. The squareBend hunt
// walked that chain backwards by hand and landed on HbyA (+5.11% on internal faces while its boundary was
// exactly right); this makes that a command instead of an afternoon.
//
// WHY PLAIN TEXT, not OpenFOAM format. Writing OF-format fields from brae needs a template file per field
// (writeVolField reads one for the FoamFile header and dimensions), and a `stage_HbyA` has no template in
// the case. That plumbing is a place to introduce bugs in the instrument itself, which is the one thing an
// instrument may not do. The comparison tool reads OF's format on one side and this on the other; the
// asymmetry costs nothing because only ONE side has to be a real OpenFOAM field.
//
// Format: a header line "n <count> <components>", then one value per line (vectors: x y z on one line).
//
// Enabled by BRAE_DUMP_STAGE=<dir>. Off by default and free when off.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <map>
#include <set>
#include <vector>

namespace brae {

// The directory to dump into, or empty when the harness is off.
inline const std::string& stageDumpDir()
{
    static const std::string d = []() -> std::string
    {
        const char* e = std::getenv("BRAE_DUMP_STAGE");
        if (!e || !*e) return {};
        std::error_code ec;
        std::filesystem::create_directories(e, ec);
        return std::string(e);
    }();
    return d;
}

inline bool stageDumpActive() { return !stageDumpDir().empty(); }

// One scalar field (cell or face -- the count says which).
inline void stageDump(const std::string& name, const std::vector<scalar>& v)
{
    if (!stageDumpActive()) return;
    std::ofstream out(stageDumpDir() + "/" + name);
    if (!out) { std::fprintf(stderr, "brae: stage dump could not open %s\n", name.c_str()); return; }
    out.precision(17);
    out << "n " << v.size() << " 1\n";
    for (scalar x : v) out << x << '\n';
}

inline void stageDump(const std::string& name, const DeviceBuffer<scalar>& b)
{
    if (!stageDumpActive()) return;
    stageDump(name, b.host());
}

// A vector field held as three component buffers, which is how brae stores U and HbyA.
inline void stageDump3(
    const std::string& name,
    const DeviceBuffer<scalar>& x,
    const DeviceBuffer<scalar>& y,
    const DeviceBuffer<scalar>& z)
{
    if (!stageDumpActive()) return;
    const std::vector<scalar> hx = x.host(), hy = y.host(), hz = z.host();
    std::ofstream out(stageDumpDir() + "/" + name);
    if (!out) { std::fprintf(stderr, "brae: stage dump could not open %s\n", name.c_str()); return; }
    out.precision(17);
    out << "n " << hx.size() << " 3\n";
    for (std::size_t i = 0; i < hx.size(); ++i) out << hx[i] << ' ' << hy[i] << ' ' << hz[i] << '\n';
}

// Only iteration 1 is dumped: the comparison is against OF's own iteration 1, and a later iteration on one
// side has already been contaminated by every earlier difference on that side.
//
// The latch is PER CALL SITE. A single global latch is the obvious implementation and it is wrong: the
// first site to run consumes it and every other site silently dumps nothing. That happened -- adding a
// second dump point made the first one stop writing, and the comparison reported "missing" for stages that
// were being computed perfectly well.
// ---------------------------------------------------------------------------------------------------
// PRESSURE STAGE TRACE. BRAE_PTRACE=<dir> writes <dir>/<name>_it<N> for iterations 1..BRAE_PTRACE_N
// (default 4). Deliberately separate from stageDump's single-iteration latch above.
//
// WHY A SECOND MECHANISM. stageDumpFirstOnly answers "does stage X match OF at iteration 1", which is
// the right question for a term that is either right or wrong. It is the WRONG question for a solver
// that agrees at iteration 1 and diverges at iteration 3 -- there the answer is a column index, not a
// number, and one latched iteration cannot produce it.
//
// WHY ONE RUN, not N runs with BRAE_DUMP_STAGE_ITER=1..N. brae's compressible path is not run-to-run
// reproducible (~1e-3, see the nondeterminism note in the rhoSimpleFoam validation log). Stitching
// iteration 3 from one process and iteration 4 from another compares that noise, not the trajectory.
inline const std::string& pTraceDir()
{
    static const std::string d = []() -> std::string
    {
        const char* e = std::getenv("BRAE_PTRACE");
        if (!e || !*e) return {};
        std::error_code ec;
        std::filesystem::create_directories(e, ec);
        return std::string(e);
    }();
    return d;
}

inline bool pTraceActive() { return !pTraceDir().empty(); }

inline int pTraceN()
{
    static const int n = []() {
        const char* e = std::getenv("BRAE_PTRACE_N");
        const int v = (e && *e) ? std::atoi(e) : 4;
        return v > 0 ? v : 4;
    }();
    return n;
}

inline void pTraceDump(const std::string& name, int it, const std::vector<scalar>& v)
{
    if (!pTraceActive() || it < 1 || it > pTraceN()) return;
    std::ofstream out(pTraceDir() + "/" + name + "_it" + std::to_string(it));
    if (!out) { std::fprintf(stderr, "brae: ptrace could not open %s\n", name.c_str()); return; }
    out.precision(17);
    out << "n " << v.size() << " 1\n";
    for (scalar x : v) out << x << '\n';
}

inline void pTraceDump(const std::string& name, int it, const DeviceBuffer<scalar>& b)
{
    if (!pTraceActive() || it < 1 || it > pTraceN()) return;
    pTraceDump(name, it, b.host());
}

// Scalars that describe the solve rather than a field (residuals, iteration counts).
inline void pTraceNote(int it, const std::string& key, double value)
{
    if (!pTraceActive() || it < 1 || it > pTraceN()) return;
    std::ofstream out(pTraceDir() + "/notes_it" + std::to_string(it), std::ios::app);
    if (!out) return;
    out.precision(17);
    out << key << ' ' << value << '\n';
}

inline bool stageDumpFirstOnly(const char* site)
{
    // Which visit to dump. Iteration 1 is the default because it is the only one both sides reach with
    // identical inputs; BRAE_DUMP_STAGE_ITER=2 walks the SECOND iteration once the first is known good,
    // which is how a defect that only appears after the first update step gets localised.
    static const int target = []() {
        const char* e = std::getenv("BRAE_DUMP_STAGE_ITER");
        const int v = (e && *e) ? std::atoi(e) : 1;
        return v > 0 ? v : 1;
    }();
    static std::map<std::string, int> visits;
    return ++visits[site] == target;
}

}   // namespace brae
