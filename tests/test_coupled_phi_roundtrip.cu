// phi's COUPLED-patch flux must survive a write -> read round trip.
//
// WHY THIS EXISTS. A cyclic/cyclicAMI/cyclicACMI face's flux is not in phi's boundary array at all: it
// lives on the interface object (cyc_.phi / ami_.phi), because that is what the coupling reads. brae's
// writer therefore emitted those patches as a bare `type cyclicAMI;` with no value, and its reader
// explicitly skipped them "since the ctor recomputes cyclic/AMI flux from U". OpenFOAM writes the
// values, and the rebuild is NOT the same number -- phi is phiHbyA minus the pressure flux, fvc::flux(U)
// is not.
//
// It is not a small difference either, because the momentum interface coefficient is upwind: OF's
// gaussConvectionScheme puts max(phi,0) straight on the diagonal. Measured on
// pimpleFoam/RAS/oscillatingInletACMI2D, restarting from OpenFOAM's own t=0.01:
//
//     stored phi, face 0                 1.557610e-03
//     brae's fvc::flux(U) rebuild        1.425933e-03      difference 1.3168e-04
//     resulting error in rAU             0.9455% on all 40 source-side interface cells, 0 elsewhere
//
// and the 1.3168e-04 matched the rAU discrepancy to five figures -- the whole of it. Fixing the round
// trip took the same comparison to 5e-14.
//
// WHAT IS ASSERTED. Leg 1: values written for a coupled patch come back byte-for-byte (the writer uses
// >=17 digits precisely so a restart is bit-identical, and a flux that round-trips to 15 digits would
// still reintroduce a continuity transient). Leg 2, the NEGATIVE CONTROL: a file with no value on the
// coupled patch -- brae's own older writes, and any solver with no interface -- must still read, giving
// zeros for the caller to overwrite with its rebuild, NOT an exception and not garbage. Leg 3 guards
// against the test passing for the wrong reason: the coupled values must differ from the non-coupled
// ones, so a reader that mixed the two up could not pass leg 1.
#include "read_surface_field.cuh"
#include "foam_field_writer.cuh"
#include "fv_patch.cuh"
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

FvPatch mkPatch(const std::string& name, const std::string& type, label start, label size)
{
    FvPatch p;
    p.name = name;
    p.type = type;
    p.start = start;
    p.size = size;
    return p;
}

}   // namespace

int main(int argc, char** argv)
{
    const std::string dir = (argc > 1) ? argv[1] : "coupled_phi_roundtrip_work";
    std::filesystem::create_directories(dir);

    // inlet (calculated, 3 faces) + ACMI1_couple (cyclicAMI, 4) + walls (calculated, 2).
    const std::vector<FvPatch> fvp = {
        mkPatch("inlet",        "patch",      100, 3),
        mkPatch("ACMI1_couple", "cyclicAMI",  103, 4),
        mkPatch("walls",        "wall",       107, 2),
    };
    const std::vector<scalar> phiInt = {1.5, -2.25, 3.125, 4.0};
    // phiBoundary EXCLUDES coupled patches -- that is the whole reason the interface flux needs its own path.
    const std::vector<scalar> phiBnd = {-1.0, -1.5, -2.0,   0.0, 0.0};
    // deliberately awkward values: a negative, a tiny one, and one needing all 17 digits to round-trip
    const std::vector<scalar> ifPhi = {1.0/3.0, -7.25e-13, 2.718281828459045, -1.0/7.0};

    // ---- leg 1: values round-trip exactly ----
    {
        const std::string f = dir + "/phi";
        std::map<std::string, std::vector<scalar>> coupled{{"ACMI1_couple", ifPhi}};
        writeSurfaceField(f, phiInt, phiBnd, fvp, 17, "[0 3 -1 0 0 0 0]", coupled);
        const SurfaceScalarField r = readSurfaceField(f, fvp, (label)phiInt.size());

        if (r.boundary[1].size() != ifPhi.size())
        {
            std::printf("  FAIL coupled patch read back %zu values, expected %zu -- the writer emitted no\n"
                        "       value and the flux cannot survive a restart\n",
                        r.boundary[1].size(), ifPhi.size());
            ++failures;
        }
        else
        {
            for (std::size_t i = 0; i < ifPhi.size(); ++i)
                if (r.boundary[1][i] != ifPhi[i])   // EXACT: 17 digits is chosen to make this hold
                {
                    std::printf("  FAIL coupled face %zu: wrote %.17g, read %.17g (diff %.3e)\n",
                                i, (double)ifPhi[i], (double)r.boundary[1][i],
                                (double)(r.boundary[1][i] - ifPhi[i]));
                    ++failures;
                }
        }
        // the ordinary patches must be untouched by the new path
        for (std::size_t i = 0; i < 3; ++i)
            if (r.boundary[0][i] != phiBnd[i])
            { std::printf("  FAIL inlet face %zu changed: %.17g vs %.17g\n", i, (double)r.boundary[0][i], (double)phiBnd[i]); ++failures; }
        for (std::size_t i = 0; i < phiInt.size(); ++i)
            if (r.internal[i] != phiInt[i])
            { std::printf("  FAIL internal face %zu changed\n", i); ++failures; }
        std::printf("  round trip: coupled patch %zu values, exact\n", ifPhi.size());
    }

    // ---- leg 2 (negative control): no value on the coupled patch is still readable ----
    {
        const std::string f = dir + "/phi_novalue";
        writeSurfaceField(f, phiInt, phiBnd, fvp, 17);   // no coupledValues -> the old value-less form
        bool threw = false;
        SurfaceScalarField r;
        try { r = readSurfaceField(f, fvp, (label)phiInt.size()); }
        catch (const std::exception& e) { threw = true; std::printf("  FAIL value-less coupled patch threw: %s\n", e.what()); ++failures; }
        if (!threw)
        {
            if (r.boundary[1].size() != 4)
            { std::printf("  FAIL value-less coupled patch sized %zu, expected 4\n", r.boundary[1].size()); ++failures; }
            else for (std::size_t i = 0; i < 4; ++i)
                if (r.boundary[1][i] != scalar(0))
                {
                    std::printf("  FAIL value-less coupled face %zu read %.17g, expected 0 (the caller's\n"
                                "       rebuild from U is what fills it)\n", i, (double)r.boundary[1][i]);
                    ++failures;
                }
            // and the ordinary patches still work in that file
            for (std::size_t i = 0; i < 3; ++i)
                if (r.boundary[0][i] != phiBnd[i])
                { std::printf("  FAIL value-less file: inlet face %zu wrong\n", i); ++failures; }
            std::printf("  value-less coupled patch: read as zeros, no exception\n");
        }
    }

    // ---- leg 3 (vacuity guard) ----
    {
        bool distinct = false;
        for (scalar a : ifPhi)
        {
            bool clash = false;
            for (scalar b : phiBnd) if (a == b) clash = true;
            for (scalar b : phiInt) if (a == b) clash = true;
            if (!clash) { distinct = true; break; }
        }
        if (!distinct)
        {
            std::printf("  FAIL vacuous: the coupled values coincide with the boundary/internal ones, so a\n"
                        "       reader that took them from the wrong list would still pass leg 1\n");
            ++failures;
        }
    }

    std::printf("coupled_phi_roundtrip: %d failures\n", failures);
    return failures ? 1 : 0;
}
