// A coupled interface brae does not implement must be REFUSED, not solved as two ordinary boundaries.
//
// A polyMesh patch that names a `neighbourPatch` is declaring itself one half of a coupled pair: its
// face values come from the other side, not from a boundary condition. brae builds that coupling for
// cyclic, cyclicAMI and cyclicACMI. Any other such type reaching the solver would have its two sides
// solved as unconnected boundaries -- and, because they still look like ordinary patches, converge
// quietly to a wrong answer.
//
// THIS IS THE CASE THAT MOTIVATED IT, and it is now implemented -- the guard is what stopped it being
// solved wrongly in the meantime, and `cyclicSlip` below stands in its place as a coupled type brae
// still does not build. `cyclicPeriodicAMI` -- an AMI whose two sides span different sectors, which
// OpenFOAM covers by applying a periodic transform repeatedly until the target is tiled -- reached the
// solver undetected, and the reason is worth remembering: its mesh entry carries
//     inGroups 1(cyclicAMI);
// so the FIELD side matched setConstraintTypes' cyclicAMI entry and happily built a cyclicAMI patch
// field, while the MESH side (buildAMIInterfaces) tested the real type and skipped it. The field
// believed it was coupled; nothing had coupled it. Neither half ever saw an unknown type, so nothing
// complained. The measured result on pimpleFoam/RAS/oscillatingInletPeriodicAMI2D was U identically
// zero on all 96 faces of the downstream half and no flux through the interface at all, against
// OpenFOAM's 1.0e-01; on RAS/axialTurbine, a turbine whose three components did not conserve mass
// between them (11% imbalance across one rotor-stator pair, where OpenFOAM matches to 1e-5).
//
// The guard keys on the STRUCTURAL marker -- the presence of neighbourPatch -- rather than on a list of
// unimplemented type names, so a coupled type nobody has thought of yet is refused too. Leg 4 is what
// pins that: it uses an invented type name that cannot appear in any list.
#include "box_mesh.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

using namespace brae;

namespace {
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       std::printf("  ok:   %s\n", what);
}

// Re-stamp one patch's type (and optionally its neighbourPatch) onto a copy of the mesh.
PrimitiveMesh retyped(const PrimitiveMesh& src, const std::string& patchName,
                      const std::string& type, const std::string& nbr)
{
    std::vector<PatchInfo> pp = src.patches();
    for (PatchInfo& q : pp)
        if (q.name == patchName) { q.type = type; q.neighbourPatch = nbr; }
    PrimitiveMesh out;
    out.assign(src.points(), src.faceVerts(), src.faceOffsets(), src.owner(), src.neighbour(),
               std::move(pp), src.nCells());
    return out;
}

// buildPatches needs geometry; build it from the mesh under test so a refusal is the only failure mode.
std::string buildAndCatch(const PrimitiveMesh& m)
{
    try
    {
        FvGeometry g;
        g.build(m);
        const std::vector<FvPatch> fvp = buildPatches(m, g);
        (void)fvp;
        return "";
    }
    catch (const std::exception& e) { return e.what(); }
}
} // namespace

int main()
{
    std::printf("== refusal: an unimplemented COUPLED patch ==\n");

    const PrimitiveMesh base = boxtest::boxMesh(4, 3, 2);

    // ---- Leg 1: cyclicPeriodicAMI is refused, and the message says what and why -------------------
    {
        const std::string msg = buildAndCatch(retyped(base, "inlet", "cyclicSlip", "outlet"));
        check(!msg.empty(), "cyclicSlip with a neighbourPatch is REFUSED (brae does not couple it)");
        check(msg.find("inlet") != std::string::npos && msg.find("cyclicSlip") != std::string::npos,
              "...and the message names the offending patch and its type");
        check(msg.find("outlet") != std::string::npos,
              "...and the neighbour it would have been coupled to");
    }

    // ---- Leg 2: the three types brae DOES couple still build ---------------------------------------
    // Without this, deleting the AMI machinery entirely would pass Leg 1.
    {
        const std::string a = buildAndCatch(retyped(base, "inlet", "cyclicAMI", "outlet"));
        check(a.empty(), "negative control: cyclicAMI is accepted");
        const std::string c = buildAndCatch(retyped(base, "inlet", "cyclic", "outlet"));
        check(c.empty(), "negative control: cyclic is accepted");
        const std::string pp = buildAndCatch(retyped(base, "inlet", "cyclicPeriodicAMI", "outlet"));
        check(pp.empty(), "negative control: cyclicPeriodicAMI is accepted now that it is implemented");
    }

    // ---- Leg 3: an ordinary patch is untouched -----------------------------------------------------
    {
        const std::string w = buildAndCatch(base);
        check(w.empty(), "negative control: the plain wall/patch mesh still builds");
    }

    // ---- Leg 4: the guard is structural, not a name list -------------------------------------------
    // An invented coupled type cannot be in any list of known-unimplemented names, so this leg can only
    // pass if the refusal keys on neighbourPatch.
    {
        const std::string msg = buildAndCatch(retyped(base, "inlet", "cyclicSomethingNobodyWroteYet", "outlet"));
        check(!msg.empty(), "an UNKNOWN coupled type is refused too -- the guard is not a name list");
    }

    // ---- Leg 5: ...and it does not fire on a type that declares no coupling ------------------------
    // The marker is the coupling, not the unfamiliarity: a patch with no neighbourPatch is an ordinary
    // boundary and is left to the boundary-condition factory to accept or reject. Pinning this keeps
    // the guard from quietly widening into "refuse anything unrecognised" later.
    {
        const std::string msg = buildAndCatch(retyped(base, "inlet", "someOrdinaryPatchType", ""));
        check(msg.empty(), "a patch with NO neighbourPatch is not this guard's business");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
