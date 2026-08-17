// The wall-function geometry has to be REBUILT when the mesh moves.
//
// DeviceWallData is a snapshot of the mesh: the near-wall distance, the wall-face deltaCoeffs, the wall
// velocity, and -- the reason this test exists -- the cyclicACMI area fractions that decide which
// non-overlap faces count as wall at all. brae built it once in the constructor and never again, while
// moveMesh already rebuilt the AMI weights beside it. A sliding ACMI re-splits its coupled areas from
// the overlap mask on EVERY move (cyclicACMIPolyPatch::movePoints -> scalePatchFaceAreas), so from the
// second step on the wall gate was answering a question about the mesh at t = 0.
//
// MEASURED on pimpleFoam/RAS/oscillatingInletACMI2D, moving, kEpsilon: the wall weight of some cell moves
// by 0.15 every single step, and an isWallCell entry flips outright on 2 of the 10 steps. Refreshing
// changes k by 1.0e-03 (L2 rel) at step 1 and 1.0e-01 by step 10 -- so this was live, not cosmetic. It is
// NOT the dominant error on that case (k is already 31% out in the moving zone at step 1, which the
// refresh does not touch), and this test asserts only what the refresh itself is responsible for.
//
// THE SECOND HALF is the wall velocity, and it is why refreshWallData reads dbU_ instead of a host U
// field. `movingWallVelocity` is assigned straight into the solver's device boundary by setPatchVelocity
// after the move; the host GeometricField still says (0 0 0). On that case the wall slides at 1.57 m/s
// against a 1 m/s inlet, so a wall function fed 0 has the near-wall velocity gradient badly wrong. Leg 3
// pins the read to the buffer the momentum assembly actually uses.
#include "acmi_mesh.cuh"
#include "acmi_area_scaling.cuh"
#include "device_simple_foam.cuh"
#include "cyclic_field.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

// The rule buildDeviceWallData implements, recomputed here from the geometry alone: a cell's wall weight
// is the largest areaFraction among its wall faces. Deliberately a re-derivation, not a call.
std::vector<scalar> expectedWallW(const PrimitiveMesh& m, const FvGeometry& g,
                                  const std::vector<FvPatch>& fvp)
{
    std::vector<scalar> w(m.nCells(), 0.0);
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (fvp[pi].type != "wall") continue;
        for (label i = 0; i < fvp[pi].size; ++i)
        {
            const label f = fvp[pi].start + i;
            const scalar raw = g.rawMagSf(f);
            const scalar frac = raw > scalar(0) ? g.magSf()[f]/raw : scalar(1);
            const label c = fvp[pi].faceCells[i];
            if (frac > w[c]) w[c] = frac;
        }
    }
    return w;
}

scalar worst(const std::vector<scalar>& a, const std::vector<scalar>& b)
{
    scalar d = 0;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) d = std::fmax(d, std::fabs(a[i] - b[i]));
    return d;
}

}   // namespace

int main()
{
    // Two overlap states of the same interface: the fixture's own offset, and the interface slid by half
    // a face. Same topology, same point ORDER -- only the positions differ, which is exactly what a
    // solidBody motion of a cellZone does.
    const scalar dy0 = acmitest::ACMI_DY;
    const scalar dy1 = acmitest::ACMI_DY + 0.125;

    PrimitiveMesh m = acmitest::twoBlockACMI(dy0, /*withBlockage*/true, "ACMI1_blockage",
                                             /*frontBackEmpty*/true);
    const std::vector<vector> movedPts =
        acmitest::twoBlockACMI(dy1, true, "ACMI1_blockage", true).points();

    FvGeometry g;
    std::vector<FvPatch> fvp;
    std::vector<AMIInterface> amis;
    buildGeometryPatchesAndAMI(m, g, fvp, amis);
    const label nC = m.nCells();

    GeometricField<vector> U =
        buildCyclicField<vector>(std::vector<vector>(nC), fvp, {}, /*wallNoSlip*/true);
    U.evaluateBoundary();
    GeometricField<scalar> p = buildCyclicField<scalar>(std::vector<scalar>(nC), fvp, {});
    p.evaluateBoundary();
    SurfaceScalarField phi = fvc::flux(U, m, g, fvp);

    DeviceSimpleControls ctl;
    ctl.nu = 1e-3;
    ctl.turbulent = false;
    DeviceSimpleSolver solver(m, g, fvp, U, p, phi, ctl);

    const std::vector<scalar> wantBefore = expectedWallW(m, g, fvp);
    const std::vector<scalar> gotBefore  = solver.wallData().wallW.host();
    std::printf("  at construction: max|wallW - rule| = %.3e\n", (double)worst(gotBefore, wantBefore));
    if (worst(gotBefore, wantBefore) > 1e-12)
    { std::printf("  FAIL the constructor's wall data does not match its own mesh\n"); ++failures; }

    // ---- move the interface ----
    m.movePoints(movedPts);
    rebuildGeometryWithACMI(m, g, fvp);
    const std::vector<scalar> wantAfter = expectedWallW(m, g, fvp);

    // VACUITY GUARD, and the whole premise: the slide must actually change the mask. If it did not, a
    // solver that never refreshed would pass every assertion below.
    const scalar moved = worst(wantBefore, wantAfter);
    std::printf("  the slide changes the wall weight by up to %.3e\n", (double)moved);
    if (moved < 1e-3)
    {
        std::printf("  FAIL vacuous: this displacement leaves the ACMI mask alone, so nothing here can\n"
                    "       distinguish a refreshed wall gate from a stale one\n");
        ++failures;
    }

    // ---- 1. STALE: the un-refreshed data still describes the OLD mesh ----
    {
        const std::vector<scalar> got = solver.wallData().wallW.host();
        std::printf("  before refresh : max|wallW - new rule| = %.3e (expected: still the old mesh)\n",
                    (double)worst(got, wantAfter));
        if (worst(got, wantBefore) > 1e-12)
        { std::printf("  FAIL moving the mesh changed the wall data by itself -- leg 2 would then prove nothing\n"); ++failures; }
    }

    // ---- 2. REFRESHED: it describes the mesh it is now on ----
    {
        solver.refreshWallData(m, g, fvp);
        const std::vector<scalar> got = solver.wallData().wallW.host();
        const scalar d = worst(got, wantAfter);
        std::printf("  after refresh  : max|wallW - new rule| = %.3e (must be 0)\n", (double)d);
        if (d > 1e-12)
        {
            std::printf("  FAIL the wall gate is still answering for the mesh at t = 0. A cyclicACMI\n"
                        "       re-splits its coupled areas on every move, so this data is a function of\n"
                        "       the CURRENT geometry, exactly like the AMI weights beside it.\n");
            ++failures;
        }
    }

    // ---- 3. the wall VELOCITY comes from the device boundary, not a host field ----
    // setPatchVelocity is how movingWallVelocity reaches the solver, and it writes dbU_ only. A refresh
    // that re-read the host U would hand the wall functions (0 0 0) on a wall sliding at 1.57 m/s.
    {
        label wallPatch = -1;
        for (std::size_t pi = 0; pi < fvp.size(); ++pi)
            if (fvp[pi].type == "wall" && fvp[pi].size > 0) { wallPatch = static_cast<label>(pi); break; }
        if (wallPatch < 0)
        { std::printf("  FAIL vacuous: the fixture has no wall patch to assign a velocity to\n"); ++failures; }
        else
        {
            const vector Uw{0.0, 1.57, 0.0};   // the tutorial's actual wall speed
            solver.setPatchVelocity(wallPatch, std::vector<vector>(fvp[wallPatch].size, Uw));
            solver.refreshWallData(m, g, fvp);
            const std::vector<scalar> uy = solver.wallData().wfUwy.host();
            scalar mx = 0;
            for (scalar v : uy) mx = std::fmax(mx, std::fabs(v));
            std::printf("  wall velocity  : max|wfUwy| = %.4g (assigned %.4g on patch '%s')\n",
                        (double)mx, (double)Uw.y, fvp[wallPatch].name.c_str());
            if (std::fabs(mx - Uw.y) > 1e-12)
            {
                std::printf("  FAIL the refresh did not pick up setPatchVelocity. dbU_ is what the momentum\n"
                            "       assembly imposes; a host U field is not, and on a moving wall they differ\n");
                ++failures;
            }
        }
    }

    std::printf("wall_data_refresh: %d failures\n", failures);
    return failures ? 1 : 0;
}
