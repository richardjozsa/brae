// cyclicACMI area scaling -- OF cyclicACMIPolyPatch::scalePatchFaceAreas (cyclicACMIPolyPatch.C:187).
//
// THE MECHANISM UNDER TEST. An ACMI interface is the SAME FACE twice: once on the coupled patch, once
// on a coincident wall (its nonOverlapPatch). OF's blockMeshDict lists one block face under both. The
// overlap mask splits that face's area between them:
//
//     coupled  Sf = Sf_raw * max(tol, mask)
//     wall     Sf = Sf_raw * (1 - min(max(mask, tol), 1 - tol))          tol = 1e-10
//
// so a face that has slid off its neighbour hands its area to the wall and becomes solid rather than
// staying coupled to nothing. All of ACMI's physics is carried by that split -- there is no special
// discretisation anywhere.
//
// WHY LEG 1 DELIBERATELY MEASURES A BROKEN MESH. The duplicated face is not free: cell volumes come
// from a pyramid sum over a cell's faces, so before scaling the shared face is counted TWICE and every
// adjacent cell has the wrong volume. That is why OF scales before cell geometry is computed and says
// so outright (cyclicACMIPolyPatch.C:452): "Initialise the AMI early to make sure we adapt the face
// areas before the cell centre calculation gets triggered." Leg 1 measures the damage on this fixture,
// so leg 2's correctness is a demonstrated repair and not an unfalsifiable "it looks right".
//
// EXACT, NOT TOLERANCED. Every cell here is 1 x 0.25 x 0.1 = 0.025 by construction, and the mask is
// hand-computable (0, 0.5, 1, 1 -- see acmi_mesh.cuh). So the assertions are equalities to round-off,
// including the conservation identity that the two scales sum to 1: whatever area leaves the coupled
// patch must arrive at the wall, or the interface has quietly gained or lost area.
#include "acmi_mesh.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "ami_interface.cuh"
#include "acmi_area_scaling.cuh"
#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

using namespace brae;

namespace {

int failures = 0;

label patchIdx(const std::vector<FvPatch>& fvp, const std::string& n)
{
    for (std::size_t i = 0; i < fvp.size(); ++i) if (fvp[i].name == n) return static_cast<label>(i);
    return -1;
}

}   // namespace

int main()
{
    PrimitiveMesh m = acmitest::twoBlockACMI(acmitest::ACMI_DY, /*withBlockage*/ true);

    // The mask depends ONLY on the face polygons, so it is computed from raw geometry -- the same
    // invariant OF relies on by building its AMI on the primitivePatch (raw point) areas rather than on
    // the scaled polyPatch ones. Normalising coverage by an already-scaled area would return 1 for
    // every face and quietly destroy the mask.
    FvGeometry g;
    g.build(m);
    std::vector<FvPatch> fvp = buildPatches(m, g);
    const std::vector<AMIInterface> amis = buildAMIInterfaces(m, g, fvp);

    const label ic = patchIdx(fvp, "ACMI1_couple"), iw = patchIdx(fvp, "ACMI1_blockage");
    if (ic < 0 || iw < 0 || amis.size() != 2)
    {
        std::printf("  FAIL fixture: couple=%d blockage=%d interfaces=%zu\n", (int)ic, (int)iw, amis.size());
        std::printf("test_acmi_area_scaling: %d failures\n", ++failures);
        return 1;
    }
    // Raw areas, before any scaling, to compare the split against.
    std::vector<scalar> rawC(fvp[ic].size), rawW(fvp[ic].size);
    for (label i = 0; i < fvp[ic].size; ++i)
    {
        rawC[i] = g.magSf()[fvp[ic].start + i];
        rawW[i] = g.magSf()[fvp[iw].start + i];
    }

    // -------------------------------------------------------------------------------------------
    // 1. WITHOUT scaling the duplicated face is double-counted and the volumes are wrong. Measured so
    // that leg 2 is a demonstrated repair.
    scalar worstBefore = 0;
    for (label c = 0; c < m.nCells(); ++c)
        worstBefore = std::fmax(worstBefore, std::fabs(g.V()[c] - acmitest::ACMI_CELL_VOL));
    std::printf("  unscaled  : worst |V - %.4f| = %.6e  (duplicate face counted twice)\n",
                acmitest::ACMI_CELL_VOL, worstBefore);
    if (worstBefore < 1e-9)
    {
        std::printf("  FAIL vacuous: cell volumes are already correct without scaling, so this fixture\n"
                    "       has no duplicated faces and cannot show that the scaling does anything\n");
        ++failures;
    }

    // -------------------------------------------------------------------------------------------
    // 2. Rebuild with the scaling in between -- face geometry, scale, THEN cell geometry.
    g.buildFaceGeometry(m);
    applyACMIAreaScaling(m, g, fvp, amis);
    g.buildCellGeometry(m);

    scalar worstAfter = 0;
    for (label c = 0; c < m.nCells(); ++c)
        worstAfter = std::fmax(worstAfter, std::fabs(g.V()[c] - acmitest::ACMI_CELL_VOL));
    std::printf("  scaled    : worst |V - %.4f| = %.6e\n", acmitest::ACMI_CELL_VOL, worstAfter);
    if (worstAfter > 1e-12)
    {
        std::printf("  FAIL cell volumes are still wrong after scaling: the coupled and non-overlap areas\n"
                    "       do not sum to the raw face area, so the shared face is not being split\n");
        ++failures;
    }

    // -------------------------------------------------------------------------------------------
    // 3. THE SPLIT ITSELF: coupled gets mask, wall gets the rest, and the two sum to the raw area.
    // Conservation is the assertion that matters -- area silently created or lost at the interface is
    // exactly the failure mode ACMI exists to avoid.
    std::printf("  face :   mask   coupled/raw   wall/raw     sum\n");
    for (label i = 0; i < fvp[ic].size; ++i)
    {
        const scalar mk = amis[0].mask[i];
        const scalar fc = g.magSf()[fvp[ic].start + i]/rawC[i];
        const scalar fw = g.magSf()[fvp[iw].start + i]/rawW[i];
        std::printf("   %2d  : %6.3f   %10.6f  %10.6f  %10.6f\n", (int)i, mk, fc, fw, fc + fw);

        const scalar wantC = std::max(ACMI_TOLERANCE, mk);
        const scalar wantW = scalar(1) - std::min(std::max(mk, ACMI_TOLERANCE), scalar(1) - ACMI_TOLERANCE);
        if (std::fabs(fc - wantC) > 1e-12)
        { std::printf("  FAIL face %d coupled scale %.12f, expected %.12f\n", (int)i, fc, wantC); ++failures; }
        if (std::fabs(fw - wantW) > 1e-12)
        { std::printf("  FAIL face %d wall scale %.12f, expected %.12f\n", (int)i, fw, wantW); ++failures; }
        if (std::fabs((fc + fw) - scalar(1)) > 2*ACMI_TOLERANCE)
        {
            std::printf("  FAIL face %d: coupled + wall = %.12f, not 1. The interface has %s area.\n",
                        (int)i, fc + fw, (fc + fw) > 1 ? "gained" : "lost");
            ++failures;
        }
    }

    // -------------------------------------------------------------------------------------------
    // 4. SCALING IS FROM RAW, ENFORCED. Applying it twice without rebuilding face geometry would
    // compound the mask every mesh move and shrink the interface away. That must be a hard error, not
    // a silently smaller interface.
    {
        bool threw = false;
        try { applyACMIAreaScaling(m, g, fvp, amis); }
        catch (const std::exception&) { threw = true; }
        std::printf("  double-scale without a rebuild: %s\n", threw ? "refused" : "ALLOWED");
        if (!threw)
        {
            std::printf("  FAIL a second scale was accepted; the mask would compound on every mesh move\n");
            ++failures;
        }
    }

    // -------------------------------------------------------------------------------------------
    // 5. A NON-COINCIDENT PAIR IS REFUSED. OF indexes the mask with the same face index on both
    // patches, so the pair must be the same faces in the same order. If they were not, the mask would
    // land on the wrong face -- a mis-split that no residual would ever reveal. Point ACMI1_couple at
    // `outlet`, which has the RIGHT FACE COUNT (4) but sits at x = 2, so only the coincidence check can
    // catch it; a size check alone would let it through.
    {
        PrimitiveMesh bad = acmitest::twoBlockACMI(acmitest::ACMI_DY, true, "outlet");
        FvGeometry gb;
        gb.build(bad);
        const std::vector<FvPatch> fb = buildPatches(bad, gb);
        const std::vector<AMIInterface> ab = buildAMIInterfaces(bad, gb, fb);
        gb.buildFaceGeometry(bad);

        bool threw = false;
        std::string msg;
        try { applyACMIAreaScaling(bad, gb, fb, ab); }
        catch (const std::exception& e) { threw = true; msg = e.what(); }
        std::printf("  wrong-surface nonOverlapPatch (same size): %s\n", threw ? "refused" : "ACCEPTED");
        if (!threw)
        {
            std::printf("  FAIL a nonOverlapPatch that is not the same surface was accepted; the mask\n"
                        "       would be applied to faces it does not describe\n");
            ++failures;
        }
        else if (msg.find("coincident") == std::string::npos)
        {
            std::printf("  FAIL refused, but not by the coincidence check: %s\n", msg.c_str());
            ++failures;
        }
    }

    // -------------------------------------------------------------------------------------------
    // 6. THE ORDERED ENTRY POINT, and the resolution of "do the weights double-count the mask?".
    //
    // THE ANSWER IS NO, AND THIS LEG USED TO ASSERT THAT THEY DO. The reasoning it asserted was:
    //
    //     OF normalises AMI weights two ways (AMIInterpolation.C:159-208), selected by requireMatch:
    //         requireMatch 1 (cyclicAMI) : denom = sum(overlap) -> weights sum to exactly 1
    //         requireMatch 0 (cyclicACMI): denom = face area    -> weights sum to the COVERAGE
    //     the ACMI boundary carries requireMatch 0, and OF's log on oscillatingInletACMI2D prints
    //     sum(weights) average 0.7578655102 = (30 + 0.3146)/40, not 31/40 = 0.775
    //
    // Every line of that is true and the conclusion drawn from it was still wrong, which is what makes
    // it worth keeping. It stops one call short. cyclicACMIPolyPatch::scalePatchFaceAreas scales the
    // coupled Sf by the mask and then, as its last act (cyclicACMIPolyPatch.C:264):
    //
    //     "Re-normalise the weights since the effect of overlap is already accounted for in the area"
    //         for (scalar& w : wghts) { w /= sum; }   sum = 1.0;
    //
    // The log line is printed from INSIDE normaliseWeights, before that override, so it reports the
    // pre-override coverage on a run whose solved weights are 1 -- it cannot distinguish the two, and
    // reading it as confirmation was the error.
    //
    // The invariant is that the coverage appears EXACTLY ONCE. This leg now pins both halves together,
    // which is the only way to state it: the mask still reads 0, 0.5, 1, 1 (it survives, in the area)
    // while every non-empty stencil sums to 1 (it is not there a second time). Asserting either alone
    // passes under the double-count.
    //
    // The cost of the old convention, measured on pimpleFoam/RAS/oscillatingInletACMI2D: 2 blended
    // faces of 136 carried 0.118 of a full face's flux against OF's 0.201, losing 0.9% of the interface
    // mass flow, and that put the inlet channel ~860 above the duct in pressure.
    {
        PrimitiveMesh m6 = acmitest::twoBlockACMI(acmitest::ACMI_DY, true);
        FvGeometry g6;
        std::vector<FvPatch> f6;
        std::vector<AMIInterface> a6;
        buildGeometryPatchesAndAMI(m6, g6, f6, a6);

        scalar worst6 = 0;
        for (label c = 0; c < m6.nCells(); ++c)
            worst6 = std::fmax(worst6, std::fabs(g6.V()[c] - acmitest::ACMI_CELL_VOL));
        std::printf("  entry point: worst |V - %.4f| = %.6e\n", acmitest::ACMI_CELL_VOL, worst6);
        if (worst6 > 1e-12)
        { std::printf("  FAIL entry point left the cell volumes wrong\n"); ++failures; }

        const scalar want[4] = {0.0, 0.5, 1.0, 1.0};
        for (const AMIInterface& A : a6)
        {
            if (f6[A.patch].name != "ACMI1_couple") continue;
            for (label i = 0; i < 4; ++i)
            {
                // the mask survived the scaling
                if (std::fabs(A.mask[i] - want[i]) > 1e-12)
                {
                    std::printf("  FAIL entry point face %d: mask %.12f, expected %.12f -- the weights were\n"
                                "       normalised by an already-scaled area, which erases the mask\n",
                                (int)i, A.mask[i], want[i]);
                    ++failures;
                }
                // ...and the weights do NOT carry it a second time: OF re-normalises them to 1 once the
                // area has the coverage. An uncovered face has no stencil and keeps a sum of 0, which
                // is OF's `if (wghts.size())` guard, not a special case.
                const scalar wantWs = want[i] > 0 ? scalar(1) : scalar(0);
                scalar ws = 0;
                for (label k = A.srcOffset[i]; k < A.srcOffset[i+1]; ++k) ws += A.weight[k];
                if (std::fabs(ws - wantWs) > 1e-12)
                {
                    std::printf("  FAIL entry point face %d: weights sum to %.12f, expected %.12f. The\n"
                                "       coverage %.4f is already in the scaled area (asserted just above);\n"
                                "       carrying it here too transmits mask^2 of the face's flux.\n",
                                (int)i, ws, wantWs, want[i]);
                    ++failures;
                }
            }
            std::printf("  entry point: mask = 0, 0.5, 1, 1 (in the area) and weight-sums = 0, 1, 1, 1\n"
                        "               -- the coverage appears exactly once\n");
        }

        // THE RAW/SCALED DISTINCTION IS ENGAGED, and load-bearing. buildAMIInterfaces normalises by
        // g.rawMagSf(), so the trap of normalising by an already-scaled area is now structurally
        // impossible to reach from outside -- which is the point of keeping the raw areas inside
        // FvGeometry rather than threading them through every call site. What can still be checked
        // here is that the two really do differ, and by how much the mask would be wrong if they were
        // ever conflated: normalising by the scaled area gives mask/max(tol,mask), i.e. 1 wherever the
        // face is coupled at all, which erases every partial overlap.
        {
            if (g6.rawArea().empty())
            {
                std::printf("  FAIL vacuous: no raw areas were stashed, so nothing was scaled and the\n"
                            "       raw-vs-scaled distinction is untested here\n");
                ++failures;
            }
            scalar worstIfConflated = 0;
            for (const AMIInterface& A : a6)
            {
                if (f6[A.patch].name != "ACMI1_couple") continue;
                for (label i = 0; i < 4; ++i)
                {
                    const label f = f6[A.patch].start + i;
                    const scalar broken = A.mask[i]/std::max(ACMI_TOLERANCE, A.mask[i]);   // = 1 unless mask==0
                    worstIfConflated = std::fmax(worstIfConflated, std::fabs(broken - A.mask[i]));
                    if (A.mask[i] < 1 - 1e-12 && std::fabs(g6.rawMagSf(f) - g6.magSf()[f]) < 1e-15)
                    {
                        std::printf("  FAIL face %d has mask %.4f but its raw and scaled areas are equal --\n"
                                    "       the scaling did not touch it\n", (int)i, A.mask[i]);
                        ++failures;
                    }
                }
            }
            std::printf("  raw/scaled: %zu faces stashed; conflating them would corrupt the mask by %.4f\n",
                        g6.rawArea().size(), worstIfConflated);
            if (worstIfConflated < 0.4)
            {
                std::printf("  FAIL vacuous: conflating raw and scaled areas would barely change the mask on\n"
                            "       this fixture, so it cannot demonstrate why the distinction exists\n");
                ++failures;
            }
        }
    }

    std::printf("test_acmi_area_scaling: %d failures\n", failures);
    return failures ? 1 : 0;
}
