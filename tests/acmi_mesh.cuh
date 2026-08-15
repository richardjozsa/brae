#pragma once
// Shared test fixture: a partially-overlapping cyclicACMI interface with a HAND-COMPUTABLE mask.
//
// Two 1 x NY x 1 blocks meet at x = 1, the second offset in y by `dy`, so the interface only partly
// overlaps -- the same situation as pimpleFoam/RAS/oscillatingInletACMI2D once its inlet channel has
// slid, but small enough that every overlap fraction can be worked out by hand:
//
//     src faces (y):  [0,.25]  [.25,.5]   [.5,.75]   [.75,1]
//     tgt faces (y):           [.375,.625]  [.625,.875]  [.875,1.125]  [1.125,1.375]
//     overlap:          none    .125/.25      full        full
//     mask:              0         0.5          1           1
//
// The 0.5 is the load-bearing entry: a fixture whose faces were all 0 or 1 would pass for an
// implementation that rounded the mask.
//
// The offset also arms the trap this whole feature turns on -- the two patch centroids differ by
// exactly `dy`, which is what brae's cyclic transform inference would subtract. For ACMI that inference
// is catastrophic (mask pinned at 1 forever, blockage wall never gets any area), so a fixture with
// aligned centroids would silently stop testing the thing that matters.
//
// withBlockage: emit the nonOverlapPatch walls as DUPLICATE faces of the coupled patches, which is how
// OF's blockMeshDict declares them (the same block face listed under both patches). Off by default,
// because a duplicated face is counted twice by the cell-volume pyramid sum until the ACMI area scaling
// splits it -- which is exactly what scalePatchFaceAreas exists to prevent, and what the area-scaling
// test asserts.
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {
namespace acmitest {

constexpr label  ACMI_NY = 4;        // faces along the interface, per side
constexpr scalar ACMI_DY = 0.375;    // block-B offset: deliberately NOT a multiple of the face height
constexpr scalar ACMI_CELL_VOL = 0.25*0.1;   // 1 (x) * 0.25 (y) * 0.1 (z)

// nonOverlap1: override the nonOverlapPatch NAME of ACMI1_couple. Used by the area-scaling test to
// point it at a patch that is the right size but the wrong surface, which must be refused.
// frontBackEmpty: put the z-normal faces on an `empty` patch instead of folding them into `walls`, as a
// real 2D case does. Off by default so every existing fixture is byte-for-byte unchanged. It matters for
// one question only: with them as walls, EVERY cell touches a wall, so a test asking "is this cell near a
// wall" can never see a cell whose only wall face is the ACMI blockage -- and that is exactly the cell
// the blockage gate exists for.
inline PrimitiveMesh twoBlockACMI(scalar dy, bool withBlockage = false,
                                  const char* nonOverlap1 = "ACMI1_blockage",
                                  bool frontBackEmpty = false)
{
    const label NY = ACMI_NY;
    std::vector<vector> pts;
    std::vector<label>  fv, foff, own, nei;
    std::vector<PatchInfo> patches;
    foff.push_back(0);

    auto addBlock = [&](scalar x0, scalar x1, scalar y0) -> label
    {
        const label base = static_cast<label>(pts.size());
        for (label k = 0; k <= 1; ++k)
            for (label j = 0; j <= NY; ++j)
                for (label i = 0; i <= 1; ++i)
                    pts.push_back(vector{i ? x1 : x0, y0 + scalar(j)/scalar(NY), scalar(k)*scalar(0.1)});
        return base;
    };
    const label bA = addBlock(0, 1, 0);
    const label bB = addBlock(1, 2, dy);

    auto quad = [&](label a, label b, label c, label d, label o, label n)
    {
        fv.push_back(a); fv.push_back(b); fv.push_back(c); fv.push_back(d);
        foff.push_back(static_cast<label>(fv.size()));
        own.push_back(o);
        if (n >= 0) nei.push_back(n);
    };
    auto pt = [&](label base, label i, label j, label k) { return base + i + 2*(j + (NY+1)*k); };

    // internal faces of BOTH blocks first (upper-triangular: owner ascending)
    for (int blk = 0; blk < 2; ++blk)
    {
        const label base = blk ? bB : bA;
        const label c0   = blk ? NY : 0;
        for (label j = 0; j + 1 < NY; ++j)
            quad(pt(base,0,j+1,0), pt(base,0,j+1,1), pt(base,1,j+1,1), pt(base,1,j+1,0), c0+j, c0+j+1);
    }

    auto beginPatch = [&](const char* nm, const char* ty)
    {
        PatchInfo pi; pi.name = nm; pi.type = ty;
        pi.start = static_cast<label>(own.size());
        patches.push_back(pi);
    };
    auto endPatch = [&]() { patches.back().size = static_cast<label>(own.size()) - patches.back().start; };

    // The coupled face of block A (x = 1, outward +x) and of block B (x = 1, outward -x).
    auto couplA = [&](label j) { quad(pt(bA,1,j,0), pt(bA,1,j+1,0), pt(bA,1,j+1,1), pt(bA,1,j,1), j, -1); };
    auto couplB = [&](label j) { quad(pt(bB,0,j,1), pt(bB,0,j+1,1), pt(bB,0,j+1,0), pt(bB,0,j,0), NY+j, -1); };

    beginPatch("inlet", "patch");
    for (label j = 0; j < NY; ++j)
        quad(pt(bA,0,j,1), pt(bA,0,j+1,1), pt(bA,0,j+1,0), pt(bA,0,j,0), j, -1);
    endPatch();

    beginPatch("ACMI1_couple", "cyclicACMI");
    for (label j = 0; j < NY; ++j) couplA(j);
    endPatch();
    if (withBlockage)   // the SAME faces again, in the same order -- OF's coincident nonOverlapPatch
    {
        beginPatch("ACMI1_blockage", "wall");
        for (label j = 0; j < NY; ++j) couplA(j);
        endPatch();
    }

    beginPatch("ACMI2_couple", "cyclicACMI");
    for (label j = 0; j < NY; ++j) couplB(j);
    endPatch();
    if (withBlockage)
    {
        beginPatch("ACMI2_blockage", "wall");
        for (label j = 0; j < NY; ++j) couplB(j);
        endPatch();
    }

    beginPatch("outlet", "patch");
    for (label j = 0; j < NY; ++j)
        quad(pt(bB,1,j,0), pt(bB,1,j+1,0), pt(bB,1,j+1,1), pt(bB,1,j,1), NY+j, -1);
    endPatch();

    beginPatch("walls", "wall");
    for (int blk = 0; blk < 2; ++blk)
    {
        const label base = blk ? bB : bA;
        const label c0   = blk ? NY : 0;
        quad(pt(base,1,0,0), pt(base,1,0,1), pt(base,0,0,1), pt(base,0,0,0), c0, -1);
        quad(pt(base,0,NY,0), pt(base,0,NY,1), pt(base,1,NY,1), pt(base,1,NY,0), c0+NY-1, -1);
        if (!frontBackEmpty)
            for (label j = 0; j < NY; ++j)
            {
                quad(pt(base,0,j+1,0), pt(base,1,j+1,0), pt(base,1,j,0), pt(base,0,j,0), c0+j, -1);
                quad(pt(base,0,j,1), pt(base,1,j,1), pt(base,1,j+1,1), pt(base,0,j+1,1), c0+j, -1);
            }
    }
    endPatch();
    if (frontBackEmpty)
    {
        beginPatch("frontAndBack", "empty");
        for (int blk = 0; blk < 2; ++blk)
        {
            const label base = blk ? bB : bA;
            const label c0   = blk ? NY : 0;
            for (label j = 0; j < NY; ++j)
            {
                quad(pt(base,0,j+1,0), pt(base,1,j+1,0), pt(base,1,j,0), pt(base,0,j,0), c0+j, -1);
                quad(pt(base,0,j,1), pt(base,1,j,1), pt(base,1,j+1,1), pt(base,0,j+1,1), c0+j, -1);
            }
        }
        endPatch();
    }

    for (PatchInfo& p : patches)
    {
        if (p.name == "ACMI1_couple") { p.neighbourPatch = "ACMI2_couple"; p.nonOverlapPatch = nonOverlap1; }
        if (p.name == "ACMI2_couple") { p.neighbourPatch = "ACMI1_couple"; p.nonOverlapPatch = "ACMI2_blockage"; }
    }

    PrimitiveMesh m;
    m.assign(std::move(pts), std::move(fv), std::move(foff), std::move(own), std::move(nei),
             std::move(patches), 2*NY);
    return m;
}

}   // namespace acmitest
}   // namespace brae
