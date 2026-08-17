// nearWallDist searches ACROSS wall patches, not within one.
//
// OF v2412's nearWallDist has two branches, chosen by `cellDistFuncs::useCombinedWallPatch`, and that
// flag DEFAULTS TO TRUE: it gathers the faces of every wall patch into a single uindirectPrimitivePatch
// and takes point-neighbours in that combined set. brae implemented the other branch, the per-patch one,
// so a wall face whose nearest wall neighbour lived on a DIFFERENT patch reported its own, larger
// distance.
//
// A cyclicACMI produces that corner routinely: its non-overlap blockage is a wall patch coincident with
// the interface, so the cell at the END of the interface touches both the blockage and the ordinary side
// wall. Measured on pimpleFoam/RAS/oscillatingInletACMI2D, cell 3200 (first face of ACMI2_couple), with
// OF's own nearWallDist dumped for comparison:
//
//                          OF          brae (per-patch)
//     y(ACMI2_blockage)    5.20833e-03   1.25e-02      <- its own face, not the nearer walls face
//     y(walls)             5.20833e-03   5.20833e-03
//
// while cell 3280, one row in and touching only the blockage, agreed at 1.25e-02 in both. Through
// epsilon0 = invNw*Cmu^.75*k^1.5/(kappa*y) that made epsilon 0.71x OpenFOAM's at the four corner cells
// (0.90x at the other two) while the median interface cell was already right to 1.7e-07. Fixing it took
// the static turbulent case's epsilon from 1.92e-02 to 8.70e-05 -- a factor of 221 -- and k from
// 7.37e-04 to 1.03e-04.
//
// THE FIXTURE. Two wall patches meeting at a right angle on a box whose cells are DELIBERATELY
// ANISOTROPIC: with cubic cells both faces of a corner cell are equidistant and the per-patch and
// combined answers coincide, so the test would pass either way. Leg 2 is the guard that says so.
#include "box_mesh.cuh"
#include "near_wall_dist.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include <cmath>
#include <cstdio>
#include <vector>
#include <algorithm>

using namespace brae;

namespace {
int failures = 0;
}

int main()
{
    // A box stretched in x: dx = 1, dy = 4 (boxMesh is unit-spaced, so scale the points by hand).
    const label N = 4;
    PrimitiveMesh m = boxtest::boxMesh(N, N, 1);
    {
        std::vector<vector> p = m.points();
        for (vector& q : p) q.y *= 4.0;
        m.movePoints(p);
    }
    FvGeometry g;
    g.build(m);
    const std::vector<FvPatch> fvp = buildPatches(m, g);

    const std::vector<std::vector<scalar>> y = nearWallDist(m, g, fvp);

    // Find a cell that touches TWO different wall patches -- the corner this is all about.
    std::vector<std::vector<std::pair<std::size_t, label>>> byCell(m.nCells());
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
    {
        if (fvp[pi].type != "wall") continue;
        for (label i = 0; i < fvp[pi].size; ++i) byCell[fvp[pi].faceCells[i]].push_back({pi, i});
    }
    label corner = -1;
    for (label c = 0; c < m.nCells(); ++c)
    {
        if (byCell[c].size() < 2) continue;
        std::size_t p0 = byCell[c][0].first;
        for (const auto& e : byCell[c]) if (e.first != p0) { corner = c; break; }
        if (corner >= 0) break;
    }
    if (corner < 0)
    { std::printf("  FAIL vacuous: the fixture has no cell touching two different wall patches\n"); return 1; }

    // The distance from this cell's centre to each of its wall faces, straight from the geometry.
    scalar own = 0, other = 0;
    std::printf("  corner cell %d touches:\n", (int)corner);
    std::vector<scalar> perFace;
    for (const auto& e : byCell[corner])
    {
        const scalar d = std::fabs(dot(g.Cf()[fvp[e.first].start + e.second] - g.C()[corner],
                                       fvp[e.first].nf[e.second]));
        perFace.push_back(d);
        std::printf("    patch '%s' face %d: own normal distance %.6g, reported y %.6g\n",
                    fvp[e.first].name.c_str(), (int)e.second, (double)d, (double)y[e.first][e.second]);
    }
    own = *std::min_element(perFace.begin(), perFace.end());
    other = *std::max_element(perFace.begin(), perFace.end());

    // ---- 1. EVERY wall face of that cell must report the SMALLEST of them ----
    {
        scalar w = 0;
        for (const auto& e : byCell[corner]) w = std::fmax(w, std::fabs(y[e.first][e.second] - own));
        std::printf("  smallest wall distance %.6g; max|y - smallest| over its faces = %.3e\n",
                    (double)own, (double)w);
        if (w > 1e-12)
        {
            std::printf("  FAIL a wall face reported its own distance instead of the cell's nearest wall.\n"
                        "       OF's nearWallDist builds ONE patch from every wall patch and takes\n"
                        "       point-neighbours in it (cellDistFuncs::useCombinedWallPatch, default true)\n");
            ++failures;
        }
    }

    // ---- 2. the fixture must be able to tell the two answers apart ----
    {
        std::printf("  the cell's two wall distances are %.6g and %.6g\n", (double)own, (double)other);
        if (!(other > own*(1 + 1e-3)))
        {
            std::printf("  FAIL vacuous: this cell's wall faces are equidistant, so the per-patch answer and\n"
                        "       the combined one coincide and leg 1 would pass for either. Stretch the mesh.\n");
            ++failures;
        }
    }

    // ---- 3. a cell touching ONE wall patch keeps its own face distance ----
    {
        label plain = -1;
        for (label c = 0; c < m.nCells(); ++c)
            if (byCell[c].size() == 1) { plain = c; break; }
        if (plain < 0) std::printf("  (no single-wall-face cell in this fixture; leg skipped)\n");
        else
        {
            const auto& e = byCell[plain][0];
            const scalar d = std::fabs(dot(g.Cf()[fvp[e.first].start + e.second] - g.C()[plain],
                                           fvp[e.first].nf[e.second]));
            std::printf("  single-wall cell %d: y = %.6g, own face distance %.6g\n",
                        (int)plain, (double)y[e.first][e.second], (double)d);
            // Not required to equal it exactly -- a diagonal neighbour on the same patch can be nearer --
            // but it must not EXCEED it, since its own face is always a candidate.
            if (y[e.first][e.second] > d + 1e-12)
            {
                std::printf("  FAIL y exceeds the cell's own wall face distance, which is always a candidate\n");
                ++failures;
            }
        }
    }

    std::printf("near_wall_dist_combined: %d failures\n", failures);
    return failures ? 1 : 0;
}
