// wallDist `method exactDistance`: the TRUE distance to the wall surface.
//
// OF's default is meshWave, a FaceCellWave that propagates a wall-face CENTRE cell-by-cell through the
// mesh graph. Its answer is the distance to the nearest PROPAGATED face centre, which is an
// over-estimate away from the wall -- deliberately so, and OF's own comment says as much. `exactDistance`
// instead finds the nearest point on the wall SURFACE (OF triangulates the patches and queries an octree;
// brae measures to the face polygon with the same pointToFaceDist that nearWallDist uses -- identical for
// a planar face, since any correct triangulation of a planar polygon covers the same points).
//
// On a box with a flat wall the exact answer is analytic: y = the perpendicular distance to the wall
// plane. Leg 1 asserts that to machine precision, over every cell, which is the whole claim.
//
// Leg 2 is what makes leg 1 worth asserting: meshWave on the SAME mesh must give something DIFFERENT and
// LARGER far from the wall. Without it, a broken exactDistance that silently fell back to meshWave would
// pass leg 1 on any mesh where the two happen to agree.
#include "box_mesh.cuh"
#include "cell_wall_dist.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

namespace {
int failures = 0;
}

int main()
{
    // A tall box: cells are unit-spaced, so with only the z faces as walls the exact answer for a cell at
    // height z is min(z, Nz - z) -- a clean analytic reference.
    const label N = 6, NZ = 8;
    PrimitiveMesh m = boxtest::boxMesh(N, N, NZ);
    FvGeometry g;
    g.build(m);
    std::vector<FvPatch> fvp = buildPatches(m, g);

    // Keep ONLY the two z-normal patches as walls; the rest become plain patches, so the reference stays
    // one-dimensional and hand-checkable.
    std::size_t nWallPatches = 0;
    for (FvPatch& p : fvp)
    {
        const bool isZ = (p.name.find("Zmin") != std::string::npos || p.name.find("Zmax") != std::string::npos);
        if (p.type == "wall" && !isZ) p.type = "patch";
        if (p.type == "wall") ++nWallPatches;
    }
    std::printf("  fixture: %d cells, %zu wall patches (z-normal only)\n", (int)m.nCells(), nWallPatches);
    if (nWallPatches != 2)
    { std::printf("  FAIL vacuous: expected exactly the two z walls\n"); ++failures; }

    const std::vector<scalar> ex = exactCellWallDist(m, g, fvp);
    const std::vector<scalar> mw = cellWallDist(m, g, fvp, nullptr);

    // ---- 1. exact == the analytic perpendicular distance ----
    {
        scalar worst = 0;
        int worstCell = -1;
        for (label c = 0; c < m.nCells(); ++c)
        {
            const scalar z = g.C()[c].z;
            const scalar want = std::fmin(z, scalar(NZ) - z);
            const scalar d = std::fabs(ex[c] - want);
            if (d > worst) { worst = d; worstCell = (int)c; }
        }
        std::printf("  exact vs analytic min(z, %d-z): max|diff| = %.3e", NZ, (double)worst);
        if (worstCell >= 0) std::printf("  (worst at z = %.3f)", (double)g.C()[worstCell].z);
        std::printf("\n");
        if (worst > 1e-12)
        {
            std::printf("  FAIL exactDistance is not the perpendicular distance to the wall plane\n");
            ++failures;
        }
    }

    // ---- 2. meshWave must DIFFER, and over-estimate, where the wall ENDS ----
    // On an orthogonal box with a full-plane wall the two agree exactly, and that is not a bug: the
    // nearest propagated face centre sits directly beneath the cell, so the wave's answer IS the
    // perpendicular distance. Shearing the mesh does not help either -- an axis-aligned wall stays
    // aligned with the cell columns. The over-estimate appears where the wall STOPS: past its edge the
    // true nearest point is on that edge, while the wave can only offer a face CENTRE half a cell
    // further on. So leg 2 keeps only HALF of one wall patch as wall and opens the rest.
    // (The first draft of this leg turned every wall off by mistake and then "passed" on the 1e15
    // no-wall sentinel, which is why the wall count is asserted below.)
    {
        std::vector<FvPatch> half;
        std::size_t nw = 0;
        for (const FvPatch& p : fvp)
        {
            if (p.type != "wall" || p.name.find("Zmin") == std::string::npos) { half.push_back(p); continue; }
            const label h = p.size / 2;
            FvPatch a = p, b = p;
            a.type = "wall";  a.size = h;
            a.faceCells.resize(h); a.deltaCoeffs.resize(h); a.nf.resize(h); a.magSf.resize(h); a.Cf.resize(h);
            b.type = "patch"; b.name = p.name + "_open"; b.start = p.start + h; b.size = p.size - h;
            b.faceCells.assign(p.faceCells.begin()+h, p.faceCells.end());
            b.deltaCoeffs.assign(p.deltaCoeffs.begin()+h, p.deltaCoeffs.end());
            b.nf.assign(p.nf.begin()+h, p.nf.end());
            b.magSf.assign(p.magSf.begin()+h, p.magSf.end());
            b.Cf.assign(p.Cf.begin()+h, p.Cf.end());
            half.push_back(a); half.push_back(b);
        }
        for (const FvPatch& p : half) if (p.type == "wall") ++nw;
        std::printf("  half-wall fixture: %zu wall patches\n", nw);
        if (!nw)
        { std::printf("  FAIL vacuous: no walls left, so both methods return their no-wall sentinel\n"); ++failures; }

        const std::vector<scalar> exh = exactCellWallDist(m, g, half);
        const std::vector<scalar> mwh = cellWallDist(m, g, half, nullptr);
        scalar biggest = 0;
        std::size_t nOver = 0, nUnder = 0;
        for (label c = 0; c < m.nCells(); ++c)
        {
            if (exh[c] >= nwdGreat*scalar(0.5) || mwh[c] >= nwdGreat*scalar(0.5)) continue;   // unreached
            const scalar d = mwh[c] - exh[c];
            if (d >  1e-9) ++nOver;
            if (d < -1e-9) ++nUnder;
            biggest = std::fmax(biggest, std::fabs(d));
        }
        std::printf("  meshWave vs exact: %zu cells larger, %zu smaller, max|diff| = %.3e\n",
                    nOver, nUnder, (double)biggest);
        if (biggest <= 1e-9)
        {
            std::printf("  FAIL vacuous: the two agree everywhere even past the wall's edge, so leg 1 would\n"
                        "       pass for an implementation that just ran meshWave\n");
            ++failures;
        }
        if (nUnder)
        {
            std::printf("  FAIL meshWave came out SHORTER than the exact distance somewhere. The wave\n"
                        "       propagates real wall-face centres, so it can only ever over-estimate.\n");
            ++failures;
        }
    }

    // ---- 3. no walls at all -> zeros, not a crash or a GREAT sentinel ----
    {
        std::vector<FvPatch> none = fvp;
        for (FvPatch& p : none) if (p.type == "wall") p.type = "patch";
        const std::vector<scalar> z = exactCellWallDist(m, g, none);
        scalar mx = 0;
        for (scalar v : z) mx = std::fmax(mx, std::fabs(v));
        std::printf("  no wall patches: max y = %.3e\n", (double)mx);
        if (mx != scalar(0))
        { std::printf("  FAIL a mesh with no walls must leave y at 0, not a sentinel\n"); ++failures; }
    }

    std::printf("exact_wall_dist: %d failures\n", failures);
    return failures ? 1 : 0;
}
