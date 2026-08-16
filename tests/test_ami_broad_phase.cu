// The AMI broad phase: a spatial search that must never exclude a pair the narrow phase would have
// found overlapping.
//
// The sweep it replaces tested every source face against every target face. On pimpleFoam/RAS/propeller
// -- 18496 against 18720 faces -- that is 3.5e8 polygon-overlap tests per direction, rebuilt every
// moving step, and the case could not reach its first time step in ten minutes. OpenFOAM walks an
// advancing front instead, which is O(n); a uniform grid is the same asymptotic win in a form that is
// parallel and ports to the device unchanged (flat arrays, independent per-source-face queries).
//
// A BROAD PHASE HAS EXACTLY ONE CORRECTNESS OBLIGATION: never reject a pair that really overlaps.
// Returning extra candidates only costs time, because the narrow phase clips them and finds zero. So
// every leg here is about that asymmetry, and Leg 3 is the one that caught a real defect.
//
// LEG 3 IS THE LESSON. A bare AABB test looks obviously correct and is not. The narrow phase projects
// each pair onto the plane perpendicular to their pair normal, which DISCARDS separation along that
// normal -- the same property that lets two box faces a box-length apart project onto each other
// perfectly. Two faces on a CURVED interface at slightly different radii therefore overlap in
// projection while their 3-D boxes do not touch. On RAS/rotatingFanInRoom (a cylindrical AMI) the
// un-padded grid dropped mean face coverage to 0.5512, with 3312 of 6984 source faces under 99%. The
// coverage refusal caught it rather than a wrong answer -- luck, not design, which is why it is pinned
// here instead.
#include "interface/ami_interface.cuh"
#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;
using brae::ami_detail::FaceGrid;

namespace {
int failures = 0;

void check(bool ok, const char* what)
{
    if (!ok) { std::printf("  FAIL: %s\n", what); ++failures; }
    else       std::printf("  ok:   %s\n", what);
}

// A quad in the z = zPlane surface, centred on (x, y), of side h.
std::vector<vector> quad(scalar x, scalar y, scalar zPlane, scalar h)
{
    const scalar a = 0.5*h;
    return { {x-a, y-a, zPlane}, {x+a, y-a, zPlane}, {x+a, y+a, zPlane}, {x-a, y+a, zPlane} };
}

void boxOf(const std::vector<vector>& p, vector& lo, vector& hi)
{
    lo = vector{ 1e300, 1e300, 1e300};
    hi = vector{-1e300,-1e300,-1e300};
    for (const vector& v : p)
    {
        lo.x = std::min(lo.x, v.x); lo.y = std::min(lo.y, v.y); lo.z = std::min(lo.z, v.z);
        hi.x = std::max(hi.x, v.x); hi.y = std::max(hi.y, v.y); hi.z = std::max(hi.z, v.z);
    }
}
} // namespace

int main()
{
    std::printf("== AMI broad phase (uniform grid) ==\n");

    // A 40x40 target patch of unit-ish faces, and a source patch offset by half a face so every source
    // face genuinely overlaps four target faces -- the non-conformal situation an AMI exists for.
    const int N = 40;
    const scalar h = 1.0;
    std::vector<std::vector<vector>> tgt, src;
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < N; ++i) tgt.push_back(quad(i*h, j*h, 0.0, h));
    for (int j = 0; j < N - 1; ++j)
        for (int i = 0; i < N - 1; ++i) src.push_back(quad(i*h + 0.5*h, j*h + 0.5*h, 0.0, h));

    FaceGrid grid;
    grid.build(tgt);
    std::vector<label> cand;
    std::vector<char> seen(tgt.size(), 0);

    // ---- Leg 1: never misses a true overlap ------------------------------------------------------------
    // Exhaustive reference: every target whose (padded) box meets the source box. The grid must return a
    // SUPERSET of the faces that actually share area.
    {
        std::size_t worstMissing = 0, totalCand = 0;
        for (const std::vector<vector>& sp : src)
        {
            vector a, b; boxOf(sp, a, b);
            grid.query(a, b, cand, seen);
            totalCand += cand.size();
            std::vector<char> got(tgt.size(), 0);
            for (const label j : cand) got[(std::size_t)j] = 1;
            std::size_t missing = 0;
            for (std::size_t j = 0; j < tgt.size(); ++j)
            {
                vector ta, tb; boxOf(tgt[j], ta, tb);
                const bool touches = !(tb.x < a.x || ta.x > b.x || tb.y < a.y || ta.y > b.y
                                    || tb.z < a.z || ta.z > b.z);
                if (touches && !got[j]) ++missing;
            }
            worstMissing = std::max(worstMissing, missing);
        }
        check(worstMissing == 0, "the grid returns every target face whose box meets the source box");
        check(totalCand > 0, "vacuity guard: candidates were actually produced");
        std::printf("        (mean candidates/source %.1f of %zu target faces)\n",
                    (double)totalCand/(double)src.size(), tgt.size());
    }

    // ---- Leg 2: it is actually a reduction ---------------------------------------------------------
    // The whole point. A "broad phase" that returns everything is correct and useless.
    {
        std::size_t total = 0;
        for (const std::vector<vector>& sp : src)
        {
            vector a, b; boxOf(sp, a, b);
            grid.query(a, b, cand, seen);
            total += cand.size();
        }
        const double mean = (double)total/(double)src.size();
        check(mean < 0.05*(double)tgt.size(),
              "candidates per source face are a small fraction of the target patch");
    }

    // ---- Leg 3: the padding, which a bare AABB test gets wrong --------------------------------------
    // Two co-located surfaces separated along their NORMAL by a fraction of a face. The narrow phase
    // projects that separation away and finds real overlap; an unpadded box test rejects the pair.
    {
        std::vector<std::vector<vector>> offsetTgt;
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < N; ++i) offsetTgt.push_back(quad(i*h, j*h, 0.30*h, h));   // 0.3 face out of plane
        FaceGrid padded;
        padded.build(offsetTgt);

        FaceGrid bare = padded;
        bare.pad = 0;                       // exactly the defect: box test with no tolerance

        std::size_t foundPadded = 0, foundBare = 0;
        std::vector<char> s2(offsetTgt.size(), 0);
        for (const std::vector<vector>& sp : src)
        {
            vector a, b; boxOf(sp, a, b);
            padded.query(a, b, cand, s2); foundPadded += cand.size();
            bare.query(a, b, cand, s2);    foundBare   += cand.size();
        }
        check(foundBare == 0, "vacuity guard: an UNPADDED box test finds nothing across a normal offset");
        check(foundPadded > 0, "padding recovers the pairs a curved interface needs");
        std::printf("        (normal offset 0.3 face: unpadded %zu candidates, padded %zu)\n",
                    foundBare, foundPadded);
    }

    // ---- Leg 4: candidates are de-duplicated -------------------------------------------------------
    // A face spanning several grid cells is listed in each of them; returning it twice would double its
    // overlap contribution and silently inflate the weights.
    {
        vector a{0.0, 0.0, -0.1}, b{N*h, N*h, 0.1};        // one query covering the whole patch
        grid.query(a, b, cand, seen);
        std::vector<char> once(tgt.size(), 0);
        bool dup = false;
        for (const label j : cand) { if (once[(std::size_t)j]) dup = true; once[(std::size_t)j] = 1; }
        check(!dup, "no candidate is returned twice, however many cells its box spans");
        check(cand.size() == tgt.size(), "...and a patch-wide query returns every face exactly once");
    }

    std::printf(failures ? "== FAILED (%d) ==\n" : "== PASSED ==\n", failures);
    return failures ? 1 : 0;
}
