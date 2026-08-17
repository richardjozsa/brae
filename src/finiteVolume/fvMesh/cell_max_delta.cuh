#pragma once
// brae::cellMaxDeltaXYZ, OpenFOAM LES delta "maxDeltaxyz": the per-cell max edge length h_max = max(dx, dy, dz), where
// (dx, dy, dz) is the extent of the cell's vertex bounding box along the coordinate axes. Used as the IDDES filter
// width h_max (SA-IDDES delta = min(max(Cw*y, Cw*h_max), h_max)). Static geometry -> computed ONCE at setup and
// uploaded as a DeviceBuffer<scalar>, exactly like cellWallDist -> y_.
//
// A cell's vertices are gathered from its faces (each face contributes its polygon vertices to the owner cell, and to
// the neighbour cell for internal faces) via the mesh face->vertex CSR (points()/faceVert()/faceSize()). Single pass
// over the faces; no cell->point connectivity is needed.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include <algorithm>
#include <cmath>
#include <vector>

namespace brae {

inline std::vector<scalar> cellMaxDeltaXYZ(const PrimitiveMesh& m)
{
    const label nCells = m.nCells();
    const label nFaces = m.nFaces();
    const label nIntF  = m.nInternalFaces();
    const std::vector<vector>& pts = m.points();
    const std::vector<label>&  own = m.owner();
    const std::vector<label>&  nei = m.neighbour();

    constexpr scalar BIG = 1e300;
    std::vector<vector> lo(nCells, vector{ BIG,  BIG,  BIG});
    std::vector<vector> hi(nCells, vector{-BIG, -BIG, -BIG});

    auto accumulate = [&](label cell, label f)
    {
        const label nv = m.faceSize(f);
        for (label k = 0; k < nv; ++k)
        {
            const vector& p = pts[m.faceVert(f, k)];
            lo[cell].x = std::min(lo[cell].x, p.x); hi[cell].x = std::max(hi[cell].x, p.x);
            lo[cell].y = std::min(lo[cell].y, p.y); hi[cell].y = std::max(hi[cell].y, p.y);
            lo[cell].z = std::min(lo[cell].z, p.z); hi[cell].z = std::max(hi[cell].z, p.z);
        }
    };
    for (label f = 0; f < nFaces; ++f) accumulate(own[f], f);   // owner is defined for every face
    for (label f = 0; f < nIntF;  ++f) accumulate(nei[f], f);   // neighbour only for internal faces

    std::vector<scalar> hmax(nCells);
    for (label c = 0; c < nCells; ++c)
        hmax[c] = std::max(hi[c].x - lo[c].x, std::max(hi[c].y - lo[c].y, hi[c].z - lo[c].z));
    return hmax;
}

// Per-cell wall-normal grid spacing h_wn, the third term of the IDDES delta min(max(max(Cw*y, Cw*hmax), h_wn), hmax).
// The wall-normal direction n = normalize(C - wallOrigin), where wallOrigin is the nearest wall-face centre propagated
// by cellWallDist (exact wall-normal for near-wall cells, where h_wn matters). h_wn is the extent of the cell's vertex
// bounding box projected onto n. Cells with no wall direction (wallOrigin == C, i.e. wave never reached / no walls)
// fall back to hmax so h_wn never constrains the delta there. Static geometry -> computed ONCE at setup, uploaded.
inline std::vector<scalar> cellWallNormalSpacing(
    const PrimitiveMesh& m, const FvGeometry& g,
    const std::vector<vector>& wallOrigin, const std::vector<scalar>& hmax)
{
    const label nCells = m.nCells();
    const label nFaces = m.nFaces();
    const label nIntF  = m.nInternalFaces();
    const std::vector<vector>& pts = m.points();
    const std::vector<vector>& C   = g.C();
    const std::vector<label>&  own = m.owner();
    const std::vector<label>&  nei = m.neighbour();

    std::vector<vector> nhat(nCells);
    std::vector<char>   ok(nCells, 0);
    for (label c = 0; c < nCells; ++c)
    {
        const scalar dx = C[c].x - wallOrigin[c].x, dy = C[c].y - wallOrigin[c].y, dz = C[c].z - wallOrigin[c].z;
        const scalar mg = std::sqrt(dx*dx + dy*dy + dz*dz);
        if (mg > 1e-12) { nhat[c] = vector{dx/mg, dy/mg, dz/mg}; ok[c] = 1; }
    }
    constexpr scalar BIG = 1e300;
    std::vector<scalar> plo(nCells, BIG), phi(nCells, -BIG);
    auto accumulate = [&](label c, label f)
    {
        if (!ok[c]) return;
        const label nv = m.faceSize(f);
        for (label k = 0; k < nv; ++k)
        {
            const vector& p = pts[m.faceVert(f, k)];
            const scalar proj = nhat[c].x*p.x + nhat[c].y*p.y + nhat[c].z*p.z;
            plo[c] = std::min(plo[c], proj); phi[c] = std::max(phi[c], proj);
        }
    };
    for (label f = 0; f < nFaces; ++f) accumulate(own[f], f);
    for (label f = 0; f < nIntF;  ++f) accumulate(nei[f], f);

    std::vector<scalar> hwn(nCells);
    for (label c = 0; c < nCells; ++c)
        hwn[c] = ok[c] ? (phi[c] - plo[c]) : hmax[c];   // no wall direction -> hmax (h_wn does not constrain the delta)
    return hwn;
}

// OF LESModels::maxDeltaxyz::calcDelta() -- the LES FILTER WIDTH a case selects with `delta maxDeltaxyz`:
//
//     hmax[c] = deltaCoeff * max over the cell's faces of |n_f . (Cf_f - C_c)|
//
// i.e. twice the largest distance from the cell centre to a face PLANE, with deltaCoeff defaulting to 2
// (maxDeltaxyz.C:118). On a Cartesian hex that is exactly max(dx, dy, dz) and agrees with the vertex
// bounding box above; on a curvilinear or skewed cell it does not, which is why this is a separate
// function rather than a rename. cellMaxDeltaXYZ stays the IDDES hmax it has always been.
inline std::vector<scalar> cellMaxDeltaFaceNormal(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    scalar deltaCoeff = 2.0)
{
    const label nCells = m.nCells();
    const label nFaces = m.nFaces();
    const label nIntF  = m.nInternalFaces();
    const std::vector<label>& own = m.owner();
    const std::vector<label>& nei = m.neighbour();
    const std::vector<vector>& C  = g.C();
    const std::vector<vector>& Cf = g.Cf();
    const std::vector<vector>& Sf = g.Sf();
    const std::vector<scalar>& mag = g.magSf();

    std::vector<scalar> h(nCells, 0.0);
    auto take = [&](label c, label f)
    {
        if (c < 0 || c >= nCells) return;
        const scalar a = mag[f];
        if (!(a > scalar(0))) return;                       // an uncovered ACMI face has no normal to speak of
        const vector n{Sf[f].x/a, Sf[f].y/a, Sf[f].z/a};
        const vector d{Cf[f].x - C[c].x, Cf[f].y - C[c].y, Cf[f].z - C[c].z};
        const scalar t = std::fabs(n.x*d.x + n.y*d.y + n.z*d.z);
        if (t > h[c]) h[c] = t;
    };
    for (label f = 0; f < nFaces; ++f) take(own[f], f);
    for (label f = 0; f < nIntF;  ++f) take(nei[f], f);
    for (label c = 0; c < nCells; ++c) h[c] *= deltaCoeff;
    return h;
}

} // namespace brae
