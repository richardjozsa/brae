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
#include <algorithm>
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

} // namespace brae
