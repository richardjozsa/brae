#pragma once
// brae::nearWallDist, OpenFOAM turbulence.y(): the near-wall distance used by wall functions.
// Mirrors src/finiteVolume/fvMesh/wallDist/nearWallDist. For each wall-patch face with adjacent
// cell C, y = the smallest distance from C to the nearest point on any wall face that is a
// point-neighbour of the face within the SAME patch (the own face included). This differs from
// the per-face normal distance 1/deltaCoeffs at corners (e.g. the pitzDaily step), where a
// point-neighbour wall face is geometrically closer than the cell's own wall face.
//
// Distance to a face polygon follows OF face::nearestPoint: fan the polygon into triangles from
// its centre and take the min point-to-triangle distance. The triangle nearest point is exact
// (Ericson closest-point-on-triangle). The fan apex is OF's AREA-WEIGHTED centroid (face::centre),
// which matters on WARPED snappy faces (see pointToFaceDist), the earlier vertex-average apex gave
// too-small near-wall distances there, blowing up omegaWallFunction (omega_vis ~ 1/y^2).
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include <vector>
#include <unordered_map>

namespace brae {

constexpr scalar nwdGreat = 1.0e15;   // OF VGREAT-style sentinel for the running minimum

// Closest point on triangle (a,b,c) to p (Ericson, Real-Time Collision Detection 5.1.5).
inline vector closestPointOnTriangle(
    const vector& p,
    const vector& a,
    const vector& b,
    const vector& c)
{
    const vector ab = b - a, ac = c - a, ap = p - a;
    const scalar d1 = dot(ab, ap), d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return a;                              // vertex region a
    const vector bp = p - b;
    const scalar d3 = dot(ab, bp), d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return b;                               // vertex region b
    const scalar vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) return a + (d1 / (d1 - d3)) * ab;   // edge ab
    const vector cp = p - c;
    const scalar d5 = dot(ab, cp), d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return c;                               // vertex region c
    const scalar vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) return a + (d2 / (d2 - d6)) * ac;   // edge ac
    const scalar va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0)                       // edge bc
        return b + ((d4 - d3) / ((d4 - d3) + (d5 - d6))) * (c - b);
    const scalar denom = 1.0 / (va + vb + vc);                        // interior
    return a + (vb * denom) * ab + (vc * denom) * ac;
}

// Distance from p to face polygon (CSR verts [v0,v1)), via OF centre-fan triangulation.
// The fan apex MUST be OF face::centre() = the AREA-WEIGHTED centroid (face.C::centre, sumAc/(3.sumA)), NOT the
// vertex average. They coincide on planar faces but DIVERGE on WARPED snappy faces, where the vertex-average fan
// has a triangle that tilts toward p and reports a too-small distance, which on omegaWallFunction (omega_vis ~ 1/y^2)
// blows omega up ~7x. With the area-weighted centroid cf's near-wall y is byte-identical to OF's meshWave (verified
// on motorBike cells 167636/175393). (OF: meshShapes/face/face.C::centre, faceIntersection.C::nearestPointClassify.)
inline scalar pointToFaceDist(
    const vector& p,
    const std::vector<vector>& pts,
    const std::vector<label>& fv,
    label v0,
    label v1)
{
    const label n = v1 - v0;
    vector cp{0.0, 0.0, 0.0};                                         // vertex average (the centroid's seed)
    for (label j = v0; j < v1; ++j)
        cp += pts[fv[j]];
    cp = (1.0 / n) * cp;
    scalar sumA = 0.0;
    vector sumAc{0.0, 0.0, 0.0};                   // OF face::centre area-weighted centroid
    for (label j = 0; j < n; ++j)
    {
        const vector& a = pts[fv[v0 + j]];
        const vector& b = pts[fv[v0 + (j + 1) % n]];
        const scalar ta = mag(cross(a - cp, b - cp));                 // 2x sub-triangle area
        sumA += ta;
        sumAc += ta * (a + b + cp);                       // ta x (3x sub-triangle centroid)
    }
    const vector ctr = (sumA > 1.0e-30) ? sumAc / (3.0 * sumA) : cp;
    scalar best = nwdGreat;
    for (label j = 0; j < n; ++j)
    {
        const vector& a = pts[fv[v0 + j]];
        const vector& b = pts[fv[v0 + (j + 1) % n]];
        const scalar d = mag(p - closestPointOnTriangle(p, ctr, a, b));
        if (d < best) best = d;
    }
    return best;
}

// Per-patch near-wall distance y (size patches.size(); non-wall patches left empty).
inline std::vector<std::vector<scalar>> nearWallDist(
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& patches)
{
    const std::vector<vector>& pts = m.points();
    const std::vector<label>&  fv  = m.faceVerts();
    const std::vector<label>&  fo  = m.faceOffsets();
    const std::vector<vector>& C   = g.C();

    std::vector<std::vector<scalar>> y(patches.size());
    for (std::size_t pi = 0; pi < patches.size(); ++pi)
    {
        const FvPatch& wp = patches[pi];
        if (wp.type != "wall" || wp.size == 0) continue;

        // point -> local wall faces within this patch (global face index = wp.start + i).
        std::unordered_map<label, std::vector<label>> pointFaces;
        for (label i = 0; i < wp.size; ++i)
        {
            const label f = wp.start + i;
            for (label j = fo[f]; j < fo[f + 1]; ++j)
                pointFaces[fv[j]].push_back(i);
        }

        y[pi].assign(wp.size, nwdGreat);
        for (label i = 0; i < wp.size; ++i)
        {
            const vector& Cc = C[wp.faceCells[i]];
            const label f = wp.start + i;
            // candidate faces: this face + every wall face in the patch sharing a vertex.
            std::vector<label> cand{i};
            for (label j = fo[f]; j < fo[f + 1]; ++j)
                for (label nf : pointFaces[fv[j]])
                    cand.push_back(nf);
            scalar best = nwdGreat;
            for (label nf : cand)
            {
                const label gf = wp.start + nf;
                const scalar d = pointToFaceDist(Cc, pts, fv, fo[gf], fo[gf + 1]);
                if (d < best) best = d;
            }
            y[pi][i] = best;
        }
    }
    return y;
}

} // namespace brae
