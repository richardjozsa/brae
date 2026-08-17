#pragma once
// brae::VolPointInterpolation -- OF volPointInterpolation (finiteVolume/interpolation/volPointInterpolation).
//
// Cell values -> POINT values, by inverse distance. Every point-based mesh-motion solver needs it: the
// motion equation is solved on CELLS and the mesh moves its POINTS.
//
// OF's rule has two cases, and the split is the whole of it (volPointInterpolation.C:169-256):
//
//   INTERIOR point (not on any real patch)
//       value = sum_c w_c * cellValue_c / sum_c w_c ,   w_c = 1/|p - C_c|   over the point's cells
//
//   PATCH point (on at least one non-empty, non-coupled patch)
//       value = sum_f w_f * faceValue_f / sum_f w_f ,   w_f = 1/|p - Cf_f|  over its PATCH faces only
//       -- the surrounding CELL values are not used at all.
//
// THE SECOND CASE IS NOT AN OPTIMISATION. On movingCone the piston face carries
// `pointMotionUx uniformFixedValue 1`, so its cell BC is a fixedValue 1; a point on that face must come
// out at exactly 1. Interpolating it from the adjacent CELLS instead gives whatever the Laplace solution
// happens to be just inside the wall -- always less than 1 -- so the piston would advance more slowly
// than the case prescribes, on a mesh that still looks perfectly well-formed.
//
// An `empty` patch is not a real patch here (a 2D case's front/back must not pin the motion), and
// neither is a coupled one (cyclic/AMI points are interior to the global mesh). Both are excluded from
// `patch faces`, exactly as OF's calcBoundaryAddressing does.
//
// The weights are pure geometry, so they are rebuilt whenever the points move -- which for a motion
// solver is every step. build() is O(nPoints) with small per-point lists; on movingCone's 4000-odd
// points that is not measurable next to the Laplace solve it feeds.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {

class VolPointInterpolation
{
public:
    // `fvp` supplies the patch TYPES: an `empty` patch, and any coupled one (cyclic/cyclicAMI/
    // cyclicACMI/processor), does not make its points into patch points.
    void build(const PrimitiveMesh& m, const FvGeometry& g, const std::vector<FvPatch>& fvp)
    {
        const label nPts   = m.nPoints();
        const label nCells = m.nCells();
        const label nFaces = m.nFaces();
        const label nIntF  = m.nInternalFaces();
        const std::vector<vector>& pts = m.points();
        const std::vector<vector>& C   = g.C();
        const std::vector<vector>& Cf  = g.Cf();

        // 1. which BOUNDARY faces are 'real' patch faces, and hence which points are patch points
        std::vector<char> isPatchFace(nFaces, 0);
        for (const FvPatch& p : fvp)
        {
            if (p.type == "empty" || isCoupledPatchType(p.type)) continue;
            for (label i = 0; i < p.size; ++i) isPatchFace[p.start + i] = 1;
        }
        isPatchPoint_.assign(nPts, 0);
        for (label f = nIntF; f < nFaces; ++f)
        {
            if (!isPatchFace[f]) continue;
            const label nv = m.faceSize(f);
            for (label k = 0; k < nv; ++k) isPatchPoint_[m.faceVert(f, k)] = 1;
        }

        // 2. point -> cells (interior points) and point -> patch faces (patch points), as CSR
        std::vector<label> cellCount(nPts + 1, 0), faceCount(nPts + 1, 0);
        auto forEachPointOfCellFaces = [&](auto&& fn)
        {
            for (label f = 0; f < nFaces; ++f)
            {
                const label nv = m.faceSize(f);
                for (label k = 0; k < nv; ++k) fn(f, m.faceVert(f, k));
            }
        };
        // A point's cells are the owners/neighbours of the faces it belongs to, DE-DUPLICATED: a cell
        // touches a point through several of its faces, and counting it once per face would weight that
        // cell by how many of its faces happen to meet there.
        std::vector<std::vector<label>> pc(nPts), pf(nPts);
        forEachPointOfCellFaces([&](label f, label pt)
        {
            if (!isPatchPoint_[pt])
            {
                pc[pt].push_back(m.owner()[f]);
                if (f < nIntF) pc[pt].push_back(m.neighbour()[f]);
            }
            else if (f >= nIntF && isPatchFace[f])
            {
                pf[pt].push_back(f);
            }
        });
        auto uniq = [](std::vector<label>& v)
        {
            std::sort(v.begin(), v.end());
            v.erase(std::unique(v.begin(), v.end()), v.end());
        };
        for (label p = 0; p < nPts; ++p) { uniq(pc[p]); uniq(pf[p]); }

        // 3. inverse-distance weights, normalised per point
        off_.assign(nPts + 1, 0);
        idx_.clear();
        w_.clear();
        for (label p = 0; p < nPts; ++p)
        {
            const std::vector<label>& src = isPatchPoint_[p] ? pf[p] : pc[p];
            scalar sum = 0;
            const std::size_t base = w_.size();
            for (const label s : src)
            {
                const vector& x = isPatchPoint_[p] ? Cf[s] : C[s];
                const vector d{pts[p].x - x.x, pts[p].y - x.y, pts[p].z - x.z};
                // A point sitting exactly on a cell centre or face centre has infinite weight. OF does
                // not guard it either (the geometry makes it impossible on a valid mesh); guard it here
                // rather than emit inf*value = nan and hunt for it downstream.
                const scalar dist = std::sqrt(d.x*d.x + d.y*d.y + d.z*d.z);
                const scalar wi = scalar(1)/std::max(dist, scalar(1e-300));
                idx_.push_back(isPatchPoint_[p] ? (s - nIntF) : s);   // patch faces indexed from 0
                w_.push_back(wi);
                sum += wi;
            }
            if (sum > scalar(0))
                for (std::size_t k = base; k < w_.size(); ++k) w_[k] /= sum;
            off_[p + 1] = static_cast<label>(w_.size());
        }
        (void)nCells;
        nBndFaces_ = nFaces - nIntF;
        nPoints_ = nPts;
    }

    // cellValues (nCells) + boundaryValues (flattened boundary faces, mesh order from nInternalFaces)
    // -> one value per mesh point. `boundaryValues` is only read at patch points.
    std::vector<scalar> interpolate(const std::vector<scalar>& cellValues,
                                    const std::vector<scalar>& boundaryValues) const
    {
        if (static_cast<label>(boundaryValues.size()) != nBndFaces_)
            throw std::runtime_error(
                "brae: volPointInterpolation needs one boundary value per boundary FACE (got "
                + std::to_string(boundaryValues.size()) + " for " + std::to_string(nBndFaces_) + ").");
        std::vector<scalar> out(static_cast<std::size_t>(nPoints_), scalar(0));
        for (label p = 0; p < nPoints_; ++p)
        {
            scalar v = 0;
            for (label k = off_[p]; k < off_[p + 1]; ++k)
                v += w_[k] * (isPatchPoint_[p] ? boundaryValues[idx_[k]] : cellValues[idx_[k]]);
            out[p] = v;
        }
        return out;
    }

    bool  isPatchPoint(label p) const { return isPatchPoint_[p] != 0; }
    label nPoints()             const { return nPoints_; }

private:
    static bool isCoupledPatchType(const std::string& t)
    {
        return t == "cyclic" || t == "cyclicAMI" || t == "cyclicACMI" || t == "processor";
    }

    std::vector<char>   isPatchPoint_;
    std::vector<label>  off_, idx_;
    std::vector<scalar> w_;
    label               nPoints_   = 0;
    label               nBndFaces_ = 0;
};

} // namespace brae
