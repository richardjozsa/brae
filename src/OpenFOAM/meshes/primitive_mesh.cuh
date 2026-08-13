#pragma once
// brae::PrimitiveMesh, OpenFOAM polyMesh primitive topology.
// Mirrors src/OpenFOAM/meshes/polyMesh: points, faces (CSR), owner, neighbour, boundary
// patches. Internal faces occupy [0, nInternalFaces); boundary faces follow, grouped by
// patch. owner is non-decreasing (upper-triangular face ordering).
#include "cf_types.cuh"
#include "foam_token_reader.cuh"
#include <string>
#include <stdexcept>
#include <vector>

namespace brae {

struct PatchInfo
{
    std::string name;
    std::string type;
    label       start = 0;  // first face index (startFace)
    label       size  = 0;  // number of faces (nFaces)
    std::string neighbourPatch;  // cyclic: paired patch name
    // cyclicACMI: the patch (a wall) that takes the UNCOVERED area fraction. The two are
    // geometrically COINCIDENT -- OF's blockMeshDict gives them the same faces -- and the AMI
    // overlap fraction splits the area between them (cyclicACMIPolyPatch.C:226,252). Empty for
    // every other patch type.
    std::string nonOverlapPatch;
    std::vector<std::string> inGroups;  // polyBoundaryMesh groups this patch belongs to (boundaryField group match)
    std::string transform;       // cyclic: "translational" / "rotational" / "unknown"
    vector      rotationAxis{0, 0, 1};    // cyclic rotational: axis direction
    vector      rotationCentre{0, 0, 0};  // cyclic rotational: a point on the axis
};

class PrimitiveMesh
{
public:
    // Read points/faces/owner/neighbour/boundary from a constant/polyMesh directory.
    void read(const std::string& polyMeshDir);

    // Construct in memory (e.g. a decomposePar local mesh). Faces must be ordered internal-first
    // (upper-triangular by owner) then boundary faces grouped by patch, exactly as read().
    void assign(
        std::vector<vector> points,
        std::vector<label> faceVerts,
        std::vector<label> faceOffsets,
        std::vector<label> owner,
        std::vector<label> neighbour,
        std::vector<PatchInfo> patches,
        label nCells)
    {
        points_ = std::move(points);
        faceVerts_ = std::move(faceVerts);
        faceOffsets_ = std::move(faceOffsets);
        owner_ = std::move(owner);
        neighbour_ = std::move(neighbour);
        patches_ = std::move(patches);
        nCells_ = nCells;
    }

    label nPoints()        const { return static_cast<label>(points_.size()); }
    label nFaces()         const { return static_cast<label>(owner_.size()); }
    label nInternalFaces() const { return static_cast<label>(neighbour_.size()); }
    label nCells()         const { return nCells_; }

    const std::vector<vector>&    points()      const { return points_; }

    // OF polyMesh::movePoints(newPoints) -- replace the point positions, keeping the topology
    // (faces, owner, neighbour, patches) untouched. The CALLER must then rebuild FvGeometry, exactly
    // as OF's fvMesh::movePoints follows polyMesh::movePoints with updateGeomNotOldVol() and
    // surfaceInterpolation::clearOut(): every derived quantity -- Sf, magSf, V, C, Cf, weights,
    // deltaCoeffs -- is a function of the points and is stale the moment they move.
    //
    // `points0` (the ORIGINAL positions) must be kept by the caller: OF's solidBodyMotionSolver
    // transforms points0 by an absolute function of t (points0MotionSolver), never the current points.
    // Transforming the current points each step would compound round-off and drift.
    void movePoints(std::vector<vector> newPoints)
    {
        if (newPoints.size() != points_.size())
            throw std::runtime_error("brae: movePoints with a different point count -- topology change "
                                     "is not supported (OF polyMesh::movePoints keeps topology fixed).");
        points_ = std::move(newPoints);
    }
    const std::vector<label>&     faceVerts()   const { return faceVerts_; }
    const std::vector<label>&     faceOffsets() const { return faceOffsets_; }
    const std::vector<label>&     owner()       const { return owner_; }
    const std::vector<label>&     neighbour()   const { return neighbour_; }
    const std::vector<PatchInfo>& patches()     const { return patches_; }

    label faceSize(label f)  const { return faceOffsets_[f + 1] - faceOffsets_[f]; }
    label faceVert(label f, label k) const { return faceVerts_[faceOffsets_[f] + k]; }

private:
    void readPoints(const std::string& dir);
    void readFaces(const std::string& dir);
    void readOwner(const std::string& dir);
    void readNeighbour(const std::string& dir);
    void readBoundary(const std::string& dir);
    // Binary mesh cache (BRAE_MESH_CACHE): the ASCII polyMesh parse is the #1 startup cost (millions of string tokens).
    // On a warm run we reload the parsed topology from a binary blob (raw fread of the POD arrays), like OpenFOAM
    // reusing its decomposePar processor* dirs. Auto-invalidated when the polyMesh/owner file is newer than the cache.
    bool loadBinary(const std::string& path);
    void writeBinary(const std::string& path) const;

    std::vector<vector>    points_;
    std::vector<label>     faceVerts_;    // CSR values
    std::vector<label>     faceOffsets_;  // CSR offsets, size nFaces+1
    std::vector<label>     owner_;
    std::vector<label>     neighbour_;
    std::vector<PatchInfo> patches_;
    label                  nCells_ = 0;
};

} // namespace brae
