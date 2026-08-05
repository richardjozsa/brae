#pragma once
// brae::FvPatch, per-boundary-patch finite-volume addressing/geometry. Mirrors OpenFOAM
// fvPatch: faceCells (owner cell of each boundary face) and boundary deltaCoeffs
// (1/|Cf - C_own|, basicFvGeometryScheme). Boundary face Sf/magSf/Cf are reached via the
// global face index (start + i) from FvGeometry.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include <string>
#include <vector>

namespace brae {

struct FvPatch {
    std::string         name;
    std::string         type;
    std::vector<std::string> inGroups;  // groups this patch belongs to (boundaryField group-keyword match)
    label               start = 0;
    label               size  = 0;
    std::vector<label>  faceCells;    // owner cell of each boundary face
    std::vector<scalar> deltaCoeffs;  // 1/|Cf - C[faceCell]|
    std::vector<vector> nf;           // unit face normal Sf/|Sf| (for slip/symmetry projection)
    std::vector<scalar> magSf;        // |Sf| face area (flowRateInletVelocity: gSum(rho*magSf))
    std::vector<vector> Cf;           // face centre (timeVaryingMapped boundaryData -> face mapping)
};

std::vector<FvPatch> buildPatches(const PrimitiveMesh& m, const FvGeometry& g);

} // namespace brae
