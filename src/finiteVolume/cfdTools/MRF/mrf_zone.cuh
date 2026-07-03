#pragma once
// brae::MRFZone, Multiple Reference Frame for a rotating cell zone. Mirrors OpenFOAM MRFZone:
//   Omega        = omega * axis                                    (angular velocity vector)
//   addCoriolis  : UEqn.source[c] -= V[c] * (Omega x U[c])         (zone cells)
//   makeRelative : phi[f]        -= (Omega x (Cf - origin)) & Sf   (zone faces; makeAbsolute adds it)
// The relative (frame) flux is subtracted on the zone's internal faces and its in-zone boundary faces,
// so the momentum/pressure equations are solved in the rotating frame inside the zone.
#include "cf_types.cuh"
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "fvc.cuh"   // SurfaceScalarField
#include <vector>

namespace brae {

struct MRFZone {
    vector Omega{0, 0, 0};
    vector origin{0, 0, 0};
    std::vector<bool>  inZone;          // per-cell membership
    std::vector<label> cells;           // zone cells (cellZone)
    std::vector<label> internalFaces;   // internal faces with BOTH cells in the zone
};

inline MRFZone buildMRFZone(const PrimitiveMesh& m, const std::vector<label>& zoneCells,
                            const vector& axis, scalar omega, const vector& origin) {
    MRFZone z; z.Omega = omega * (axis / mag(axis)); z.origin = origin; z.cells = zoneCells;
    z.inZone.assign(m.nCells(), false);
    for (label c : zoneCells) z.inZone[c] = true;
    const std::vector<label>& own = m.owner(); const std::vector<label>& nei = m.neighbour();
    for (label f = 0; f < m.nInternalFaces(); ++f)
        if (z.inZone[own[f]] && z.inZone[nei[f]]) z.internalFaces.push_back(f);
    return z;
}

// phi -= sign * (Omega x (Cf - origin)) & Sf  over the zone faces. sign=+1 -> makeRelative, -1 -> makeAbsolute.
inline void mrfApplyFrameFlux(SurfaceScalarField& phi, const MRFZone& z, const FvGeometry& g,
                              const std::vector<FvPatch>& fvp, scalar sign) {
    for (label f : z.internalFaces)
        phi.internal[f] -= sign * dot(cross(z.Omega, g.Cf()[f] - z.origin), g.Sf()[f]);
    for (std::size_t pi = 0; pi < fvp.size(); ++pi)
        for (label i = 0; i < fvp[pi].size; ++i)
            if (z.inZone[fvp[pi].faceCells[i]]) {
                const label f = fvp[pi].start + i;
                phi.boundary[pi][i] -= sign * dot(cross(z.Omega, g.Cf()[f] - z.origin), g.Sf()[f]);
            }
}
inline void makeRelative(SurfaceScalarField& phi, const MRFZone& z, const FvGeometry& g, const std::vector<FvPatch>& fvp) { mrfApplyFrameFlux(phi, z, g, fvp, +1.0); }
inline void makeAbsolute(SurfaceScalarField& phi, const MRFZone& z, const FvGeometry& g, const std::vector<FvPatch>& fvp) { mrfApplyFrameFlux(phi, z, g, fvp, -1.0); }

// UEqn.source[c] -= V[c] * (Omega x U[c])  for zone cells (Coriolis added to the LHS momentum operator).
inline void addCoriolis(std::vector<vector>& Usource, const std::vector<vector>& U,
                        const std::vector<scalar>& V, const MRFZone& z) {
    for (label c : z.cells) Usource[c] -= V[c] * cross(z.Omega, U[c]);
}

} // namespace brae
