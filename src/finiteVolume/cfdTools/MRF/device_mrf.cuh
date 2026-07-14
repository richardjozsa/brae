#pragma once
// Device MRF, Coriolis momentum source + relative flux (makeRelative/makeAbsolute) for a rotating cell zone.
// Device mirror of brae::MRFZone (host, validated by ctest mrf_rotation). The frame flux per face,
// (Omega x (Cf-origin)) & Sf, is STATIC geometry x Omega so it is precomputed once and zero outside the zone:
//   makeRelative : phi[f] -= frameFlux[f]   (zone internal faces + in-zone boundary faces)
//   Coriolis     : src_kk[c] -= V[c]*(Omega x U)_kk   (zone cells; explicit, feeds predictor + H()/HbyA)
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "mrf_zone.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "primitive_mesh.cuh"
#include <vector>

namespace brae {

struct DeviceMRF
{
    vector Omega{0, 0, 0};
    DeviceBuffer<label>  zoneCell;       // per-cell membership (1 in zone, 0 else)
    DeviceBuffer<scalar> frameFluxInt;   // (Omega x (Cf-origin)) & Sf per internal face (0 off-zone)
    DeviceBuffer<scalar> frameFluxBnd;   // per boundary face (0 off-zone)
    bool active = false;
};

// Build device MRF from the host MRFZone (precomputes the per-face frame flux + the cell mask). nonRotating
// patches (stationary in the absolute frame) are excluded from the boundary frame flux (OF MRFZone excludedFaces).
inline DeviceMRF buildDeviceMRF(
    const MRFZone& z,
    const PrimitiveMesh& m,
    const FvGeometry& g,
    const std::vector<FvPatch>& fvp,
    const std::vector<std::string>& nonRotating = {})
{
    DeviceMRF d;
    d.Omega = z.Omega;
    d.active = true;
    const label nC = m.nCells(), nIf = m.nInternalFaces();
    std::vector<label> zc(nC, 0);
    for (label c : z.cells)
        zc[c] = 1;
    d.zoneCell.copyFrom(zc);
    std::vector<scalar> ffi(nIf, 0.0);                    // internal: both cells in zone (= z.internalFaces)
    for (label f : z.internalFaces)
        ffi[f] = dot(cross(z.Omega, g.Cf()[f] - z.origin), g.Sf()[f]);
    d.frameFluxInt.copyFrom(ffi);
    std::vector<scalar> ffb;                              // boundary: faceCell in zone, patch not nonRotating
    for (const FvPatch& p : fvp)
    {
        bool nonRot = false;
        for (const auto& nm : nonRotating)
            if (p.name == nm)
            {
                nonRot = true;
                break;
            }
        for (label i = 0; i < p.size; ++i)
        {
            const label f = p.start + i;
            ffb.push_back((!nonRot && z.inZone[p.faceCells[i]]) ? dot(cross(z.Omega, g.Cf()[f] - z.origin), g.Sf()[f]) : 0.0);
        }
    }
    d.frameFluxBnd.copyFrom(ffb);
    return d;
}

// src[c] -= V[c]*(Omega x U)_kk for zone cells (kk = 0,1,2).  (Adds to the per-component momentum source.)
void deviceMrfCoriolis(const DeviceMRF& mrf, const DeviceBuffer<scalar>& V,
                       const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
                       int kk, DeviceBuffer<scalar>& src);

// phiInt -= sign*frameFluxInt; phiBnd -= sign*frameFluxBnd. sign=+1 makeRelative, -1 makeAbsolute.
void deviceMrfApplyFrameFlux(const DeviceMRF& mrf, scalar sign, DeviceBuffer<scalar>& phiInt, DeviceBuffer<scalar>& phiBnd);

} // namespace brae
