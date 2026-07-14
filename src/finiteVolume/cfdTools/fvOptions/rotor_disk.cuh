#pragma once
// cf rotorDiskSource (Froude blade-element momentum), scoped to OF's common path: geometryMode specified, fixedTrim
// (theta0 + cyclic), no coning (Rcone = I), a single lookup profileModel + a bladeModel table, inletFlowType local|fixed.
// Per disk cell each iteration (OF rotorDiskSource::calculate): in the cylindrical frame (e1 radial, e2 azimuthal,
// e3 axis) Uc=(e1.U, e2.U, e3.U); Uc.x=0; Uc.y = r*omega - Uc.y; alphaEff = (theta0+twist) - atan2(-Uc.z, Uc.y);
// (Cd,Cl)=lookup(alphaEff); f = 0.5*|Uc|^2*chord*nBlades*area/r/(2pi); localForce=(0,-f*Cd, tip*f*Cl);
// force = transform(Rcyl, localForce) = -f*Cd*e2 + tip*f*Cl*e3; eqn -= force -> cf relaxSrc -= force.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include <vector>

namespace brae {

struct DeviceRotorDisk
{
    bool   active = false;
    int    n = 0;                         // number of disk cells
    vector axis{0,1,0};                   // e3 (rotation axis, unit)
    scalar omega = 0, nBlades = 0, tipEffect = 1, rMax = 0, theta0 = 0;   // theta0 [rad] (collective)
    bool   localInflow = true;
    vector inletVel{0,0,0};   // inletFlowType local (use U[cell]) | fixed (inletVel)
    DeviceBuffer<label>  cells;           // disk cells (global indices)
    DeviceBuffer<scalar> e2x, e2y, e2z;   // azimuthal unit vector per cell
    DeviceBuffer<scalar> radius, area, twist, chord;   // per cell (twist [rad])
    // single lookup profile: alpha[rad] (sorted ascending), Cd, Cl
    DeviceBuffer<scalar> pAlpha, pCd, pCl;
    int nProf = 0;
};

// Build the device rotor from the dict params + the host mesh geometry (cell centres + face areas for the disk area).
struct RotorDiskParams                    // filled by readFvOptions
{
    bool   active = false;
    vector origin{0,0,0}, axis{0,1,0}, refDir{0,0,1};
    scalar omega = 0, tipEffect = 1, theta0 = 0;
    int nBlades = 0;
    bool   localInflow = true;
    vector inletVel{0,0,0};
    std::vector<label>  cells;
    std::vector<scalar> bladeR, bladeTwist, bladeChord;   // blade table (radius / twist[rad] / chord), sorted by radius
    std::vector<scalar> pAlpha, pCd, pCl;                 // profile table (alpha[rad] / Cd / Cl), sorted by alpha
};

// cellC = cell centres, Sf = face area vectors, owner/neighbour = internal face addressing (for the disk area).
DeviceRotorDisk buildDeviceRotorDisk(
    const RotorDiskParams& p,
    const std::vector<vector>& cellC,
    const std::vector<vector>& Sf,
    const std::vector<label>& owner,
    const std::vector<label>& neighbour,
    label nInternalFaces);

// Compute the per-cell rotor body force (3 component buffers, 0 outside the disk) from the current U; the solver then
// does relaxSrc[comp] -= force[comp] (OF eqn -= force). nCells = full field size.
void deviceRotorForce(
    const DeviceRotorDisk& rd,
    int nCells,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    DeviceBuffer<scalar>& fx,
    DeviceBuffer<scalar>& fy,
    DeviceBuffer<scalar>& fz);

} // namespace brae
