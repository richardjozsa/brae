#pragma once
// Device-resident forceCoeffs integration for the steady simpleFoam history. The reduction consumes the
// already-solved U/p state and returns only twelve reduced scalars; complete face fields never cross to the CPU.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
#include "device_cyclic.cuh"
#include "device_ami.cuh"
#include <vector>

namespace brae {

struct DeviceForceSelection
{
    int n = 0;
    DeviceBuffer<label> boundaryIndex;       // DeviceBoundary / DeviceMesh boundary order
    DeviceBuffer<scalar> cfx, cfy, cfz;      // selected face centres, for moments about CofR
};

struct DeviceForceResult
{
    vector pressure{0,0,0}, viscous{0,0,0};
    vector momentP{0,0,0}, momentV{0,0,0};
};

DeviceForceResult deviceWallForceReduce(
    const DeviceMesh& dm,
    const DeviceVectorBoundary& dbU,
    const DeviceBoundary& dbP,
    const DeviceBuffer<scalar>& Ux,
    const DeviceBuffer<scalar>& Uy,
    const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<scalar>& p,
    const DeviceBuffer<scalar>& nutBnd,
    const DeviceForceSelection& selection,
    scalar nu,
    scalar rhoRef,
    scalar pRef,
    const vector& CofR,
    DeviceCyclic* cyc = nullptr,
    DeviceAMI* ami = nullptr);

} // namespace brae
