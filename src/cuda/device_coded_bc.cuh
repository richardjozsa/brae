#pragma once
// Runtime-compiled (NVRTC) device boundary conditions -- the GPU-resident analogue of OpenFOAM codedFixedValue. The
// user's per-face snippet is compiled to a CUDA kernel at SETUP (NVRTC -> PTX -> driver-API module load) and LAUNCHED
// over the coded patch's boundary faces each step, writing the boundary refValue directly on the device. The user code
// never runs on the host, so there is no device->host->device round-trip: it stays fully device-resident, matching brae's
// "host orchestrates, device computes" model (every existing per-step BC update is already a per-face kernel too).
//
// The generated kernel exposes, per boundary face f of the patch: fx,fy,fz (face centre), t (time), cx,cy,cz (the
// adjacent internal-cell value of U), and expects the snippet to set `out` (a vec3). See compileCodedVectorBc.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include <string>

namespace brae {

// A compiled coded-BC kernel handle. module/func are opaque CUmodule/CUfunction (kept void* so cuda.h stays out of
// headers). valid() == false means compilation was skipped/failed.
struct CodedBcKernel
{
    void*       module = nullptr;   // CUmodule
    void*       func   = nullptr;   // CUfunction
    std::string name;
    bool valid() const { return func != nullptr; }
};

// Compile `body` into a VECTOR-field boundary kernel `bc_<name>`. The generated kernel body is, per face f in [0,n):
//   int g = offset + f;  fx=CfX[g],fy=CfY[g],fz=CfZ[g];  t=<arg>;  cx=Ux[faceCell[g]],...;  vec3 out={0,0,0};
//   { <body> }           refX[g]=out.x; refY[g]=out.y; refZ[g]=out.z;
// `body` is device C++ (e.g. `out = make3(10.0*(1.0 - (fy/0.0254)*(fy/0.0254)), 0.0, 0.0);` for a parabolic inlet).
// Throws std::runtime_error with the NVRTC compile log on a bad snippet. Compiled ONCE; launch each step.
CodedBcKernel compileCodedVectorBc(const std::string& name, const std::string& body);

// Launch a compiled vector coded BC over `count` faces starting at flat boundary index `offset`, filling refX/refY/refZ
// (the boundary refValue components) from CfX/Y/Z (flat boundary face centres), the adjacent-cell U (Ux/Uy/Uz indexed by
// faceCell), and time t. All buffers are device-resident; no host transfer. No-op if !k.valid().
void launchCodedVectorBc(
    const CodedBcKernel& k, int offset, int count, scalar t,
    const DeviceBuffer<scalar>& CfX, const DeviceBuffer<scalar>& CfY, const DeviceBuffer<scalar>& CfZ,
    const DeviceBuffer<scalar>& Ux,  const DeviceBuffer<scalar>& Uy,  const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<label>&  faceCell,
    DeviceBuffer<scalar>& refX, DeviceBuffer<scalar>& refY, DeviceBuffer<scalar>& refZ);

// Compile `body` into a SCALAR-field boundary kernel `bc_<name>` (for p / k / omega / epsilon / nuTilda). Per face:
//   fx,fy,fz (face centre), t (time), cc (the adjacent internal-cell value of THIS field); set `out` (a double).
// e.g. `out = cc;` (zeroGradient-like) or `out = 1.5*(0.05*10.0*(fy>0?1:1))*...;` (a profiled inlet k).
CodedBcKernel compileCodedScalarBc(const std::string& name, const std::string& body);

// Launch a compiled scalar coded BC over `count` faces from flat boundary index `offset`, filling `ref` (the scalar
// boundary refValue) from CfX/Y/Z, the adjacent-cell own-field value fld[faceCell], and time t. Device-resident.
void launchCodedScalarBc(
    const CodedBcKernel& k, int offset, int count, scalar t,
    const DeviceBuffer<scalar>& CfX, const DeviceBuffer<scalar>& CfY, const DeviceBuffer<scalar>& CfZ,
    const DeviceBuffer<scalar>& fld, const DeviceBuffer<label>& faceCell,
    DeviceBuffer<scalar>& ref);

// codedMixed (Robin): the snippet sets `out` (refValue) AND `vf` (valueFraction: 1=fixedValue .. 0=zeroGradient). The
// kernel writes the boundary refValue AND valueFraction; the device mixed assembly blends value = vf*refValue + (1-vf)*internal.
CodedBcKernel compileCodedMixedVectorBc(const std::string& name, const std::string& body);
CodedBcKernel compileCodedMixedScalarBc(const std::string& name, const std::string& body);

void launchCodedMixedVectorBc(
    const CodedBcKernel& k, int offset, int count, scalar t,
    const DeviceBuffer<scalar>& CfX, const DeviceBuffer<scalar>& CfY, const DeviceBuffer<scalar>& CfZ,
    const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<label>& faceCell,
    DeviceBuffer<scalar>& refX, DeviceBuffer<scalar>& refY, DeviceBuffer<scalar>& refZ,
    DeviceBuffer<scalar>& vfX, DeviceBuffer<scalar>& vfY, DeviceBuffer<scalar>& vfZ);

void launchCodedMixedScalarBc(
    const CodedBcKernel& k, int offset, int count, scalar t,
    const DeviceBuffer<scalar>& CfX, const DeviceBuffer<scalar>& CfY, const DeviceBuffer<scalar>& CfZ,
    const DeviceBuffer<scalar>& fld, const DeviceBuffer<label>& faceCell,
    DeviceBuffer<scalar>& ref, DeviceBuffer<scalar>& vfrac);

} // namespace brae
