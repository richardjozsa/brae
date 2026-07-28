// NVRTC device-coded-BC implementation: compile a per-face CUDA snippet to a kernel at runtime (nvrtc -> PTX), load it
// with the CUDA driver API (cuModuleLoadData / cuModuleGetFunction), and launch it over the coded patch's boundary faces
// each step. Vector (U) and scalar (p/k/omega/...) variants share the compile+load path. See device_coded_bc.cuh.
#include "device_coded_bc.cuh"
#include <cuda_runtime.h>
#include <cuda.h>          // driver API: CUmodule/CUfunction, cuModuleLoadData, cuLaunchKernel
#include <nvrtc.h>         // runtime CUDA compiler
#include <stdexcept>
#include <string>
#include <vector>

namespace brae {

namespace {
inline void nvrtcCheck(nvrtcResult r, const char* what)
{
    if (r != NVRTC_SUCCESS) throw std::runtime_error(std::string("NVRTC ") + what + ": " + nvrtcGetErrorString(r));
}
inline void cuCheck(CUresult r, const char* what)
{
    if (r != CUDA_SUCCESS)
    {
        const char* msg = nullptr; cuGetErrorString(r, &msg);
        throw std::runtime_error(std::string("CUDA driver ") + what + ": " + (msg ? msg : "?"));
    }
}
constexpr int TPB = 256;
inline int nBlocks(int n) { return (n + TPB - 1) / TPB; }

// NVRTC-compile `src` to PTX for the running GPU's virtual arch, load it via the driver API, and fetch kernel `fn`.
CodedBcKernel compileSource(const std::string& src, const std::string& fn, const std::string& name)
{
    nvrtcProgram prog;
    nvrtcCheck(nvrtcCreateProgram(&prog, src.c_str(), "coded_bc.cu", 0, nullptr, nullptr), "createProgram");
    int dev = 0; cudaGetDevice(&dev);
    int major = 0, minor = 0;
    cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
    cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
    const std::string arch = "--gpu-architecture=compute_" + std::to_string(major) + std::to_string(minor);
    const char* opts[] = { arch.c_str(), "--std=c++17" };
    const nvrtcResult crc = nvrtcCompileProgram(prog, 2, opts);
    if (crc != NVRTC_SUCCESS)
    {
        std::size_t logSize = 0; nvrtcGetProgramLogSize(prog, &logSize);
        std::string log(logSize, '\0'); nvrtcGetProgramLog(prog, log.data());
        nvrtcDestroyProgram(&prog);
        throw std::runtime_error("brae codedFixedValue '" + name + "' failed to compile (NVRTC):\n" + log);
    }
    std::size_t ptxSize = 0; nvrtcCheck(nvrtcGetPTXSize(prog, &ptxSize), "getPTXSize");
    std::vector<char> ptx(ptxSize);
    nvrtcCheck(nvrtcGetPTX(prog, ptx.data()), "getPTX");
    nvrtcDestroyProgram(&prog);

    cuCheck(cuInit(0), "cuInit");
    cudaFree(nullptr);                                   // force the runtime primary context to exist
    CUmodule mod = nullptr;
    cuCheck(cuModuleLoadData(&mod, ptx.data()), "moduleLoadData");
    CUfunction func = nullptr;
    cuCheck(cuModuleGetFunction(&func, mod, fn.c_str()), "moduleGetFunction");

    CodedBcKernel k;
    k.module = mod; k.func = func; k.name = name;
    return k;
}

const char* VEC3_PREAMBLE =
    "struct vec3 { double x, y, z; };\n"
    "__device__ inline vec3 make3(double a, double b, double c){ vec3 v; v.x=a; v.y=b; v.z=c; return v; }\n";
} // namespace

CodedBcKernel compileCodedVectorBc(const std::string& name, const std::string& body)
{
    const std::string fn = "bc_" + name;
    const std::string src = std::string(VEC3_PREAMBLE) +
        "extern \"C\" __global__ void " + fn + "(\n"
        "    int n, int offset,\n"
        "    const double* CfX, const double* CfY, const double* CfZ, double t,\n"
        "    const double* Ux, const double* Uy, const double* Uz, const int* faceCell,\n"
        "    double* refX, double* refY, double* refZ)\n"
        "{\n"
        "    int tid = blockIdx.x*blockDim.x + threadIdx.x;\n"
        "    if (tid >= n) return;\n"
        "    int g = offset + tid;\n"
        "    const double fx = CfX[g], fy = CfY[g], fz = CfZ[g];\n"
        "    const int ic = faceCell[g];\n"
        "    const double cx = Ux[ic], cy = Uy[ic], cz = Uz[ic];\n"
        "    (void)fx;(void)fy;(void)fz;(void)cx;(void)cy;(void)cz;(void)t;\n"
        "    vec3 out; out.x = 0; out.y = 0; out.z = 0;\n"
        "    {\n" + body + "\n    }\n"
        "    refX[g] = out.x; refY[g] = out.y; refZ[g] = out.z;\n"
        "}\n";
    return compileSource(src, fn, name);
}

CodedBcKernel compileCodedScalarBc(const std::string& name, const std::string& body)
{
    const std::string fn = "bc_" + name;
    const std::string src = std::string(VEC3_PREAMBLE) +
        "extern \"C\" __global__ void " + fn + "(\n"
        "    int n, int offset,\n"
        "    const double* CfX, const double* CfY, const double* CfZ, double t,\n"
        "    const double* fld, const int* faceCell,\n"
        "    double* ref)\n"
        "{\n"
        "    int tid = blockIdx.x*blockDim.x + threadIdx.x;\n"
        "    if (tid >= n) return;\n"
        "    int g = offset + tid;\n"
        "    const double fx = CfX[g], fy = CfY[g], fz = CfZ[g];\n"
        "    const int ic = faceCell[g];\n"
        "    const double cc = fld[ic];\n"                // adjacent internal-cell value of THIS field
        "    (void)fx;(void)fy;(void)fz;(void)cc;(void)t;\n"
        "    double out = 0;\n"
        "    {\n" + body + "\n    }\n"
        "    ref[g] = out;\n"
        "}\n";
    return compileSource(src, fn, name);
}

void launchCodedVectorBc(
    const CodedBcKernel& k, int offset, int count, scalar t,
    const DeviceBuffer<scalar>& CfX, const DeviceBuffer<scalar>& CfY, const DeviceBuffer<scalar>& CfZ,
    const DeviceBuffer<scalar>& Ux,  const DeviceBuffer<scalar>& Uy,  const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<label>&  faceCell,
    DeviceBuffer<scalar>& refX, DeviceBuffer<scalar>& refY, DeviceBuffer<scalar>& refZ)
{
    if (!k.valid() || count <= 0) return;
    int n = count, off = offset;
    scalar tt = t;
    const scalar* cfx = CfX.data(); const scalar* cfy = CfY.data(); const scalar* cfz = CfZ.data();
    const scalar* ux = Ux.data();   const scalar* uy = Uy.data();   const scalar* uz = Uz.data();
    const label*  fc = faceCell.data();
    scalar* rx = refX.data(); scalar* ry = refY.data(); scalar* rz = refZ.data();
    void* args[] = { &n, &off, &cfx, &cfy, &cfz, &tt, &ux, &uy, &uz, &fc, &rx, &ry, &rz };
    cuCheck(cuLaunchKernel(static_cast<CUfunction>(k.func),
                           nBlocks(count), 1, 1, TPB, 1, 1, 0, CU_STREAM_PER_THREAD, args, nullptr),
            "launchKernel");
}

void launchCodedScalarBc(
    const CodedBcKernel& k, int offset, int count, scalar t,
    const DeviceBuffer<scalar>& CfX, const DeviceBuffer<scalar>& CfY, const DeviceBuffer<scalar>& CfZ,
    const DeviceBuffer<scalar>& fld, const DeviceBuffer<label>& faceCell,
    DeviceBuffer<scalar>& ref)
{
    if (!k.valid() || count <= 0) return;
    int n = count, off = offset;
    scalar tt = t;
    const scalar* cfx = CfX.data(); const scalar* cfy = CfY.data(); const scalar* cfz = CfZ.data();
    const scalar* f = fld.data();
    const label*  fc = faceCell.data();
    scalar* r = ref.data();
    void* args[] = { &n, &off, &cfx, &cfy, &cfz, &tt, &f, &fc, &r };
    cuCheck(cuLaunchKernel(static_cast<CUfunction>(k.func),
                           nBlocks(count), 1, 1, TPB, 1, 1, 0, CU_STREAM_PER_THREAD, args, nullptr),
            "launchKernel");
}

// ---- codedMixed (Robin): the snippet sets `out` (refValue) AND `vf` (valueFraction, 1=fixedValue .. 0=zeroGradient) ----

CodedBcKernel compileCodedMixedVectorBc(const std::string& name, const std::string& body)
{
    const std::string fn = "bc_" + name;
    const std::string src = std::string(VEC3_PREAMBLE) +
        "extern \"C\" __global__ void " + fn + "(\n"
        "    int n, int offset,\n"
        "    const double* CfX, const double* CfY, const double* CfZ, double t,\n"
        "    const double* Ux, const double* Uy, const double* Uz, const int* faceCell,\n"
        "    double* refX, double* refY, double* refZ, double* vfX, double* vfY, double* vfZ)\n"
        "{\n"
        "    int tid = blockIdx.x*blockDim.x + threadIdx.x;\n"
        "    if (tid >= n) return;\n"
        "    int g = offset + tid;\n"
        "    const double fx = CfX[g], fy = CfY[g], fz = CfZ[g];\n"
        "    const int ic = faceCell[g];\n"
        "    const double cx = Ux[ic], cy = Uy[ic], cz = Uz[ic];\n"
        "    (void)fx;(void)fy;(void)fz;(void)cx;(void)cy;(void)cz;(void)t;\n"
        "    vec3 out; out.x = 0; out.y = 0; out.z = 0; double vf = 1;\n"
        "    {\n" + body + "\n    }\n"
        "    refX[g]=out.x; refY[g]=out.y; refZ[g]=out.z; vfX[g]=vf; vfY[g]=vf; vfZ[g]=vf;\n"
        "}\n";
    return compileSource(src, fn, name);
}

CodedBcKernel compileCodedMixedScalarBc(const std::string& name, const std::string& body)
{
    const std::string fn = "bc_" + name;
    const std::string src = std::string(VEC3_PREAMBLE) +
        "extern \"C\" __global__ void " + fn + "(\n"
        "    int n, int offset,\n"
        "    const double* CfX, const double* CfY, const double* CfZ, double t,\n"
        "    const double* fld, const int* faceCell,\n"
        "    double* ref, double* vfrac)\n"
        "{\n"
        "    int tid = blockIdx.x*blockDim.x + threadIdx.x;\n"
        "    if (tid >= n) return;\n"
        "    int g = offset + tid;\n"
        "    const double fx = CfX[g], fy = CfY[g], fz = CfZ[g];\n"
        "    const int ic = faceCell[g];\n"
        "    const double cc = fld[ic];\n"
        "    (void)fx;(void)fy;(void)fz;(void)cc;(void)t;\n"
        "    double out = 0; double vf = 1;\n"
        "    {\n" + body + "\n    }\n"
        "    ref[g] = out; vfrac[g] = vf;\n"
        "}\n";
    return compileSource(src, fn, name);
}

void launchCodedMixedVectorBc(
    const CodedBcKernel& k, int offset, int count, scalar t,
    const DeviceBuffer<scalar>& CfX, const DeviceBuffer<scalar>& CfY, const DeviceBuffer<scalar>& CfZ,
    const DeviceBuffer<scalar>& Ux, const DeviceBuffer<scalar>& Uy, const DeviceBuffer<scalar>& Uz,
    const DeviceBuffer<label>& faceCell,
    DeviceBuffer<scalar>& refX, DeviceBuffer<scalar>& refY, DeviceBuffer<scalar>& refZ,
    DeviceBuffer<scalar>& vfX, DeviceBuffer<scalar>& vfY, DeviceBuffer<scalar>& vfZ)
{
    if (!k.valid() || count <= 0) return;
    int n = count, off = offset; scalar tt = t;
    const scalar* cfx = CfX.data(); const scalar* cfy = CfY.data(); const scalar* cfz = CfZ.data();
    const scalar* ux = Ux.data(); const scalar* uy = Uy.data(); const scalar* uz = Uz.data();
    const label* fc = faceCell.data();
    scalar* rx = refX.data(); scalar* ry = refY.data(); scalar* rz = refZ.data();
    scalar* vx = vfX.data(); scalar* vy = vfY.data(); scalar* vz = vfZ.data();
    void* args[] = { &n, &off, &cfx, &cfy, &cfz, &tt, &ux, &uy, &uz, &fc, &rx, &ry, &rz, &vx, &vy, &vz };
    cuCheck(cuLaunchKernel(static_cast<CUfunction>(k.func), nBlocks(count),1,1, TPB,1,1, 0, CU_STREAM_PER_THREAD, args, nullptr), "launchKernel");
}

void launchCodedMixedScalarBc(
    const CodedBcKernel& k, int offset, int count, scalar t,
    const DeviceBuffer<scalar>& CfX, const DeviceBuffer<scalar>& CfY, const DeviceBuffer<scalar>& CfZ,
    const DeviceBuffer<scalar>& fld, const DeviceBuffer<label>& faceCell,
    DeviceBuffer<scalar>& ref, DeviceBuffer<scalar>& vfrac)
{
    if (!k.valid() || count <= 0) return;
    int n = count, off = offset; scalar tt = t;
    const scalar* cfx = CfX.data(); const scalar* cfy = CfY.data(); const scalar* cfz = CfZ.data();
    const scalar* f = fld.data();
    const label* fc = faceCell.data();
    scalar* r = ref.data(); scalar* v = vfrac.data();
    void* args[] = { &n, &off, &cfx, &cfy, &cfz, &tt, &f, &fc, &r, &v };
    cuCheck(cuLaunchKernel(static_cast<CUfunction>(k.func), nBlocks(count),1,1, TPB,1,1, 0, CU_STREAM_PER_THREAD, args, nullptr), "launchKernel");
}

} // namespace brae
