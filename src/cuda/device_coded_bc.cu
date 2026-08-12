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
    "__device__ inline vec3 make3(double a, double b, double c){ vec3 v; v.x=a; v.y=b; v.z=c; return v; }\n"
    // ------------------------------------------------------------------------------------------
    // OpenFOAM Field-API compatibility, IN ADDITION to the per-face model above (which several
    // cases already use and which is unchanged).
    //
    // OF snippets are written against whole-field types:
    //     const vector axis(1, 0, 0);
    //     vectorField v(2.0*this->patch().Cf() ^ axis);
    //     v.replace(vector::X, 1.0);
    //     operator==(v);
    // (simpleFoam/pipeCyclic). That is a different programming model from `out.x = ...`, not a
    // missing symbol -- which is why the snippet failed to compile at all.
    //
    // WHY A PER-FACE PROXY IS EXACT HERE, not an approximation: every operator in that expression is
    // ELEMENT-WISE (scale, cross product, component replace, assign). A field expression built only
    // from element-wise operators evaluated face-by-face is identical to evaluating it on the whole
    // field. So `vectorField` is a typedef of the per-face vector.
    //
    // THE LIMIT, and it is a real one: anything with a REDUCTION (gSum, average, max over the patch)
    // is NOT element-wise and would silently give the per-face value instead of the field value.
    // Those are refused by name at parse time rather than compiled -- see rejectFieldReductions().
    "struct ofVec {\n"
    "  double x, y, z;\n"
    "  __device__ ofVec() : x(0), y(0), z(0) {}\n"
    "  __device__ ofVec(double a, double b, double c) : x(a), y(b), z(c) {}\n"
    "  __device__ void replace(int c, double v) { if (c==0) x=v; else if (c==1) y=v; else z=v; }\n"
    "  __device__ double component(int c) const { return c==0?x:(c==1?y:z); }\n"
    "  static const int X = 0, Y = 1, Z = 2;\n"
    "};\n"
    "typedef ofVec vector;\n"
    "typedef ofVec vectorField;\n"      // per-face proxy: exact for element-wise expressions
    "typedef double scalar;\n"
    "typedef double scalarField;\n"
    "__device__ inline ofVec operator^(const ofVec& a, const ofVec& b){\n"
    "  return ofVec(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x); }\n"
    "__device__ inline ofVec operator*(double s, const ofVec& a){ return ofVec(s*a.x, s*a.y, s*a.z); }\n"
    "__device__ inline ofVec operator*(const ofVec& a, double s){ return ofVec(s*a.x, s*a.y, s*a.z); }\n"
    "__device__ inline ofVec operator+(const ofVec& a, const ofVec& b){ return ofVec(a.x+b.x, a.y+b.y, a.z+b.z); }\n"
    "__device__ inline ofVec operator-(const ofVec& a, const ofVec& b){ return ofVec(a.x-b.x, a.y-b.y, a.z-b.z); }\n"
    "__device__ inline double operator&(const ofVec& a, const ofVec& b){ return a.x*b.x + a.y*b.y + a.z*b.z; }\n"
    "__device__ inline double mag(const ofVec& a){ return sqrt(a.x*a.x + a.y*a.y + a.z*a.z); }\n"
    "struct ofPatch {\n"
    "  double cfx, cfy, cfz;\n"
    "  __device__ ofVec Cf() const { return ofVec(cfx, cfy, cfz); }\n"
    "};\n";
} // namespace

// A per-face proxy is exact for ELEMENT-WISE field expressions and wrong for REDUCTIONS: gSum over a
// patch is one number for the whole patch, and evaluating it face-by-face silently yields the local
// value instead. That is precisely the "plausible wrong answer" this project keeps refusing, so these
// are rejected by name rather than compiled into something that looks like it worked.
static void rejectFieldReductions(const std::string& name, const std::string& body)
{
    static const char* kReductions[] = {
        "gSum", "gMax", "gMin", "gAverage", "sum(", "average(", "max(", "min(", "returnReduce"
    };
    for (const char* r : kReductions)
        if (body.find(r) != std::string::npos)
            throw std::runtime_error(
                std::string("brae coded BC '") + name + "': '" + r + "' is a field REDUCTION. brae "
                "evaluates coded snippets per face, which is exact for element-wise expressions but "
                "would give the local value instead of the patch value here. Refused rather than run.");
}

CodedBcKernel compileCodedVectorBc(const std::string& name, const std::string& body)
{
    rejectFieldReductions(name, body);
    const std::string fn = "bc_" + name;
    // The body becomes a MEMBER FUNCTION, not an inline block. OF snippets say `this->patch().Cf()`
    // and `operator==(v)`, both of which need a `this` to resolve against; a free block cannot provide
    // one. The per-face variables stay as members, so snippets written against the older model
    // ("out.x = fx*2") resolve exactly as before -- both models compile from the same preamble.
    const std::string src = std::string(VEC3_PREAMBLE) +
        "struct Ctx {\n"
        "  double fx, fy, fz, cx, cy, cz, t;\n"
        "  ofPatch p_;\n"
        "  vec3* outp;\n"
        "  __device__ const ofPatch& patch() const { return p_; }\n"
        "  __device__ void operator==(const ofVec& v) { outp->x=v.x; outp->y=v.y; outp->z=v.z; }\n"
        "  __device__ void run(vec3& out);\n"
        "};\n"
        "__device__ void Ctx::run(vec3& out) {\n"
        "  (void)fx;(void)fy;(void)fz;(void)cx;(void)cy;(void)cz;(void)t;(void)out;\n"
        + body + "\n}\n"
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
        "    vec3 out; out.x = 0; out.y = 0; out.z = 0;\n"
        "    Ctx ctx; ctx.fx=fx; ctx.fy=fy; ctx.fz=fz; ctx.cx=cx; ctx.cy=cy; ctx.cz=cz; ctx.t=t;\n"
        "    ctx.p_.cfx=fx; ctx.p_.cfy=fy; ctx.p_.cfz=fz; ctx.outp=&out;\n"
        "    ctx.run(out);\n"
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
