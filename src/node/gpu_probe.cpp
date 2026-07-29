#include "gpu_probe.h"

#include "sha256.h"

#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <unistd.h>

namespace brae::node {
namespace {

// The slice of NVML the agent uses. Declared here rather than including nvml.h so the build needs no CUDA
// toolkit at all -- these signatures are ABI-stable and documented.
using nvmlReturn = int;
constexpr nvmlReturn NVML_SUCCESS = 0;

using FnInit = nvmlReturn (*)();
using FnShutdown = nvmlReturn (*)();
using FnDeviceCount = nvmlReturn (*)(unsigned*);
using FnHandleByIndex = nvmlReturn (*)(unsigned, void**);
using FnName = nvmlReturn (*)(void*, char*, unsigned);
using FnUuid = nvmlReturn (*)(void*, char*, unsigned);
using FnMemory = nvmlReturn (*)(void*, unsigned long long*);      // reads into {total, free, used}
using FnUtil = nvmlReturn (*)(void*, unsigned*);                  // reads into {gpu, memory}
using FnTemp = nvmlReturn (*)(void*, int, unsigned*);
using FnCudaCap = nvmlReturn (*)(void*, int*, int*);
using FnDriver = nvmlReturn (*)(char*, unsigned);

struct Nvml
{
    void* handle = nullptr;
    FnInit init = nullptr;
    FnShutdown shutdown = nullptr;
    FnDeviceCount count = nullptr;
    FnHandleByIndex byIndex = nullptr;
    FnName name = nullptr;
    FnUuid uuid = nullptr;
    FnMemory memory = nullptr;
    FnUtil util = nullptr;
    FnTemp temp = nullptr;
    FnCudaCap cudaCap = nullptr;
    FnDriver driver = nullptr;
    std::string error;

    bool ok() const { return handle && init && count && byIndex; }

    ~Nvml()
    {
        if (handle)
        {
            if (shutdown) shutdown();
            ::dlclose(handle);
        }
    }
};

template <typename T>
T sym(void* h, const char* name)
{
    return reinterpret_cast<T>(::dlsym(h, name));
}

// Open NVML and resolve what we need. Any failure is reported, never thrown: "this machine has no GPU" is a
// state the agent must survive and describe, not an exception.
bool openNvml(Nvml& n)
{
    ::dlerror();
    n.handle = ::dlopen(nvmlLibraryName().c_str(), RTLD_LAZY | RTLD_LOCAL);
    if (!n.handle)
    {
        const char* e = ::dlerror();
        n.error = std::string("cannot load ") + nvmlLibraryName() + (e ? std::string(": ") + e : "");
        return false;
    }
    n.init      = sym<FnInit>(n.handle, "nvmlInit_v2");
    n.shutdown  = sym<FnShutdown>(n.handle, "nvmlShutdown");
    n.count     = sym<FnDeviceCount>(n.handle, "nvmlDeviceGetCount_v2");
    n.byIndex   = sym<FnHandleByIndex>(n.handle, "nvmlDeviceGetHandleByIndex_v2");
    n.name      = sym<FnName>(n.handle, "nvmlDeviceGetName");
    n.uuid      = sym<FnUuid>(n.handle, "nvmlDeviceGetUUID");
    n.memory    = sym<FnMemory>(n.handle, "nvmlDeviceGetMemoryInfo");
    n.util      = sym<FnUtil>(n.handle, "nvmlDeviceGetUtilizationRates");
    n.temp      = sym<FnTemp>(n.handle, "nvmlDeviceGetTemperature");
    n.cudaCap   = sym<FnCudaCap>(n.handle, "nvmlDeviceGetCudaComputeCapability");
    n.driver    = sym<FnDriver>(n.handle, "nvmlSystemGetDriverVersion");

    if (!n.ok())
    {
        n.error = nvmlLibraryName() + " is missing the NVML entry points this agent needs";
        return false;
    }
    if (n.init() != NVML_SUCCESS)
    {
        n.error = "nvmlInit failed (driver present but not usable)";
        return false;
    }
    return true;
}

// Total system RAM, in MiB.
//
// Needed because NVML reports no memory at all on unified-memory parts: on a GB10, both
// nvmlDeviceGetMemoryInfo and its _v2 form return NVML_ERROR_NOT_SUPPORTED with total = 0. That is not a
// failure to work around quietly -- on those machines the GPU genuinely shares the CPU's memory, so system RAM
// IS the figure the scheduler needs when deciding whether a job fits. Dropping the card instead (which an
// earlier version did) made a GB10 report zero GPUs and left it unable to register at all.
int systemMemoryMb()
{
    const long pages = ::sysconf(_SC_PHYS_PAGES);
    const long pageSize = ::sysconf(_SC_PAGE_SIZE);
    if (pages <= 0 || pageSize <= 0) return 0;
    return static_cast<int>((static_cast<long long>(pages) * pageSize) / (1024 * 1024));
}

int systemMemoryUsedMb()
{
    const long pages = ::sysconf(_SC_PHYS_PAGES);
    const long avail = ::sysconf(_SC_AVPHYS_PAGES);
    const long pageSize = ::sysconf(_SC_PAGE_SIZE);
    if (pages <= 0 || avail < 0 || pageSize <= 0) return 0;
    const long long used = (static_cast<long long>(pages) - avail) * pageSize;
    return static_cast<int>(used / (1024 * 1024));
}

}  // namespace

std::string nvmlLibraryName()
{
    if (const char* over = std::getenv("BRAE_NVML_LIB")) return over;
    return "libnvidia-ml.so.1";
}

GpuProbeResult probeGpus()
{
    GpuProbeResult out;
    Nvml n;
    if (!openNvml(n)) { out.unavailableReason = n.error; return out; }
    out.nvmlAvailable = true;

    std::string driverVersion;
    if (n.driver)
    {
        char buf[128] = {0};
        if (n.driver(buf, sizeof buf) == NVML_SUCCESS) driverVersion = buf;
    }

    unsigned count = 0;
    if (n.count(&count) != NVML_SUCCESS) { out.unavailableReason = "nvmlDeviceGetCount failed"; return out; }

    for (unsigned i = 0; i < count; ++i)
    {
        void* dev = nullptr;
        if (n.byIndex(i, &dev) != NVML_SUCCESS) continue;

        GpuIdentity g;
        g.index = static_cast<int>(i);
        g.driverVersion = driverVersion;

        if (n.name)
        {
            char buf[128] = {0};
            if (n.name(dev, buf, sizeof buf) == NVML_SUCCESS) g.model = buf;
        }
        if (n.memory)
        {
            unsigned long long mem[3] = {0, 0, 0};       // total, free, used
            if (n.memory(dev, mem) == NVML_SUCCESS) g.vramMb = static_cast<int>(mem[0] / (1024ull * 1024ull));
        }
        if (g.vramMb <= 0) g.vramMb = systemMemoryMb();   // unified memory: see systemMemoryMb
        if (n.cudaCap)
        {
            int major = 0, minor = 0;
            if (n.cudaCap(dev, &major, &minor) == NVML_SUCCESS)
                g.computeCapability = std::to_string(major) + "." + std::to_string(minor);
        }
        if (n.uuid)
        {
            char buf[128] = {0};
            if (n.uuid(dev, buf, sizeof buf) == NVML_SUCCESS)
            {
                // Hashed immediately. The raw serial is not stored in g, not returned, and not logged.
                g.uuidHash = sha256Hex(buf);
                std::memset(buf, 0, sizeof buf);
            }
        }
        // A card we cannot identify is not reported: the registry keys GPUs by uuid_hash, and an entry without
        // one would register as a new card on every restart.
        if (!g.uuidHash.empty() && g.vramMb > 0) out.gpus.push_back(std::move(g));
    }
    return out;
}

std::vector<GpuState> probeGpuState()
{
    std::vector<GpuState> out;
    Nvml n;
    if (!openNvml(n)) return out;

    unsigned count = 0;
    if (n.count(&count) != NVML_SUCCESS) return out;

    for (unsigned i = 0; i < count; ++i)
    {
        void* dev = nullptr;
        if (n.byIndex(i, &dev) != NVML_SUCCESS) continue;

        GpuState s;
        s.index = static_cast<int>(i);
        if (n.memory)
        {
            unsigned long long mem[3] = {0, 0, 0};
            if (n.memory(dev, mem) == NVML_SUCCESS)
            {
                s.memoryTotalMb = static_cast<int>(mem[0] / (1024ull * 1024ull));
                s.memoryUsedMb = static_cast<int>(mem[2] / (1024ull * 1024ull));
            }
        }
        if (n.util)
        {
            unsigned u[2] = {0, 0};                      // gpu, memory
            if (n.util(dev, u) == NVML_SUCCESS) s.utilizationPercent = static_cast<int>(u[0]);
        }
        if (n.temp)
        {
            unsigned t = 0;
            if (n.temp(dev, 0 /* NVML_TEMPERATURE_GPU */, &t) == NVML_SUCCESS)
                s.temperatureCelsius = static_cast<int>(t);
        }

        // Unified memory again: NVML reports nothing, so the machine's own memory is the honest figure.
        if (s.memoryTotalMb <= 0)
        {
            s.memoryTotalMb = systemMemoryMb();
            s.memoryUsedMb = systemMemoryUsedMb();
        }

        // The API rejects used > total (422), and a partial NVML read can produce exactly that. Clamp rather
        // than send a snapshot the server will refuse -- one bad field must not cost the whole heartbeat.
        if (s.memoryTotalMb <= 0) continue;
        if (s.memoryUsedMb > s.memoryTotalMb) s.memoryUsedMb = s.memoryTotalMb;
        if (s.utilizationPercent > 100) s.utilizationPercent = 100;
        out.push_back(s);
    }
    return out;
}

}  // namespace brae::node
