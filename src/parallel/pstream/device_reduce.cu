// See device_reduce.cuh. The one NVSHMEM host-side on-stream collective brae adds beyond the halo put/barrier.
#include "device_reduce.cuh"
#include "cf_pstream.cuh"                 // Pstream::nProcs / allReduce (MPG fallback)
#include <cstdlib>                        // getenv / atoi (A/B knob)
#ifdef BRAE_WITH_NVSHMEM
#include <nvshmem_host.h>                 // symmetric heap + team world (no device defines -> no -rdc)
#include <host/nvshmemx_coll_api.h>       // nvshmemx_double_sum_reduce_on_stream (host-callable, on-stream)
#endif

namespace brae {

static_assert(sizeof(scalar) == 8, "DeviceReducer uses nvshmemx_DOUBLE_sum_reduce; scalar must be FP64");

DeviceReducer::DeviceReducer(int maxN)
    : src_(static_cast<std::size_t>(maxN)), dst_(static_cast<std::size_t>(maxN))
{
    // brae binds rank -> (rank % nDevices), so more ranks than visible GPUs means at least one GPU is shared
    // (MPG). NVSHMEM's on-stream reduce is unsupported under MPG; detect it once here to pick the path.
    int nDev = 0;
    cudaGetDeviceCount(&nDev);
    mpg_ = (nDev > 0) && (Pstream::nProcs() > nDev);
    // BRAE_FORCE_HOST_REDUCE=1 forces the host-MPI_Allreduce path even on real multi-GPU -- an A/B knob to
    // isolate the on-stream-reduce speedup against the old reduction path on identical everything else.
    if (const char* e = std::getenv("BRAE_FORCE_HOST_REDUCE")) if (std::atoi(e) != 0) mpg_ = true;
}

void DeviceReducer::sumReduce(int n, cudaStream_t stream)
{
#ifdef BRAE_WITH_NVSHMEM
    if (!mpg_)
    {
        // 1 PE per GPU (real multi-GPU): on-stream global SUM across NVSHMEM_TEAM_WORLD. Enqueued on `stream`, no
        // host block, no scalar over PCIe. At np=1 (one PE) it is the identity, matching the host-MPI np=1 path.
        nvshmemx_double_sum_reduce_on_stream(NVSHMEM_TEAM_WORLD, dst_.data(), src_.data(),
                                             static_cast<std::size_t>(n), stream);
        return;
    }
    // MPG (oversubscribed single-GPU oracle): on-stream reduce unsupported -> blocking host MPI_Allreduce. Only
    // the correctness oracle takes this; real multi-GPU uses the on-stream path. Bounded by DeviceReducer's maxN.
    scalar h[8];
    cudaMemcpyAsync(h, src_.data(), static_cast<std::size_t>(n) * sizeof(scalar), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    Pstream::allReduce(h, n, ReduceOp::Sum);
    cudaMemcpyAsync(dst_.data(), h, static_cast<std::size_t>(n) * sizeof(scalar), cudaMemcpyHostToDevice, stream);
#else
    // No NVSHMEM: a single PE, so the SUM reduction is the identity -> device copy (stream-ordered).
    cudaMemcpyAsync(dst_.data(), src_.data(), static_cast<std::size_t>(n) * sizeof(scalar),
                    cudaMemcpyDeviceToDevice, stream);
#endif
}

} // namespace brae
