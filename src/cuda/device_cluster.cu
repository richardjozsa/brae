// cf GPU offload (#7): cluster + DSM reductions. Each thread-block does a grid-stride load + an in-block
// shared-memory reduction to one partial; then the blocks of a CLUSTER combine their partials through
// DISTRIBUTED SHARED MEMORY, block rank 0 maps each sibling block's shared array (cluster.map_shared_rank)
// and sums them after a cluster-wide barrier, so the cross-block step never touches global memory. One
// partial per cluster is written out; the (few) cluster partials are summed to the scalar result. Launched
// with cudaLaunchKernelEx + cudaLaunchAttributeClusterDimension.
#include "device_cluster.cuh"
#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <vector>

namespace cg = cooperative_groups;

namespace brae {

namespace {
constexpr int TPB     = 256;   // threads / block
constexpr int CLUSTER = 8;     // blocks / cluster (GB10 portable max; 8*256 = 2048 threads, shared via DSM)
constexpr int NCL     = 6;     // clusters (6*8 = 48 blocks == 48 SMs on GB10); grid-stride covers any n

// OP 0: dot(x,y).  OP 1: sum|x|.  Writes one partial per cluster to out[clusterIdx].
template <int OP>
__global__ void clusterReduceKernel(const scalar* __restrict__ x, const scalar* __restrict__ y,
                                    int n, scalar* __restrict__ out) {
#if __CUDA_ARCH__ >= 900   // thread-block clusters exist only on sm_90+ (Hopper/Blackwell). Pre-Hopper compiles this
                           // kernel as a no-op; it is never launched there (host gates on deviceClusterSupported()).
    cg::cluster_group cluster = cg::this_cluster();
    __shared__ scalar sdata[TPB];
    const int tid    = threadIdx.x;
    const int gtid   = blockIdx.x * blockDim.x + tid;
    const int stride = gridDim.x * blockDim.x;

    scalar local = 0;
    for (int i = gtid; i < n; i += stride) local += (OP == 0) ? x[i] * y[i] : fabs(x[i]);
    sdata[tid] = local;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) { if (tid < s) sdata[tid] += sdata[tid + s]; __syncthreads(); }
    // sdata[0] now holds this block's partial.

    cluster.sync();                                          // all sibling blocks have written their partial
    if (cluster.block_rank() == 0 && tid == 0) {             // cluster leader gathers siblings via DSM
        scalar cs = 0;
        for (int b = 0; b < CLUSTER; ++b) {
            const scalar* remote = cluster.map_shared_rank(sdata, b);   // sibling block's shared array
            cs += remote[0];
        }
        out[blockIdx.x / CLUSTER] = cs;
    }
    cluster.sync();                                          // no block frees its shared mem before the read
#endif
}

scalar launchClusterReduce(const scalar* x, const scalar* y, int n, int op) {
    DeviceBuffer<scalar> partials(NCL);
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(NCL * CLUSTER); cfg.blockDim = dim3(TPB); cfg.dynamicSmemBytes = 0; cfg.stream = cudaStreamPerThread;
    cudaLaunchAttribute attr[1] = {};
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = CLUSTER; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    if (op == 0) cudaCheck(cudaLaunchKernelEx(&cfg, clusterReduceKernel<0>, x, y, n, partials.data()), "clusterDot");
    else         cudaCheck(cudaLaunchKernelEx(&cfg, clusterReduceKernel<1>, x, (const scalar*)nullptr, n, partials.data()), "clusterSumMag");
    cudaCheck(cudaGetLastError(), "clusterReduce launch");
    const std::vector<scalar> h = partials.host();          // NCL partials -> deterministic final sum
    scalar s = 0; for (scalar v : h) s += v; return s;
}
} // namespace

namespace {
// Cluster+DSM SpMV. Cluster 'cid' owns the contiguous chunk [cid*chunk, cid*chunk+chunk); block 'rank' holds
// cells [chunkBase+rank*cpb, ..) and caches their psi in DSM. A neighbour read goes to DSM (map_shared_rank to
// the owning sibling block) if the neighbour is in this cluster's chunk, else to global psi. Same gather order
// as deviceAmul -> bit-identical. Reads old psi everywhere (DSM loaded once, global unmodified) -> exact.
__global__ void clusterAmulKernel(int n, int chunk, int cpb,
        const scalar* __restrict__ psi, const scalar* __restrict__ diag,
        const label* __restrict__ ownerStart, const label* __restrict__ nei, const scalar* __restrict__ upper,
        const label* __restrict__ losortStart, const label* __restrict__ losort, const label* __restrict__ owner,
        const scalar* __restrict__ lower, scalar* __restrict__ Apsi) {
#if __CUDA_ARCH__ >= 900   // sm_90+ only (thread-block clusters + DSM); pre-Hopper compiles as a no-op, never launched.
    cg::cluster_group cluster = cg::this_cluster();
    const int rank = cluster.block_rank();
    const int cid = blockIdx.x / CLUSTER;
    extern __shared__ scalar sh[];                          // this block's psi slice (cpb scalars)
    const int chunkBase = cid * chunk;
    const int chunkEnd  = min(chunkBase + chunk, n);        // this cluster's actual chunk end
    const int blockBase = chunkBase + rank * cpb;
    const int myCount   = max(0, min(cpb, n - blockBase));
    for (int i = threadIdx.x; i < cpb; i += blockDim.x) sh[i] = (i < myCount) ? psi[blockBase + i] : 0.0;
    __syncthreads();
    cluster.sync();
    for (int i = threadIdx.x; i < myCount; i += blockDim.x) {
        const int c = blockBase + i;
        scalar Ax = diag[c] * sh[i];
        for (int f = ownerStart[c]; f < ownerStart[c + 1]; ++f) {
            const int g = nei[f];
            scalar pv; if (g >= chunkBase && g < chunkEnd) { const scalar* r = cluster.map_shared_rank(sh, (g - chunkBase) / cpb); pv = r[(g - chunkBase) % cpb]; } else pv = psi[g];
            Ax += upper[f] * pv;
        }
        for (int k = losortStart[c]; k < losortStart[c + 1]; ++k) {
            const int f = losort[k], g = owner[f];
            scalar pv; if (g >= chunkBase && g < chunkEnd) { const scalar* r = cluster.map_shared_rank(sh, (g - chunkBase) / cpb); pv = r[(g - chunkBase) % cpb]; } else pv = psi[g];
            Ax += lower[f] * pv;
        }
        Apsi[c] = Ax;
    }
    cluster.sync();                                        // no block frees its DSM while a sibling still reads it
#endif
}
} // namespace

void deviceClusterAmul(const DeviceLduView& A, const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& Apsi) {
    const int n = A.nCells;
    const int NCL_ = 6;                                     // 6 clusters * 8 = 48 blocks == 48 SMs
    const int chunk = (n + NCL_ - 1) / NCL_;
    const int cpb = (chunk + CLUSTER - 1) / CLUSTER;
    const std::size_t shBytes = static_cast<std::size_t>(cpb) * sizeof(scalar);
    Apsi.resize(n);
    cudaCheck(cudaFuncSetAttribute(clusterAmulKernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024), "clusterAmul shmem optin");
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(NCL_ * CLUSTER); cfg.blockDim = dim3(TPB); cfg.dynamicSmemBytes = shBytes; cfg.stream = cudaStreamPerThread;
    cudaLaunchAttribute attr[1] = {};
    attr[0].id = cudaLaunchAttributeClusterDimension;
    attr[0].val.clusterDim.x = CLUSTER; attr[0].val.clusterDim.y = 1; attr[0].val.clusterDim.z = 1;
    cfg.attrs = attr; cfg.numAttrs = 1;
    cudaCheck(cudaLaunchKernelEx(&cfg, clusterAmulKernel, n, chunk, cpb,
        psi.data(), A.diag, A.ownerStart, A.nei, A.upper, A.losortStart, A.losort, A.owner, A.lower, Apsi.data()),
        "clusterAmul");
    cudaCheck(cudaGetLastError(), "clusterAmul launch");
}

bool deviceClusterSupported() {
    int v = 0, dev = 0; cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&v, cudaDevAttrClusterLaunch, dev);
    return v != 0;
}

scalar deviceClusterDot(const DeviceBuffer<scalar>& x, const DeviceBuffer<scalar>& y) {
    return launchClusterReduce(x.data(), y.data(), static_cast<int>(x.size()), 0);
}
scalar deviceClusterSumMag(const DeviceBuffer<scalar>& x) {
    return launchClusterReduce(x.data(), nullptr, static_cast<int>(x.size()), 1);
}

} // namespace brae
