#pragma once
// cf GPU offload (#7): thread-block CLUSTERS + distributed shared memory (DSM). The architecture's specified
// residency mechanism (single GPU, clusters/DSM, NOT cooperative grid). This first brick is the Krylov scalar
// reduction (dot / sumMag), the operation that today forces a host round-trip every solver iteration. The
// cross-block combine here uses DSM (a block reads its cluster-siblings' shared memory via map_shared_rank +
// a cluster-wide barrier) instead of global atomics/passes, the building block for keeping the reductions
// on-chip. Validated to the same value as deviceDot / deviceSumMag (the CPU-oracle-backed reductions).
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_ldu.cuh"

namespace brae {

bool   deviceClusterSupported();                                            // cudaDevAttrClusterLaunch
scalar deviceClusterDot(const DeviceBuffer<scalar>& x, const DeviceBuffer<scalar>& y);   // x . y  (cluster+DSM)
scalar deviceClusterSumMag(const DeviceBuffer<scalar>& x);                  // sum |x|  (cluster+DSM)

// Cluster+DSM SpMV: each cluster owns a contiguous cell chunk, caches its psi in distributed shared memory, and
// gathers neighbour values from DSM (within-chunk) or global (cross-chunk). Same result as deviceAmul (bit-exact).
void deviceClusterAmul(const DeviceLduView& A, const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& Apsi);

} // namespace brae
