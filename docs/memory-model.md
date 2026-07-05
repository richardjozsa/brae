# Memory model and data layout

Why brae lays out its data the way it does, answering the two questions that come up most: *why a device
memory pool instead of pinned memory or `thrust` vectors?* and *why keep OpenFOAM's LDU matrix instead of
converting to CSR for cuSPARSE?*

## Summary

| concern | brae's choice | why |
|---|---|---|
| per-iteration transfers | none (device-resident) | the actual speedup; no migration tax |
| bulk storage | explicit `cudaMalloc`, pooled | makes "0 copies/iter" measurable; kills allocator churn |
| temporaries | caching pool (size-keyed free list) | per-iteration malloc/free churn -> ~0; predictable timing |
| pinned host memory | 2 scalars only (reduction mailbox) | fast single-value sync; not a data store |
| field layout | structure-of-arrays | coalesced access |
| matrix | LDU + cell-gather (not CSR/cuSPARSE) | atomic-free, reproducible, shared addressing; net faster in-solver |
| rebuild costs | AMG hierarchy cache + CUDA-graph replay | reuse across iterations |

The one theme behind all of it: **the speedup is residency, not any single allocator trick**, and everything
below exists to *do the expensive thing once and reuse it*, whether that is a device buffer, the AMG
hierarchy, or a captured CUDA graph.

## The detail

**Explicit device memory, not managed.** Buffers use `cudaMalloc` behind an RAII wrapper (`DeviceBuffer<T>`),
not `cudaMallocManaged`. With explicit memory every host-device copy is a named call, so "0 copies per
iteration" is something we measure and enforce; unified memory would hide exactly the transfers we are trying
to eliminate.

**The device pool.** The loop reuses many short-lived temporaries each iteration (interpolated fields, fluxes,
gradients, residual and AMG workspace); allocating/freeing them through the driver adds overhead and
synchronization. `DevicePool` (`src/cuda/device_buffer.cuh`) retains freed blocks in a size-keyed free list
and hands them back to the next same-size request, so after the first iteration steady-state malloc/free drops
to ~0. A/B it with `BRAE_NO_DEVICE_POOL=1`.

**Pool vs pinned, they solve different problems.** Pinned host memory speeds up *host-device transfers*; brae's
point is to *not do those inside the loop*, so pinned is not the lever. The only pinned allocation is two
scalars (`device_blas.cu`): a host mirror for reduction results, so the one unavoidable device-to-host sync per
reduction is fast. A mailbox for one number, not a data store. (Likewise `thrust::device_vector` temporaries
carry the same alloc + sync cost the pool removes, and lack the stable pointer CUDA-graph capture needs.)

**LDU matrix + cell-gather, not CSR.** Brae keeps OpenFOAM's LDU format (diagonal + `upper`/`lower` by face,
`owner`/`neighbour` + `ownerStart`/`losort` addressing). The SpMV is a per-cell gather: one thread per cell, so
each output is written by one thread, no atomics, no race, fixed (reproducible) accumulation order. Converting
to CSR + cuSPARSE was measured and rejected (`validation/perf/spmv_bench.cu`): cuSPARSE's kernel is marginally
faster in isolation (~1.3x), but **net ~1.25x slower in-solver** because every call must convert LDU->CSR
(doubling traffic on a bandwidth-bound kernel) and would not survive CUDA-graph capture. Beyond the benchmark,
assembly, boundary handling, pressure correction, and turbulence all share the same LDU addressing, so keeping
one representation is simpler than re-deriving CSR per operation.

**Caches beyond the pool** (same "reuse, don't rebuild" idea): the AMG hierarchy is static per mesh, so its
structure is serialized and reloaded warm (the `-partition` step); and the multigrid V-cycle + device-resident
PCG body are captured once as CUDA graphs and replayed, which needs the stable pointers the pool provides.

**The tradeoff.** Cell-gather over an unstructured mesh means some loops have irregular neighbour access
(scattered in memory). That is inherent to unstructured FV; renumbering improves locality but does not remove
it. For LDU matrices where every stage shares addressing, gather was the better first layout, measured, not
assumed.
