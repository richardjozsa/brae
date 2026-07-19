#pragma once
// On-GPU global SUM reduction across all PEs, ON-STREAM (no host round-trip), for the distributed Krylov solvers.
// This replaces the multi-GPU per-iteration bottleneck: today every Krylov dot does deviceDot -> a blocking D2H
// copy of one scalar -> a blocking host MPI_Allreduce (see device_pcg.cu). Here the caller writes its local
// partial sums into src() (device, in the NVSHMEM symmetric heap), calls sumReduce(n), and reads the global sums
// from dst() (device, symmetric) -- entirely stream-ordered, the host untouched. Only the convergence test still
// reads a scalar to the host, and the callers batch that to every K iterations.
//
// NVSHMEM's team reduce (nvshmemx_double_sum_reduce_on_stream) is the on-stream device collective; it requires
// its operands to live in the symmetric heap, hence SymBuffer. When NVSHMEM is inactive (np=1, or a no-device CI
// node) the SUM over the single PE is the identity, so sumReduce degrades to a device copy and the np=1 oracle is
// bit-unchanged. SPUMA cannot do this -- its reductions are host MPI_Allreduce-bound (see the SPUMA analysis).
#include "cf_types.cuh"
#include "sym_buffer.cuh"
#include <cuda_runtime.h>

namespace brae {

class DeviceReducer
{
public:
    explicit DeviceReducer(int maxN);
    scalar*       src()       { return src_.data(); }   // write the n LOCAL partial sums here (device)
    const scalar* dst() const { return dst_.data(); }   // the n GLOBAL sums land here (device)
    scalar*       dst()       { return dst_.data(); }
    // Reduce the first n scalars of src() across all PEs (SUM), result in dst(), on `stream`. On real multi-GPU
    // (1 PE/GPU) this is the on-stream NVSHMEM collective (no host sync); under MPG (multiple PEs per GPU, e.g. an
    // oversubscribed single-GPU CI node, where NVSHMEM's on-stream reduce is unsupported) it falls back to a
    // blocking host MPI_Allreduce -- correct, and only the correctness oracle ever takes that path.
    void sumReduce(int n, cudaStream_t stream = cudaStreamPerThread);
private:
    SymBuffer<scalar> src_, dst_;
    bool              mpg_ = false;   // multiple PEs share one GPU -> on-stream reduce unsupported, use host MPI
};

} // namespace brae
