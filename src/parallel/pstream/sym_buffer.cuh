#pragma once
// brae SymBuffer<T>: RAII device memory usable as a target of an NVSHMEM put/get.
//
// With BRAE_WITH_NVSHMEM it lives in the NVSHMEM symmetric heap (nvshmem_malloc/free) -- the only memory a
// remote PE may put into. nvshmem_malloc/free are COLLECTIVE and SYMMETRIC: every PE must construct and
// destroy the same byte size in the same order. Interface buffers whose per-partition size differs are
// therefore allocated to a common maximum (see the device processor interface). The surface mirrors
// DeviceBuffer so call sites look identical.
//
// Without BRAE_WITH_NVSHMEM it degrades to a plain cudaMalloc buffer (a single-GPU build never exchanges), so
// all code still compiles and runs when NVSHMEM is absent.
#include "cf_types.cuh"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef BRAE_WITH_NVSHMEM
#include <nvshmem_host.h>   // host-side nvshmem_malloc/free (no device defines -> no -rdc)
#endif

namespace brae {

template <typename T>
class SymBuffer
{
public:
    SymBuffer() = default;
    explicit SymBuffer(std::size_t n) { alloc(n); }
    ~SymBuffer() { release(); }

    SymBuffer(const SymBuffer&) = delete;
    SymBuffer& operator=(const SymBuffer&) = delete;
    SymBuffer(SymBuffer&& o) noexcept : d_(o.d_), n_(o.n_) { o.d_ = nullptr; o.n_ = 0; }
    SymBuffer& operator=(SymBuffer&& o) noexcept
    {
        if (this != &o)
        {
            release();
            d_ = o.d_;
            n_ = o.n_;
            o.d_ = nullptr;
            o.n_ = 0;
        }
        return *this;
    }

    void copyFrom(const std::vector<T>& h)                       // H2D
    {
        if (h.size() != n_) alloc(h.size());
        if (n_) cudaMemcpy(d_, h.data(), n_ * sizeof(T), cudaMemcpyHostToDevice);
    }
    std::vector<T> host() const                                  // D2H
    {
        std::vector<T> h(n_);
        if (n_) cudaMemcpy(h.data(), d_, n_ * sizeof(T), cudaMemcpyDeviceToHost);
        return h;
    }

    T*          data()       { return d_; }
    const T*    data() const { return d_; }
    std::size_t size() const { return n_; }

private:
    void alloc(std::size_t n)
    {
        release();
        n_ = n;
        if (!n_)
        {
            d_ = nullptr;
            return;
        }
#ifdef BRAE_WITH_NVSHMEM
        d_ = static_cast<T*>(nvshmem_malloc(n_ * sizeof(T)));    // collective + symmetric across all PEs
        if (!d_) throw std::runtime_error("brae SymBuffer: nvshmem_malloc failed");
#else
        if (cudaMalloc(&d_, n_ * sizeof(T)) != cudaSuccess)
            throw std::runtime_error("brae SymBuffer: cudaMalloc failed");
#endif
    }
    void release()
    {
        if (!d_) return;
#ifdef BRAE_WITH_NVSHMEM
        nvshmem_free(d_);
#else
        cudaFree(d_);
#endif
        d_ = nullptr;
        n_ = 0;
    }

    T*          d_ = nullptr;
    std::size_t n_ = 0;
};

} // namespace brae
