// Phase 0a gate: Pstream::init binds each rank to a GPU (rank % visibleGPUs).
//
// Verifies the binding is consistent (Pstream::deviceId == the CUDA current device == rank % nDev) and that
// N ranks sharing one GPU (GB10, or --oversubscribe) init without context errors. On a host-only node with
// no GPU the binding is a no-op (deviceId == -1) and the test passes trivially.
//
// Run: mpirun -np {1,2,4,8} test_gpu_bind
#include "cf_pstream.cuh"

#include <cstdio>
#include <cuda_runtime.h>

using namespace brae;

int main(int argc, char** argv)
{
    Pstream::init(argc, argv);
    const int rank  = Pstream::myProcNo();
    const int nproc = Pstream::nProcs();

    int failures = 0;

    int nDev = 0;
    if (cudaGetDeviceCount(&nDev) != cudaSuccess) { cudaGetLastError(); nDev = 0; }

    const int bound = Pstream::deviceId();

    if (nDev <= 0)
    {
        if (bound != -1)
        {
            std::printf("[rank %d] FAIL: no GPU but deviceId=%d (expected -1)\n", rank, bound);
            ++failures;
        }
    }
    else
    {
        const int expect = rank % nDev;
        int current = -1;
        cudaGetDevice(&current);
        if (bound != expect || current != expect)
        {
            std::printf("[rank %d] FAIL: deviceId=%d current=%d expected=%d (nDev=%d)\n",
                        rank, bound, current, expect, nDev);
            ++failures;
        }
        // A trivial allocation confirms the bound context is actually usable (no context error under sharing).
        void* p = nullptr;
        if (cudaMalloc(&p, 32) != cudaSuccess)
        {
            std::printf("[rank %d] FAIL: cudaMalloc on bound device %d failed\n", rank, expect);
            cudaGetLastError();
            ++failures;
        }
        else
            cudaFree(p);
        std::printf("[rank %d/%d] bound to GPU %d of %d\n", rank, nproc, bound, nDev);
    }

    const label totalFail = Pstream::allReduce(static_cast<label>(failures), ReduceOp::Sum);
    if (Pstream::master())
        std::printf("test_gpu_bind: nproc=%d  %s (failures=%d)\n",
                    nproc, totalFail == 0 ? "PASS" : "FAIL", static_cast<int>(totalFail));

    Pstream::finalize();
    return totalFail == 0 ? 0 : 1;
}
