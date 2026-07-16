// Phase 1a smoke test: NVSHMEM bootstraps off MPI, the symmetric heap allocates, and a put between PEs
// delivers the value. Ring topology: each PE puts its value into the NEXT PE's buffer, so after a barrier
// every PE holds what its PREVIOUS neighbour sent. On GB10 (one GPU) the PEs share device 0 over the P2P
// self/loopback path -- enough to prove bootstrap + heap + put before the device interface (Phase 1b).
//
// Run: mpirun -np {1,2,4} test_nvshmem_smoke
#include "cf_pstream.cuh"
#include "sym_buffer.cuh"

#include <cstdio>
#include <vector>
#ifdef BRAE_WITH_NVSHMEM
#include <nvshmem_host.h>
#endif

using namespace brae;

int main(int argc, char** argv)
{
    Pstream::init(argc, argv);
    const int rank  = Pstream::myProcNo();
    const int nproc = Pstream::nProcs();

#ifndef BRAE_WITH_NVSHMEM
    if (Pstream::master()) std::printf("test_nvshmem_smoke: built without NVSHMEM -- SKIP\n");
    Pstream::finalize();
    return 0;
#else
    int fails = 0;

    if (!Pstream::nvshmemActive())
    {
        if (Pstream::master()) std::printf("test_nvshmem_smoke: NVSHMEM inactive (no GPU?) -- SKIP\n");
        Pstream::finalize();
        return 0;
    }

    // (1) PE identity lines up with the MPI rank (bootstrapped off MPI_COMM_WORLD).
    if (nvshmem_my_pe() != rank || nvshmem_n_pes() != nproc)
    {
        std::printf("[rank %d] FAIL: nvshmem pe=%d/%d vs mpi=%d/%d\n",
                    rank, nvshmem_my_pe(), nvshmem_n_pes(), rank, nproc);
        ++fails;
    }

    // (2) symmetric-heap put: src holds our value, put it into the next PE's dst.
    // NOTE: symmetric buffers must be freed BEFORE nvshmem is finalized -- hence the inner scope so the
    // SymBuffer destructors (nvshmem_free) run before Pstream::finalize(). Real code must observe the same
    // ordering (tear down the device interfaces before Pstream::finalize()).
    {
        const int next = (rank + 1) % nproc;
        const int prev = (rank - 1 + nproc) % nproc;
        SymBuffer<double> src(1), dst(1);
        src.copyFrom(std::vector<double>{100.0 + rank});
        dst.copyFrom(std::vector<double>{-1.0});
        nvshmem_barrier_all();
        nvshmem_putmem(dst.data(), src.data(), sizeof(double), next);   // write our src into next PE's dst
        nvshmem_barrier_all();                                          // completion + sync

        const double got    = dst.host()[0];
        const double expect = 100.0 + prev;
        if (got != expect)
        {
            std::printf("[rank %d] FAIL put: got %.1f expect %.1f (from prev=%d)\n", rank, got, expect, prev);
            ++fails;
        }
    }

    const label totalFail = Pstream::allReduce(static_cast<label>(fails), ReduceOp::Sum);
    if (Pstream::master())
        std::printf("test_nvshmem_smoke: nproc=%d  %s (failures=%d)\n",
                    nproc, totalFail == 0 ? "PASS" : "FAIL", static_cast<int>(totalFail));

    Pstream::finalize();
    return totalFail == 0 ? 0 : 1;
#endif
}
