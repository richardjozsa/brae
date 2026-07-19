// brae::Pstream MPI backend. Re-expressed from OpenFOAM src/Pstream/mpi semantics
// (non-blocking exchange + Allreduce); not copied verbatim.
#include <chrono>
#include <cstdlib>
#include "cf_pstream.cuh"

#include <mpi.h>
#include <cuda_runtime.h>
#include <vector>
#ifdef BRAE_WITH_NVSHMEM
#include <nvshmem_host.h>   // host-only API (no device defines -> no -rdc needed for the bootstrap)
#endif

namespace brae {
namespace {
    bool                     g_init    = false;
    int                      g_rank    = 0;
    int                      g_nprocs  = 1;
    int                      g_device  = -1;   // CUDA device this rank owns (round-robin over visible GPUs)
    bool                     g_nvshmem = false; // NVSHMEM device transport bootstrapped
    std::vector<MPI_Request> g_requests;   // pending isend/irecv

    // Bind this rank to one GPU (rank % visibleGPUs). Defensive: a host-only build or a CPU-only CI node has
    // no device, so a missing/zero device count is not an error -- the binding is simply skipped and any CUDA
    // error state is cleared so later code is unaffected. On GB10 (one GPU) every rank binds to device 0.
    void bindRankToDevice()
    {
        int nDev = 0;
        if (cudaGetDeviceCount(&nDev) != cudaSuccess || nDev <= 0)
        {
            cudaGetLastError();   // swallow the no-device error, leave the CUDA state clean
            g_device = -1;
            return;
        }
        g_device = g_rank % nDev;
        cudaSetDevice(g_device);
    }

    // Bootstrap NVSHMEM off the existing MPI communicator so PE ids line up with MPI ranks. The CUDA device
    // must already be selected (bindRankToDevice runs first). No-op unless built with BRAE_WITH_NVSHMEM.
    void bootstrapNvshmem()
    {
#ifdef BRAE_WITH_NVSHMEM
        if (g_device < 0) return;                       // no GPU -> no device transport
        MPI_Comm comm = MPI_COMM_WORLD;
        nvshmemx_init_attr_t attr = NVSHMEMX_INIT_ATTR_INITIALIZER;
        nvshmemx_set_attr_mpi_comm_args(&comm, &attr);
        // Call the exported host entry directly: the inline nvshmemx_init_attr() resolves nvshmemi_init_thread
        // from the device static lib (needs -rdc); nvshmemx_hostlib_init_attr is the host .so's real symbol.
        nvshmemx_hostlib_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr);
        g_nvshmem = true;
#endif
    }

    inline MPI_Op mpiOp(ReduceOp op)
    {
        switch (op)
        {
            case ReduceOp::Sum: return MPI_SUM;
            case ReduceOp::Min: return MPI_MIN;
            case ReduceOp::Max: return MPI_MAX;
        }
        return MPI_SUM;
    }
} // namespace

void Pstream::init(int& argc, char**& argv)
{
    int already = 0;
    MPI_Initialized(&already);
    if (!already)
        MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nprocs);
    bindRankToDevice();
    bootstrapNvshmem();
    g_init = true;
}

void Pstream::finalize()
{
    if (!g_requests.empty())
        waitAll();
#ifdef BRAE_WITH_NVSHMEM
    if (g_nvshmem)                       // before MPI_Finalize: NVSHMEM uses MPI
    {
        nvshmemx_hostlib_finalize();
        g_nvshmem = false;
    }
#endif
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized)
        MPI_Finalize();
    g_init = false;
}

bool Pstream::initialized() { return g_init; }
int  Pstream::myProcNo()    { return g_rank; }
int  Pstream::nProcs()      { return g_nprocs; }
bool Pstream::parRun()      { return g_nprocs > 1; }
bool Pstream::master()      { return g_rank == 0; }
int  Pstream::deviceId()    { return g_device; }
bool Pstream::nvshmemActive() { return g_nvshmem; }

void Pstream::isend(const scalar* data, int count, int toProc, int tag)
{
    MPI_Request r;
    MPI_Isend(data, count, MPI_DOUBLE, toProc, tag, MPI_COMM_WORLD, &r);
    g_requests.push_back(r);
}

void Pstream::irecv(scalar* buf, int count, int fromProc, int tag)
{
    MPI_Request r;
    MPI_Irecv(buf, count, MPI_DOUBLE, fromProc, tag, MPI_COMM_WORLD, &r);
    g_requests.push_back(r);
}

void Pstream::waitAll()
{
    if (g_requests.empty()) return;
    MPI_Waitall(static_cast<int>(g_requests.size()),
                g_requests.data(), MPI_STATUSES_IGNORE);
    g_requests.clear();
}


scalar Pstream::allReduce(scalar v, ReduceOp op)
{
    scalar out = v;
    MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, mpiOp(op), MPI_COMM_WORLD);
    return out;
}

label Pstream::allReduce(label v, ReduceOp op)
{
    label out = v;
    MPI_Allreduce(&v, &out, 1, MPI_INT32_T, mpiOp(op), MPI_COMM_WORLD);
    return out;
}

void Pstream::allReduce(scalar* data, int count, ReduceOp op)
{
    MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_DOUBLE, mpiOp(op), MPI_COMM_WORLD);
}

void Pstream::broadcast(label* data, int count, int root)
{
    MPI_Bcast(data, count, MPI_INT32_T, root, MPI_COMM_WORLD);
}

void Pstream::barrier() { MPI_Barrier(MPI_COMM_WORLD); }

} // namespace brae
