// brae::Pstream MPI backend. Re-expressed from OpenFOAM src/Pstream/mpi semantics
// (non-blocking exchange + Allreduce); not copied verbatim.
#include "cf_pstream.cuh"

#include <mpi.h>
#include <vector>

namespace brae {
namespace {
    bool                     g_init    = false;
    int                      g_rank    = 0;
    int                      g_nprocs  = 1;
    std::vector<MPI_Request> g_requests;   // pending isend/irecv

    inline MPI_Op mpiOp(ReduceOp op) {
        switch (op) {
            case ReduceOp::Sum: return MPI_SUM;
            case ReduceOp::Min: return MPI_MIN;
            case ReduceOp::Max: return MPI_MAX;
        }
        return MPI_SUM;
    }
}

void Pstream::init(int& argc, char**& argv) {
    int already = 0;
    MPI_Initialized(&already);
    if (!already) MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_nprocs);
    g_init = true;
}

void Pstream::finalize() {
    if (!g_requests.empty()) waitAll();
    int finalized = 0;
    MPI_Finalized(&finalized);
    if (!finalized) MPI_Finalize();
    g_init = false;
}

bool Pstream::initialized() { return g_init; }
int  Pstream::myProcNo()    { return g_rank; }
int  Pstream::nProcs()      { return g_nprocs; }
bool Pstream::parRun()      { return g_nprocs > 1; }
bool Pstream::master()      { return g_rank == 0; }

void Pstream::isend(const scalar* data, int count, int toProc, int tag) {
    MPI_Request r;
    MPI_Isend(data, count, MPI_DOUBLE, toProc, tag, MPI_COMM_WORLD, &r);
    g_requests.push_back(r);
}

void Pstream::irecv(scalar* buf, int count, int fromProc, int tag) {
    MPI_Request r;
    MPI_Irecv(buf, count, MPI_DOUBLE, fromProc, tag, MPI_COMM_WORLD, &r);
    g_requests.push_back(r);
}

void Pstream::waitAll() {
    if (g_requests.empty()) return;
    MPI_Waitall(static_cast<int>(g_requests.size()),
                g_requests.data(), MPI_STATUSES_IGNORE);
    g_requests.clear();
}

scalar Pstream::allReduce(scalar v, ReduceOp op) {
    scalar out = v;
    MPI_Allreduce(&v, &out, 1, MPI_DOUBLE, mpiOp(op), MPI_COMM_WORLD);
    return out;
}

label Pstream::allReduce(label v, ReduceOp op) {
    label out = v;
    MPI_Allreduce(&v, &out, 1, MPI_INT32_T, mpiOp(op), MPI_COMM_WORLD);
    return out;
}

void Pstream::allReduce(scalar* data, int count, ReduceOp op) {
    MPI_Allreduce(MPI_IN_PLACE, data, count, MPI_DOUBLE, mpiOp(op), MPI_COMM_WORLD);
}

void Pstream::broadcast(label* data, int count, int root) {
    MPI_Bcast(data, count, MPI_INT32_T, root, MPI_COMM_WORLD);
}

void Pstream::barrier() { MPI_Barrier(MPI_COMM_WORLD); }

} // namespace brae
