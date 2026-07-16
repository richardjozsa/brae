// Phase 1b gate: the device halo exchange (NVSHMEM) reproduces the host test_interface_exchange oracle.
//
// Same 1-D chain: cells split contiguously across ranks, psi[localCell] = its GLOBAL index. At each rank
// boundary the former internal face is a processor interface; the device exchange must deliver the far-side
// global index, and the coupled-interface stencil (updateInterfaceMatrix) must reproduce the +/-1 interior
// step -- bit-for-bit the same values the host MPI ProcessorInterface produces.
//
// Run: mpirun -np {1,2,4,8} test_gpu_interface_exchange
#include "cf_pstream.cuh"
#include "device_halo.cuh"
#include "device_buffer.cuh"

#include <cstdio>
#include <vector>

using namespace brae;

int main(int argc, char** argv)
{
    Pstream::init(argc, argv);
    const int rank  = Pstream::myProcNo();
    const int nproc = Pstream::nProcs();

    if (!Pstream::nvshmemActive())
    {
        if (Pstream::master()) std::printf("test_gpu_interface_exchange: NVSHMEM inactive -- SKIP\n");
        Pstream::finalize();
        return 0;
    }

    const label M = 10;                               // cells per rank
    std::vector<scalar> psiH(M);
    for (label i = 0; i < M; ++i) psiH[i] = static_cast<scalar>(rank * M + i);   // known field = global index
    DeviceBuffer<scalar> psi(psiH);

    int failures = 0;
    {   // scope: DeviceHalo's symmetric buffers must be freed before Pstream::finalize()
        std::vector<int>                nbr;
        std::vector<std::vector<label>> fc;
        int leftIdx = -1, rightIdx = -1;
        if (rank > 0)         { leftIdx  = static_cast<int>(nbr.size()); nbr.push_back(rank - 1); fc.push_back({0});     }
        if (rank < nproc - 1) { rightIdx = static_cast<int>(nbr.size()); nbr.push_back(rank + 1); fc.push_back({M - 1}); }

        DeviceHalo halo(rank, nbr, fc);
        halo.exchange(psi.data());
        cudaStreamSynchronize(0);

        // (a) ghost value == the true global index of the neighbour-rank boundary cell.
        if (leftIdx >= 0)
        {
            const scalar got = halo.neighbourField(leftIdx)[0];
            const scalar exp = static_cast<scalar>(rank * M - 1);
            if (got != exp) { std::printf("[rank %d] FAIL left: got %.1f exp %.1f\n", rank, got, exp); ++failures; }
        }
        if (rightIdx >= 0)
        {
            const scalar got = halo.neighbourField(rightIdx)[0];
            const scalar exp = static_cast<scalar>(rank * M + M);
            if (got != exp) { std::printf("[rank %d] FAIL right: got %.1f exp %.1f\n", rank, got, exp); ++failures; }
        }

        // (b) the three-call contract: a Laplacian-like stencil sum_nbr(psiNbr - psi_self) at the boundary
        //     cells must equal the analytic interior value (+/-1). updateInterfaceMatrix does
        //     result -= coeff*psiNbr with coeff=-1 (i.e. += psiNbr); then subtract psi_self.
        DeviceBuffer<scalar> lap(std::vector<scalar>(M, 0.0));
        DeviceBuffer<scalar> coeff(std::vector<scalar>{-1.0});
        for (int i = 0; i < halo.nInterfaces(); ++i)
            halo.updateInterfaceMatrix(i, lap.data(), coeff.data());
        cudaStreamSynchronize(0);
        std::vector<scalar> lapH = lap.host();
        if (leftIdx  >= 0) lapH[0]     -= psiH[0];
        if (rightIdx >= 0) lapH[M - 1] -= psiH[M - 1];
        if (rank > 0 && lapH[0] != -1.0)
        { std::printf("[rank %d] FAIL stencil[0]: got %.1f exp -1.0\n", rank, lapH[0]); ++failures; }
        if (rank < nproc - 1 && lapH[M - 1] != 1.0)
        { std::printf("[rank %d] FAIL stencil[M-1]: got %.1f exp 1.0\n", rank, lapH[M - 1]); ++failures; }
    }

    const label totalFail = Pstream::allReduce(static_cast<label>(failures), ReduceOp::Sum);
    if (Pstream::master())
        std::printf("test_gpu_interface_exchange: nproc=%d  %s (failures=%d)\n",
                    nproc, totalFail == 0 ? "PASS" : "FAIL", static_cast<int>(totalFail));

    Pstream::finalize();
    return totalFail == 0 ? 0 : 1;
}
