// Phase 1 gate §6.4, ghost-cell exchange == internal neighbour.
//
// A 1-D chain of cells is split contiguously across ranks; psi[cell] = its GLOBAL cell
// index (a known field). At each rank boundary the former internal face becomes a
// processor interface. The exchange must deliver the neighbour's boundary global index ,
// exactly the value an internal face would have provided in the undecomposed mesh.
//
// This is the unit the GPU DSM exchange (Phase G0) must later match bit-for-bit.
// Run: mpirun -np {1,2,4,8} test_interface_exchange
#include "cf_pstream.cuh"
#include "processor_interface.cuh"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    Pstream::init(argc, argv);
    const int rank  = Pstream::myProcNo();
    const int nproc = Pstream::nProcs();

    const label M = 10;                              // cells per rank
    std::vector<scalar> psi(M);
    for (label i = 0; i < M; ++i)
        psi[i] = static_cast<scalar>(rank * M + i);  // known field = global cell index

    // Processor interfaces at the rank boundaries (1-D chain).
    std::vector<ProcessorInterface> ifaces;
    if (rank > 0)         ifaces.emplace_back(rank, rank - 1, std::vector<label>{0});      // left
    if (rank < nproc - 1) ifaces.emplace_back(rank, rank + 1, std::vector<label>{M - 1});  // right

    // init-all, then a single wait, mirrors lduMatrix init/updateMatrixInterfaces.
    for (auto& it : ifaces) it.initInterfaceMatrixUpdate(psi.data());
    Pstream::waitAll();

    int failures = 0;

    // (a) ghost value == the true global index of the neighbour-rank boundary cell.
    for (auto& it : ifaces) {
        const scalar got = it.neighbourField()[0];
        const scalar expect = (it.neighbProcNo() == rank - 1)
                                  ? static_cast<scalar>(rank * M - 1)   // left neighbour
                                  : static_cast<scalar>(rank * M + M);  // right neighbour
        if (got != expect) {
            std::printf("[rank %d] FAIL iface->%d: got %.1f expect %.1f\n",
                        rank, it.neighbProcNo(), got, expect);
            ++failures;
        }
    }

    // (b) full three-call contract: a Laplacian-like stencil  sum_nbr (psiNbr - psi_self)
    //     at the boundary cells must equal the analytic interior value (= +/-1 step sums).
    //     updateInterfaceMatrix applies result -= coeff*psiNbr with coeff = -1, i.e. += psiNbr;
    //     we then subtract psi_self * (#interface neighbours) to form the stencil.
    std::vector<scalar> lap(M, 0.0);
    const scalar coeff = -1.0;  // so updateInterfaceMatrix does lap[fc] += psiNbr
    for (auto& it : ifaces) it.updateInterfaceMatrix(lap.data(), &coeff);
    for (auto& it : ifaces) {
        const label fc = it.faceCells()[0];
        lap[fc] -= psi[fc];  // one interface-neighbour contribution per boundary face
    }
    // Boundary cell 0 (if rank>0) sees neighbour (g-1): stencil contribution = (g-1) - g = -1.
    // Boundary cell M-1 (if rank<nproc-1) sees (g+1): contribution = (g+1) - g = +1.
    if (rank > 0 && lap[0] != -1.0) {
        std::printf("[rank %d] FAIL stencil[0]: got %.1f expect -1.0\n", rank, lap[0]); ++failures;
    }
    if (rank < nproc - 1 && lap[M - 1] != 1.0) {
        std::printf("[rank %d] FAIL stencil[M-1]: got %.1f expect 1.0\n", rank, lap[M - 1]); ++failures;
    }

    // (c) global reduction: sum over all psi == T(T-1)/2,  T = nproc*M.
    scalar localSum = 0.0; for (scalar v : psi) localSum += v;
    const scalar globalSum = Pstream::allReduce(localSum, ReduceOp::Sum);
    const scalar T = static_cast<scalar>(nproc * M);
    const scalar expectSum = T * (T - 1.0) / 2.0;
    if (std::fabs(globalSum - expectSum) > 1e-9) {
        if (Pstream::master())
            std::printf("FAIL allReduce: got %.1f expect %.1f\n", globalSum, expectSum);
        ++failures;
    }

    const label totalFail = Pstream::allReduce(static_cast<label>(failures), ReduceOp::Sum);
    if (Pstream::master())
        std::printf("test_interface_exchange: nproc=%d  %s (failures=%d)\n",
                    nproc, totalFail == 0 ? "PASS" : "FAIL", static_cast<int>(totalFail));

    Pstream::finalize();
    return totalFail == 0 ? 0 : 1;
}
