// Phase 1 gate §6.4 on REAL topology, SCOTCH-decomposed pitzDaily, exchanged over MPI,
// reproduces the serial adjacency.
//
// Decompose on master, broadcast the map (deterministic). Each rank owns one partition and
// builds its processor interfaces. With psi[localCell] = its GLOBAL cell index, the halo
// exchange must deliver, at every interface face, the global index of the cell on the far
// side of that face in the UNDECOMPOSED mesh. Plus reconstruction invariants.
// Run: mpirun -np {2,4} test_decompose_exchange <constant/polyMesh>
#include "primitive_mesh.cuh"
#include "scotch_decomposition.cuh"
#include "domain_decomposition.cuh"
#include "cf_pstream.cuh"

#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    Pstream::init(argc, argv);
    const int rank  = Pstream::myProcNo();
    const int nproc = Pstream::nProcs();
    if (argc < 2) { if (Pstream::master()) std::printf("usage: %s <polyMeshDir>\n", argv[0]); Pstream::finalize(); return 2; }

    PrimitiveMesh m;
    m.read(argv[1]);
    const label nC = m.nCells();

    // Decompose once on master, broadcast to all ranks.
    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(m, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);

    DomainDecomposition dd(m, cellToPart, rank);

    // Known field: psi = global cell index.
    std::vector<scalar> psi(dd.nLocalCells());
    for (label i = 0; i < dd.nLocalCells(); ++i)
        psi[i] = static_cast<scalar>(dd.cellProcAddressing()[i]);

    // Exchange (init all interfaces, single wait).
    for (auto& it : dd.interfaces()) it.initInterfaceMatrixUpdate(psi.data());
    Pstream::waitAll();

    int fails = 0;

    // (a) ghost value at each interface face == far-side global cell index.
    for (std::size_t k = 0; k < dd.interfaces().size(); ++k) {
        const auto& got = dd.interfaces()[k].neighbourField();
        const auto& exp = dd.nbrGlobalCells()[k];
        for (std::size_t i = 0; i < got.size(); ++i) {
            if (got[i] != static_cast<scalar>(exp[i])) {
                if (fails < 5)
                    std::printf("[rank %d] FAIL iface->%d face %zu got %.0f exp %d\n",
                                rank, dd.interfaces()[k].neighbProcNo(), i, got[i], (int)exp[i]);
                ++fails;
            }
        }
    }

    // (b) reconstruction invariants (global).
    const label sumLocalCells = Pstream::allReduce((label)dd.nLocalCells(), ReduceOp::Sum);
    label sumIfaceLocal = 0;
    for (auto& it : dd.interfaces()) sumIfaceLocal += it.size();
    const label sumIfaceAll = Pstream::allReduce(sumIfaceLocal, ReduceOp::Sum);

    label nCut = 0;  // global cut internal faces (every rank can compute it)
    for (label f = 0; f < m.nInternalFaces(); ++f)
        if (cellToPart[m.owner()[f]] != cellToPart[m.neighbour()[f]]) ++nCut;

    if (Pstream::master()) {
        if (sumLocalCells != nC) {
            std::printf("FAIL sum(localCells)=%d != nCells=%d\n", (int)sumLocalCells, (int)nC); ++fails;
        }
        if (sumIfaceAll != 2 * nCut) {  // each cut face appears on both partitions
            std::printf("FAIL sum(ifaceFaces)=%d != 2*nCut=%d\n", (int)sumIfaceAll, (int)(2 * nCut)); ++fails;
        }
    }

    const label totalFails = Pstream::allReduce((label)fails, ReduceOp::Sum);
    if (Pstream::master())
        std::printf("test_decompose_exchange: nproc=%d cells=%d cutFaces=%d  %s (fails=%d)\n",
                    nproc, (int)nC, (int)nCut, totalFails == 0 ? "PASS" : "FAIL", (int)totalFails);

    Pstream::finalize();
    return totalFails == 0 ? 0 : 1;
}
