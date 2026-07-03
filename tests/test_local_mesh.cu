// lduInterface increment 4a: decomposePar local-mesh construction. Build each partition's local
// PrimitiveMesh, compute its FvGeometry, and verify the local cell centres/volumes match the
// global cells they map to (cellProcAddressing) -- i.e. the renumbered topology is geometrically
// identical. Also checks the processor-patch face count == the global cut faces.
// Run: mpirun -np {1,2,4,8} test_local_mesh <polyMeshDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "scotch_decomposition.cuh"
#include "local_mesh.cuh"
#include "cf_pstream.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    Pstream::init(argc, argv);
    const int rank = Pstream::myProcNo(), nproc = Pstream::nProcs();
    if (argc < 2) { if (Pstream::master()) std::printf("usage: %s <polyMeshDir>\n", argv[0]); Pstream::finalize(); return 2; }

    PrimitiveMesh gm;  gm.read(argv[1]);
    FvGeometry gg;     gg.build(gm);
    const label nC = gm.nCells();

    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);

    LocalMesh L = buildLocalMesh(gm, cellToPart, rank);
    FvGeometry lg; lg.build(L.mesh);

    // Local geometry must match the global cells (renumber-invariant).
    scalar maxV = 0, magV = 0, maxC = 0;
    for (label lc = 0; lc < L.mesh.nCells(); ++lc) {
        const label gc = L.cellProcAddr[lc];
        maxV = std::fmax(maxV, std::fabs(lg.V()[lc] - gg.V()[gc]));  magV = std::fmax(magV, gg.V()[gc]);
        maxC = std::fmax(maxC, mag(lg.C()[lc] - gg.C()[gc]));
    }
    const scalar gV = Pstream::allReduce(maxV, ReduceOp::Max) / Pstream::allReduce(magV, ReduceOp::Max);
    const scalar gCm = Pstream::allReduce(maxC, ReduceOp::Max);

    // Cell + cut-face accounting.
    const label sumCells = Pstream::allReduce((label)L.mesh.nCells(), ReduceOp::Sum);
    label procFaces = 0; for (const auto& fc : L.procFaceCells) procFaces += (label)fc.size();
    const label sumProcFaces = Pstream::allReduce(procFaces, ReduceOp::Sum);
    label nCut = 0; for (label f = 0; f < gm.nInternalFaces(); ++f) if (cellToPart[gm.owner()[f]] != cellToPart[gm.neighbour()[f]]) ++nCut;

    if (Pstream::master()) {
        std::printf("test_local_mesh np=%d: V rel=%.3e  C maxAbs=%.3e  (Scells=%d/%d  procFaces=%d exp=%d)\n",
                    nproc, gV, gCm, (int)sumCells, (int)nC, (int)sumProcFaces, (int)(2 * nCut));
        const bool ok = gV < 1e-12 && gCm < 1e-9 && sumCells == nC && sumProcFaces == 2 * nCut;
        std::printf("%s\n", ok ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return 0;
}
