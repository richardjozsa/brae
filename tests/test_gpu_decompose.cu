// Phase 0b gate: per-rank DeviceMesh upload + device-path reconstruction.
//
// Decompose pitzDaily; each rank builds ONLY its partition's DeviceMesh (nCells == its local cells), uploads
// a device field psi = global cell index, reads it back (D2H), and reconstructs the global field via
// cellProcAddr. The reconstructed field must equal the serial [0..nCells) exactly, and each rank's device
// mesh must hold only its partition (the measurable OOM-relief signal: maxLocalCells < nCells for nproc>1).
//
// Run: mpirun -np {1,2,4,8} test_gpu_decompose <constant/polyMesh>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "scotch_decomposition.cuh"
#include "local_mesh.cuh"
#include "device_mesh.cuh"
#include "device_buffer.cuh"
#include "reconstruct.cuh"
#include "cf_pstream.cuh"

#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv)
{
    Pstream::init(argc, argv);
    const int rank  = Pstream::myProcNo();
    const int nproc = Pstream::nProcs();
    if (argc < 2)
    {
        if (Pstream::master()) std::printf("usage: %s <polyMeshDir>\n", argv[0]);
        Pstream::finalize();
        return 2;
    }

    PrimitiveMesh m;
    m.read(argv[1]);
    const label nC = m.nCells();

    // Decompose once on master, broadcast the map (deterministic).
    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(m, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);

    // This rank's partition ONLY: local mesh -> geometry -> device mesh.
    const LocalMesh lm = buildLocalMesh(m, cellToPart, rank);
    FvGeometry g;
    g.build(lm.mesh);
    const std::vector<FvPatch> fvp = buildPatches(lm.mesh, g);
    const DeviceMesh dm = buildDeviceMesh(lm.mesh, g, fvp);

    int fails = 0;
    const label nLocal = static_cast<label>(lm.cellProcAddr.size());

    // (1) the device mesh holds only this partition.
    if (dm.nCells != nLocal)
    {
        std::printf("[rank %d] FAIL: dm.nCells=%d != nLocalCells=%d\n", rank, dm.nCells, (int)nLocal);
        ++fails;
    }

    // (2) device field round-trip: psi = global cell index, H2D then D2H.
    std::vector<scalar> psiLocal(nLocal);
    for (label i = 0; i < nLocal; ++i) psiLocal[i] = static_cast<scalar>(lm.cellProcAddr[i]);
    DeviceBuffer<scalar> psi_d(psiLocal);
    const std::vector<scalar> back = psi_d.host();
    for (label i = 0; i < nLocal; ++i)
        if (back[i] != psiLocal[i])
        {
            if (fails < 5) std::printf("[rank %d] FAIL roundtrip i=%d got %.1f exp %.1f\n", rank, (int)i, back[i], psiLocal[i]);
            ++fails;
        }

    // (3) reconstruction: gather to the global numbering, must equal serial [0..nCells).
    const std::vector<scalar> global = reconstructField(lm.cellProcAddr, back, nC);
    if (Pstream::master())
        for (label c = 0; c < nC; ++c)
            if (global[c] != static_cast<scalar>(c))
            {
                if (fails < 5) std::printf("FAIL reconstruct c=%d got %.1f\n", (int)c, global[c]);
                ++fails;
            }

    const label totalFails = Pstream::allReduce((label)fails, ReduceOp::Sum);
    const label maxLocal   = Pstream::allReduce(nLocal, ReduceOp::Max);
    if (Pstream::master())
        std::printf("test_gpu_decompose: nproc=%d cells=%d maxLocalCells=%d  %s (fails=%d)\n",
                    nproc, (int)nC, (int)maxLocal, totalFails == 0 ? "PASS" : "FAIL", (int)totalFails);

    Pstream::finalize();
    return totalFails == 0 ? 0 : 1;
}
