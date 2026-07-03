// lduInterface increment 4b: processor-face cell-to-cell delta coefficients (exchanged) match the
// global internal-face deltaCoeffs of the same cut faces. This is the geometric crux of local
// matrix assembly: with it, a locally-assembled laplacian/convection coeff (gamma*magSf*procDelta)
// equals the global cut-face coeff, so the local assembly reproduces the serial matrix.
// Run: mpirun -np {1,2,4,8} test_proc_delta <polyMeshDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "scotch_decomposition.cuh"
#include "local_mesh.cuh"
#include "local_assembly.cuh"
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
    const std::vector<FvPatch> lpatches = buildPatches(L.mesh, lg);

    const std::vector<std::vector<scalar>> pdc = computeProcDeltaCoeffs(L, lg, lpatches);

    // Compare each processor face's exchanged delta to the global internal-face delta of the same
    // global cut face.
    scalar maxAbs = 0, mag = 0; label nFaces = 0;
    for (std::size_t j = 0; j < pdc.size(); ++j)
        for (std::size_t i = 0; i < pdc[j].size(); ++i) {
            const label gf = L.procGlobalFaces[j][i];        // a global internal (cut) face
            const scalar gd = gg.deltaCoeffs()[gf];
            maxAbs = std::fmax(maxAbs, std::fabs(pdc[j][i] - gd));  mag = std::fmax(mag, std::fabs(gd));
            ++nFaces;
        }
    const scalar gMax = Pstream::allReduce(maxAbs, ReduceOp::Max);
    const scalar gMag = Pstream::allReduce(mag, ReduceOp::Max);
    const label  gFaces = Pstream::allReduce(nFaces, ReduceOp::Sum);

    if (Pstream::master()) {
        const scalar rel = gMag > 0 ? gMax / gMag : gMax;
        std::printf("test_proc_delta np=%d: procDeltaCoeffs vs global cut-face delta rel=%.3e  (%d proc faces)\n",
                    nproc, rel, (int)gFaces);
        std::printf("%s\n", rel < 1e-12 ? "PASS" : (nproc == 1 ? "PASS" : "FAIL"));
    }
    Pstream::finalize();
    return 0;
}
