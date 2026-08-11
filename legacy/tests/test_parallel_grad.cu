// lduInterface increment 4d: processor-aware fvc. A field distributed to the local mesh, with its
// processor patches carrying interpolation weights, runs the UNCHANGED serial fvc::gaussGrad and
// reproduces the global gradient at every local cell. This is the proof that the existing fvc
// operators work on a decomposed mesh. Run: mpirun -np {1,2,4,8} test_parallel_grad <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvc.cuh"
#include "scotch_decomposition.cuh"
#include "local_mesh.cuh"
#include "local_assembly.cuh"
#include "field_distribute.cuh"
#include "cf_pstream.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv) {
    Pstream::init(argc, argv);
    const int rank = Pstream::myProcNo(), nproc = Pstream::nProcs();
    if (argc < 2) { if (Pstream::master()) std::printf("usage: %s <caseDir>\n", argv[0]); Pstream::finalize(); return 2; }
    const std::string caseDir = argv[1];

    PrimitiveMesh gm;  gm.read(caseDir + "/constant/polyMesh");
    FvGeometry gg;     gg.build(gm);
    const std::vector<FvPatch> gpatches = buildPatches(gm, gg);
    const label nC = gm.nCells();

    const FieldData<scalar> pfd = readField<scalar>(caseDir + "/282/p");
    GeometricField<scalar> pG = buildField<scalar>(pfd, gpatches, nC);  pG.evaluateBoundary();
    const std::vector<vector> gradGlobal = fvc::gaussGrad(pG, gm, gg, gpatches);

    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    LocalMesh Lm = buildLocalMesh(gm, cellToPart, rank);
    FvGeometry lg; lg.build(Lm.mesh);
    const std::vector<FvPatch> lpatches = buildPatches(Lm.mesh, lg);
    const auto procW = computeProcWeights(Lm, lg, lpatches);

    GeometricField<scalar> pL = distributeField<scalar>(pfd, gm.patches(), Lm, lpatches, procW, rank);
    pL.evaluateBoundary();   // exchange halos + interpolate processor faces
    const std::vector<vector> gradLocal = fvc::gaussGrad(pL, Lm.mesh, lg, lpatches);

    scalar maxAbs = 0, mag = 0;
    for (label lc = 0; lc < Lm.mesh.nCells(); ++lc) {
        const label gc = Lm.cellProcAddr[lc];
        maxAbs = std::fmax(maxAbs, brae::mag(gradLocal[lc] - gradGlobal[gc]));
        mag    = std::fmax(mag, brae::mag(gradGlobal[gc]));
    }
    const scalar gMax = Pstream::allReduce(maxAbs, ReduceOp::Max);
    const scalar gMag = Pstream::allReduce(mag, ReduceOp::Max);

    if (Pstream::master()) {
        const scalar rel = gMag > 0 ? gMax / gMag : gMax;
        std::printf("test_parallel_grad np=%d: local fvc::gaussGrad vs serial rel=%.3e\n", nproc, rel);
        std::printf("%s\n", rel < 1e-12 ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return 0;
}
