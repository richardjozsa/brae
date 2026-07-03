// lduInterface increment 4d: processor-face interpolation (the heart of processor-aware fvc). Each
// rank distributes a field, exchanges the halo, and forms the cut-face value w*local + (1-w)*remote;
// it must equal the global linear interpolation w_g*p_own + (1-w_g)*p_nei at that face. With this,
// fvc::interpolate/grad/flux at processor faces reproduce the serial operators.
// Run: mpirun -np {1,2,4,8} test_proc_interp <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "scotch_decomposition.cuh"
#include "local_mesh.cuh"
#include "local_assembly.cuh"
#include "processor_interface.cuh"
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
    const label nC = gm.nCells(), nIf = gm.nInternalFaces();
    const std::vector<label>& gOwn = gm.owner();
    const std::vector<label>& gNei = gm.neighbour();

    const std::vector<scalar> pG = readField<scalar>(caseDir + "/282/p").internalField;
    // Global linear face value at every internal face.
    std::vector<scalar> pfGlobal(nIf);
    for (label f = 0; f < nIf; ++f) pfGlobal[f] = gg.weights()[f] * pG[gOwn[f]] + (1.0 - gg.weights()[f]) * pG[gNei[f]];

    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    LocalMesh Lm = buildLocalMesh(gm, cellToPart, rank);
    FvGeometry lg; lg.build(Lm.mesh);
    const std::vector<FvPatch> lpatches = buildPatches(Lm.mesh, lg);
    const auto procW = computeProcWeights(Lm, lg, lpatches);

    // Local field + halo exchange.
    std::vector<scalar> pL(Lm.mesh.nCells());
    for (label lc = 0; lc < Lm.mesh.nCells(); ++lc) pL[lc] = pG[Lm.cellProcAddr[lc]];
    std::vector<ProcessorInterface> interfaces;
    for (std::size_t j = 0; j < Lm.procNbr.size(); ++j) interfaces.emplace_back(rank, Lm.procNbr[j], Lm.procFaceCells[j]);
    for (auto& it : interfaces) it.initInterfaceMatrixUpdate(pL.data());
    Pstream::waitAll();

    // Interpolated processor-face value vs the global cut-face value.
    scalar maxAbs = 0, mag = 0; label nF = 0;
    for (std::size_t j = 0; j < interfaces.size(); ++j) {
        const std::vector<scalar>& halo = interfaces[j].neighbourField();
        for (label i = 0; i < (label)halo.size(); ++i) {
            const scalar pfLocal = procW[j][i] * pL[interfaces[j].faceCells()[i]] + (1.0 - procW[j][i]) * halo[i];
            const label gf = Lm.procGlobalFaces[j][i];
            maxAbs = std::fmax(maxAbs, std::fabs(pfLocal - pfGlobal[gf]));  mag = std::fmax(mag, std::fabs(pfGlobal[gf]));
            ++nF;
        }
    }
    const scalar gMax = Pstream::allReduce(maxAbs, ReduceOp::Max);
    const scalar gMag = Pstream::allReduce(mag, ReduceOp::Max);
    const label  gF   = Pstream::allReduce(nF, ReduceOp::Sum);

    if (Pstream::master()) {
        const scalar rel = gMag > 0 ? gMax / gMag : gMax;
        std::printf("test_proc_interp np=%d: processor-face interp vs global rel=%.3e  (%d faces)\n", nproc, rel, (int)gF);
        std::printf("%s\n", rel < 1e-12 ? "PASS" : (nproc == 1 ? "PASS" : "FAIL"));
    }
    Pstream::finalize();
    return 0;
}
