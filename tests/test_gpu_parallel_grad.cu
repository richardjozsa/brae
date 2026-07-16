// Phase 4a foundation: the DEVICE fvc::gaussGrad on a decomposed mesh reproduces the serial gradient, with
// processor faces coupled on-device.
//
// The processor face is treated as a coupled boundary: DeviceHalo exchanges the field and scatterBoundaryValues
// writes the interpolated neighbour value into the boundary array (bval), so the UNCHANGED deviceGaussGrad reads
// the right value there. Real-boundary bval comes from the field's evaluated BCs; the processor slots are zeroed
// and filled on-device, so a correct gradient proves the device coupling. Mirrors host test_parallel_grad.
//
// Run: mpirun -np {1,2,4,8} test_gpu_parallel_grad <caseDir>
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
#include "device_mesh.cuh"
#include "device_halo.cuh"
#include "device_buffer.cuh"
#include "cf_pstream.cuh"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace brae;

int main(int argc, char** argv)
{
    Pstream::init(argc, argv);
    const int rank = Pstream::myProcNo(), nproc = Pstream::nProcs();
    if (argc < 2)
    {
        if (Pstream::master()) std::printf("usage: %s <caseDir>\n", argv[0]);
        Pstream::finalize();
        return 2;
    }
    if (!Pstream::nvshmemActive())
    {
        if (Pstream::master()) std::printf("test_gpu_parallel_grad: NVSHMEM inactive -- SKIP\n");
        Pstream::finalize();
        return 0;
    }
    const std::string caseDir = argv[1];

    // Global mesh + serial gradient reference.
    PrimitiveMesh gm;  gm.read(caseDir + "/constant/polyMesh");
    FvGeometry gg;     gg.build(gm);
    const std::vector<FvPatch> gpatches = buildPatches(gm, gg);
    const label nC = gm.nCells();

    const FieldData<scalar> pfd = readField<scalar>(caseDir + "/282/p");
    GeometricField<scalar> pG = buildField<scalar>(pfd, gpatches, nC);
    pG.evaluateBoundary();
    const std::vector<vector> gradGlobal = fvc::gaussGrad(pG, gm, gg, gpatches);

    // Decompose; build this partition's local mesh, geometry and interpolation weights.
    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    LocalMesh lm = buildLocalMesh(gm, cellToPart, rank);
    FvGeometry lg;  lg.build(lm.mesh);
    const std::vector<FvPatch> lpatches = buildPatches(lm.mesh, lg);
    const std::vector<std::vector<scalar>> procW = computeProcWeights(lm, lg, lpatches);

    GeometricField<scalar> pL = distributeField<scalar>(pfd, gm.patches(), lm, lpatches, procW, rank);
    pL.evaluateBoundary();

    // Real-boundary bval as usual; processor patches zeroed here and filled on-device below.
    std::vector<std::vector<scalar>> pbnd;
    std::vector<label> procStart;
    label bidx = 0;
    for (std::size_t k = 0; k < lpatches.size(); ++k)
    {
        if (lpatches[k].type == "cyclic" || lpatches[k].type == "cyclicAMI") continue;   // buildDeviceMesh skips
        if (lpatches[k].type == "processor")
        {
            procStart.push_back(bidx);
            pbnd.push_back(std::vector<scalar>(lpatches[k].size, 0.0));
        }
        else
            pbnd.push_back(pL.boundary[k]->value());
        bidx += lpatches[k].size;
    }

    scalar gMax = 0, gMag = 0;
    {   // scope: DeviceHalo symmetric buffers freed before Pstream::finalize()
        const DeviceMesh dm = buildDeviceMesh(lm.mesh, lg, lpatches);
        DeviceBuffer<scalar> psi_d(pL.internal);
        DeviceBuffer<scalar> bval_d(flattenBoundary(pbnd));

        std::vector<int> nbrParts;
        for (int q : lm.procNbr) nbrParts.push_back(q);
        DeviceHalo halo(rank, nbrParts, lm.procFaceCells);

        std::vector<DeviceBuffer<scalar>> weightsD(procW.size());
        for (std::size_t i = 0; i < procW.size(); ++i) weightsD[i].copyFrom(procW[i]);

        // Exchange the field, then inject the interpolated processor-face boundary values.
        halo.exchange(psi_d.data());
        halo.scatterBoundaryValues(psi_d.data(), weightsD, procStart, bval_d.data());

        DeviceBuffer<scalar> gx, gy, gz;
        deviceGaussGrad(dm, psi_d, bval_d, gx, gy, gz);
        const std::vector<scalar> gxH = gx.host(), gyH = gy.host(), gzH = gz.host();

        scalar maxAbs = 0, mag = 0;
        for (label lc = 0; lc < lm.mesh.nCells(); ++lc)
        {
            const label gc = lm.cellProcAddr[lc];
            const vector gLoc{gxH[lc], gyH[lc], gzH[lc]};
            maxAbs = std::fmax(maxAbs, brae::mag(gLoc - gradGlobal[gc]));
            mag    = std::fmax(mag, brae::mag(gradGlobal[gc]));
        }
        gMax = Pstream::allReduce(maxAbs, ReduceOp::Max);
        gMag = Pstream::allReduce(mag, ReduceOp::Max);
    }

    const scalar rel = gMag > 0 ? gMax / gMag : gMax;
    const bool pass = rel < 1e-11;
    if (Pstream::master())
    {
        std::printf("test_gpu_parallel_grad np=%d: device gaussGrad vs serial rel=%.3e\n", nproc, rel);
        std::printf("%s\n", pass ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return pass ? 0 : 1;
}
