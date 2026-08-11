// Phase 4a L8 (blocker fix): a processor face must contribute ZERO matrix coefficients on the DEVICE.
//
// This is the invariant the whole distributed loop rests on. Host ProcessorFvPatchField overrides all four
// coeff hooks to zero ("a processor patch contributes NOTHING to the matrix as a boundary; its coupling is
// added as an interface separately"). On the device, ProcessorFvPatchField does NOT override bcCategory(), so
// it reports 0 (= zeroGradient) whose valueInternalCoeffs is 1 -- which would make deviceBCDivCoeffs emit
// iC = phi at every cut face and DOUBLE-COUNT the interface diagonal that deviceMomentumInterface already
// supplies. The fix maps "processor" patches to bcType 8 (coupled). This test pins that down:
//   - db.bcType == 8 on every processor face
//   - deviceBCDivCoeffs  -> iC == 0 and bC == 0 there  (the branch that was wrong)
//   - deviceBCLaplacianCoeffsFace -> iC == 0 and bC == 0 there
//   - and the host's own internalCoeffs are zero there too (the two paths now agree)
//
// Run: mpirun -np {1,2,4,8} test_gpu_parallel_bc <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "fvc.cuh"
#include "scotch_decomposition.cuh"
#include "local_mesh.cuh"
#include "local_assembly.cuh"
#include "field_distribute.cuh"
#include "device_mesh.cuh"
#include "device_boundary.cuh"
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
    const std::string caseDir = argv[1];

    PrimitiveMesh gm;  gm.read(caseDir + "/constant/polyMesh");
    FvGeometry gg;     gg.build(gm);
    const std::vector<FvPatch> gpatches = buildPatches(gm, gg);
    const label nC = gm.nCells();

    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    LocalMesh lm = buildLocalMesh(gm, cellToPart, rank);
    FvGeometry lg;  lg.build(lm.mesh);
    const std::vector<FvPatch> lp = buildPatches(lm.mesh, lg);
    const std::vector<std::vector<scalar>> procW = computeProcWeights(lm, lg, lp);

    // A distributed pressure field: its processor patches are ProcessorFvPatchFields.
    const FieldData<scalar> pfd = readField<scalar>(caseDir + "/282/p");
    GeometricField<scalar> p = distributeField<scalar>(pfd, gm.patches(), lm, lp, procW, rank);
    p.evaluateBoundary();
    // A boundary flux for the div coeffs (non-zero at processor faces -> a wrong iC would be visible).
    const FieldData<vector> Ufd = readField<vector>(caseDir + "/282/U");
    GeometricField<vector> U = distributeField<vector>(Ufd, gm.patches(), lm, lp, procW, rank);
    U.evaluateBoundary();
    const SurfaceScalarField phi = fvc::flux(U, lm.mesh, lg, lp);

    int fails = 0;
    {
        const DeviceBoundary db = buildDeviceBoundary(p, lp, lg);

        // Flatten the boundary flux + a gamma in the same order the device gather uses.
        std::vector<scalar> phiBH, gammaBH;
        std::vector<label>  isProc;
        for (std::size_t k = 0; k < lp.size(); ++k)
        {
            if (lp[k].type == "cyclic" || lp[k].type == "cyclicAMI") continue;
            const bool proc = (lp[k].type == "processor");
            for (label i = 0; i < lp[k].size; ++i)
            {
                phiBH.push_back(phi.boundary[k][i]);
                gammaBH.push_back(1e-3);
                isProc.push_back(proc ? 1 : 0);
            }
        }
        DeviceBuffer<scalar> phiB(phiBH), gammaB(gammaBH);

        // (a) processor faces must be typed COUPLED (8).
        const std::vector<label> ty = db.bcType.host();
        label nProcFaces = 0;
        for (std::size_t i = 0; i < isProc.size(); ++i)
        {
            if (!isProc[i]) continue;
            ++nProcFaces;
            if (ty[i] != 8)
            {
                if (fails < 3) std::printf("[rank %d] FAIL bcType at proc face %zu: got %d expect 8\n", rank, i, (int)ty[i]);
                ++fails;
            }
        }

        // (b) the div coeffs -- the branch that was wrong -- must be zero there.
        DeviceBuffer<scalar> iC, bC;
        deviceBCDivCoeffs(db, phiB, iC, bC);
        const std::vector<scalar> iCH = iC.host(), bCH = bC.host();
        // (c) the laplacian face coeffs must be zero there too.
        DeviceBuffer<scalar> liC, lbC;
        deviceBCLaplacianCoeffsFace(db, gammaB, liC, lbC);
        const std::vector<scalar> liCH = liC.host(), lbCH = lbC.host();

        scalar worstDiv = 0, worstLap = 0, phiScale = 0;
        for (std::size_t i = 0; i < isProc.size(); ++i)
        {
            if (!isProc[i]) continue;
            worstDiv = std::fmax(worstDiv, std::fmax(std::fabs(iCH[i]), std::fabs(bCH[i])));
            worstLap = std::fmax(worstLap, std::fmax(std::fabs(liCH[i]), std::fabs(lbCH[i])));
            phiScale = std::fmax(phiScale, std::fabs(phiBH[i]));
        }
        if (worstDiv != 0.0)
        {
            std::printf("[rank %d] FAIL div coeffs at processor faces: worst |coeff|=%.3e (phi scale %.3e) -- "
                        "the interface diagonal is being double-counted\n", rank, worstDiv, phiScale);
            ++fails;
        }
        if (worstLap != 0.0)
        {
            std::printf("[rank %d] FAIL laplacian coeffs at processor faces: worst |coeff|=%.3e\n", rank, worstLap);
            ++fails;
        }

        // (d) the host agrees: its internalCoeffs are zero on processor patches.
        const FvScalarMatrix Mh = fvm::div(phi.internal, phi.boundary, p, lm.mesh, lp);
        scalar worstHost = 0;
        for (std::size_t k = 0; k < lp.size(); ++k)
        {
            if (lp[k].type != "processor") continue;
            for (label i = 0; i < lp[k].size; ++i)
                worstHost = std::fmax(worstHost, std::fabs(Mh.internalCoeffs[k][i]));
        }
        if (worstHost != 0.0)
        {
            std::printf("[rank %d] FAIL host internalCoeffs non-zero on a processor patch: %.3e\n", rank, worstHost);
            ++fails;
        }

        const label gProcFaces = Pstream::allReduce(nProcFaces, ReduceOp::Sum);
        if (Pstream::master())
            std::printf("  processor faces checked: %d (device+host both zero-coeff)\n", (int)gProcFaces);
    }

    const label totalFail = Pstream::allReduce(static_cast<label>(fails), ReduceOp::Sum);
    if (Pstream::master())
        std::printf("test_gpu_parallel_bc: nproc=%d  %s (failures=%d)\n",
                    nproc, totalFail == 0 ? "PASS" : "FAIL", static_cast<int>(totalFail));

    Pstream::finalize();
    return totalFail == 0 ? 0 : 1;
}
