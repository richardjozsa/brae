// Phase 4a L5: the DEVICE processor-face flux is CONSERVATIVE and matches the host fvc::flux.
//
// phiHbyA at a processor face = (coupled face value of the vector field) & Sf. The coupled value is injected
// by DeviceHalo::scatterBoundaryValues per component, then deviceBoundaryFlux dots it with Sf.
//
// Conservation is the point: both sides of a cut face must see EQUAL AND OPPOSITE flux, or global mass is not
// conserved and the pressure equation has no solution. It holds because computeProcWeights gives the two sides
// w and 1-w, so both interpolate the SAME face value, and Sf points opposite -> phi_p = -phi_q. This test
// exchanges the computed processor-face flux and asserts phi_local + phi_remote == 0.
//
// Run: mpirun -np {1,2,4,8} test_gpu_parallel_flux <caseDir>
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
#include "device_simple.cuh"
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
        if (Pstream::master()) std::printf("test_gpu_parallel_flux: NVSHMEM inactive -- SKIP\n");
        Pstream::finalize();
        return 0;
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
    const label lnC = lm.mesh.nCells();

    const FieldData<vector> Ufd = readField<vector>(caseDir + "/282/U");
    GeometricField<vector> U = distributeField<vector>(Ufd, gm.patches(), lm, lp, procW, rank);
    U.evaluateBoundary();
    const SurfaceScalarField phiHost = fvc::flux(U, lm.mesh, lg, lp);   // host oracle (processor-aware)

    // Per-component boundary arrays: real patches from the evaluated BCs; processor slots zeroed (the device
    // fills them), so a correct flux there proves the device coupling.
    std::vector<std::vector<scalar>> bx, by, bz;
    std::vector<label> procStart;
    label bidx = 0;
    for (std::size_t k = 0; k < lp.size(); ++k)
    {
        if (lp[k].type == "cyclic" || lp[k].type == "cyclicAMI") continue;
        if (lp[k].type == "processor")
        {
            procStart.push_back(bidx);
            bx.push_back(std::vector<scalar>(lp[k].size, 0.0));
            by.push_back(std::vector<scalar>(lp[k].size, 0.0));
            bz.push_back(std::vector<scalar>(lp[k].size, 0.0));
        }
        else
        {
            const std::vector<vector>& bv = U.boundary[k]->value();
            std::vector<scalar> cx(bv.size()), cy(bv.size()), cz(bv.size());
            for (std::size_t i = 0; i < bv.size(); ++i)
            {
                cx[i] = bv[i].x;
                cy[i] = bv[i].y;
                cz[i] = bv[i].z;
            }
            bx.push_back(std::move(cx));
            by.push_back(std::move(cy));
            bz.push_back(std::move(cz));
        }
        bidx += lp[k].size;
    }

    int fails = 0;
    {
        const DeviceMesh dm = buildDeviceMesh(lm.mesh, lg, lp);
        std::vector<scalar> uxH(lnC), uyH(lnC), uzH(lnC);
        for (label c = 0; c < lnC; ++c)
        {
            uxH[c] = U.internal[c].x;
            uyH[c] = U.internal[c].y;
            uzH[c] = U.internal[c].z;
        }
        DeviceBuffer<scalar> ux_d(uxH), uy_d(uyH), uz_d(uzH);
        DeviceBuffer<scalar> bx_d(flattenBoundary(bx)), by_d(flattenBoundary(by)), bz_d(flattenBoundary(bz));

        std::vector<int> nbrParts;
        for (int q : lm.procNbr) nbrParts.push_back(q);
        DeviceHalo halo(rank, nbrParts, lm.procFaceCells);
        std::vector<DeviceBuffer<scalar>> weightsD(procW.size());
        for (std::size_t i = 0; i < procW.size(); ++i) weightsD[i].copyFrom(procW[i]);

        // Inject the coupled processor-face value for each component. The trailing waitExchange() after each
        // scatter is REQUIRED: the halo's recv buffer is reused, so without it a neighbour's next component
        // put can clobber our recv buffer while this scatter is still reading it (see device_halo.cuh).
        halo.exchange(ux_d.data());
        halo.scatterBoundaryValues(ux_d.data(), weightsD, procStart, bx_d.data());
        halo.waitExchange();
        halo.exchange(uy_d.data());
        halo.scatterBoundaryValues(uy_d.data(), weightsD, procStart, by_d.data());
        halo.waitExchange();
        halo.exchange(uz_d.data());
        halo.scatterBoundaryValues(uz_d.data(), weightsD, procStart, bz_d.data());
        halo.waitExchange();

        DeviceBuffer<scalar> phiB;
        deviceBoundaryFlux(dm, bx_d, by_d, bz_d, phiB);
        cudaDeviceSynchronize();
        const std::vector<scalar> phiBH = phiB.host();

        // (a) correctness: the device processor-face flux == the host fvc::flux there.
        {
            std::size_t pj = 0;
            label bi = 0;
            scalar num = 0, den = 0;
            for (std::size_t k = 0; k < lp.size(); ++k)
            {
                if (lp[k].type == "cyclic" || lp[k].type == "cyclicAMI") continue;
                if (lp[k].type == "processor")
                {
                    for (label i = 0; i < lp[k].size; ++i)
                    {
                        num = std::fmax(num, std::fabs(phiBH[bi + i] - phiHost.boundary[k][i]));
                        den = std::fmax(den, std::fabs(phiHost.boundary[k][i]));
                    }
                    ++pj;
                }
                bi += lp[k].size;
            }
            const scalar rel = den > 0 ? num / den : num;
            if (rel > 1e-11)
            {
                std::printf("[rank %d] FAIL proc flux vs host rel=%.3e\n", rank, rel);
                ++fails;
            }
        }

        // (b) CONSERVATION: exchange each processor face's flux; phi_local + phi_remote must be 0.
        {
            std::vector<std::vector<scalar>> sendF(procStart.size()), recvF(procStart.size());
            for (std::size_t j = 0; j < procStart.size(); ++j)
            {
                const label n = static_cast<label>(lm.procFaceCells[j].size());
                sendF[j].assign(phiBH.begin() + procStart[j], phiBH.begin() + procStart[j] + n);
                recvF[j].assign(n, 0.0);
                Pstream::irecv(recvF[j].data(), static_cast<int>(n), lm.procNbr[j], 7);
                Pstream::isend(sendF[j].data(), static_cast<int>(n), lm.procNbr[j], 7);
            }
            Pstream::waitAll();
            scalar worst = 0, scale = 0;
            for (std::size_t j = 0; j < procStart.size(); ++j)
                for (std::size_t i = 0; i < sendF[j].size(); ++i)
                {
                    worst = std::fmax(worst, std::fabs(sendF[j][i] + recvF[j][i]));   // must cancel
                    scale = std::fmax(scale, std::fabs(sendF[j][i]));
                }
            const scalar gWorst = Pstream::allReduce(worst, ReduceOp::Max);
            const scalar gScale = Pstream::allReduce(scale, ReduceOp::Max);
            const scalar rel = gScale > 0 ? gWorst / gScale : gWorst;
            if (rel > 1e-11)
            {
                if (Pstream::master()) std::printf("FAIL flux conservation rel=%.3e\n", rel);
                ++fails;
            }
            else if (Pstream::master())
                std::printf("  cut-face flux cancellation: rel=%.3e (conservative)\n", rel);
        }
    }

    const label totalFail = Pstream::allReduce(static_cast<label>(fails), ReduceOp::Sum);
    if (Pstream::master())
        std::printf("test_gpu_parallel_flux: nproc=%d  %s (failures=%d)\n",
                    nproc, totalFail == 0 ? "PASS" : "FAIL", static_cast<int>(totalFail));

    Pstream::finalize();
    return totalFail == 0 ? 0 : 1;
}
