// Phase 4a L7: the DEVICE processor-face pressure flux (pEqn.flux()) == host parallelMatrixFlux, and cancels
// across a cut face.
//
// flux[f] = -ifCoeff[f]*(p_nbr[f] - p_local[faceCells[f]]). This is the term the SIMPLE corrector subtracts
// from phiHbyA; if it did not cancel across cut faces the corrected phi would not be globally conservative and
// the pressure equation would drift. Both checks here: vs the validated host operator, and equal-and-opposite.
//
// Run: mpirun -np {1,2,4,8} test_gpu_parallel_pflux <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "scotch_decomposition.cuh"
#include "domain_decomposition.cuh"
#include "parallel_amul.cuh"
#include "device_buffer.cuh"
#include "device_halo.cuh"
#include "parallel_device_simple.cuh"
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
        if (Pstream::master()) std::printf("test_gpu_parallel_pflux: NVSHMEM inactive -- SKIP\n");
        Pstream::finalize();
        return 0;
    }
    const std::string caseDir = argv[1];

    PrimitiveMesh gm;  gm.read(caseDir + "/constant/polyMesh");
    FvGeometry gg;     gg.build(gm);
    const std::vector<FvPatch> gpatches = buildPatches(gm, gg);
    const label nC = gm.nCells();

    GeometricField<scalar> pG = buildField<scalar>(readField<scalar>(caseDir + "/282/p"), gpatches, nC);
    pG.evaluateBoundary();
    const FvScalarMatrix M = fvm::laplacian(pG, 1.0, gm, gg, gpatches);

    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    DomainDecomposition dd(gm, cellToPart, rank);
    const DistributedMatrix L = distribute(M, dd);
    const std::vector<label>& addr = dd.cellProcAddressing();
    const label lnC = static_cast<label>(addr.size());

    std::vector<scalar> pL(lnC);
    for (label lc = 0; lc < lnC; ++lc) pL[lc] = pG.internal[addr[lc]];

    // HOST oracle for the processor faces: -interfaceCoeff*(halo - p_local). Exchange p to get the halo.
    std::vector<ProcessorInterface> ifs = dd.interfaces();
    for (auto& it : ifs) it.initInterfaceMatrixUpdate(pL.data());
    Pstream::waitAll();
    std::vector<std::vector<scalar>> hostFlux(ifs.size());
    for (std::size_t j = 0; j < ifs.size(); ++j)
    {
        const std::vector<scalar>& halo = ifs[j].neighbourField();
        hostFlux[j].resize(ifs[j].size());
        for (label i = 0; i < ifs[j].size(); ++i)
            hostFlux[j][i] = -L.interfaceCoeffs[j][i] * (halo[i] - pL[ifs[j].faceCells()[i]]);
    }

    int fails = 0;
    {
        // Flattened boundary-flux array over just the processor faces (procStart = running offset).
        std::vector<label> procStart;
        label tot = 0;
        for (std::size_t j = 0; j < ifs.size(); ++j)
        {
            procStart.push_back(tot);
            tot += ifs[j].size();
        }
        DeviceBuffer<scalar> p_d(pL), fluxB(std::vector<scalar>(tot > 0 ? tot : 1, 0.0));

        std::vector<int>                nbrParts;
        std::vector<std::vector<label>> faceCellsH;
        for (auto& it : ifs)
        {
            nbrParts.push_back(it.neighbProcNo());
            faceCellsH.push_back(it.faceCells());
        }
        DeviceHalo halo(rank, nbrParts, faceCellsH);
        std::vector<DeviceBuffer<label>>  faceCellsD(faceCellsH.size());
        std::vector<DeviceBuffer<scalar>> ifCoeff(L.interfaceCoeffs.size());
        for (std::size_t i = 0; i < faceCellsH.size(); ++i) faceCellsD[i].copyFrom(faceCellsH[i]);
        for (std::size_t i = 0; i < L.interfaceCoeffs.size(); ++i) ifCoeff[i].copyFrom(L.interfaceCoeffs[i]);

        deviceParallelMatrixFluxInterface(halo, faceCellsD, ifCoeff, procStart, p_d, fluxB);
        cudaDeviceSynchronize();
        const std::vector<scalar> fluxH = fluxB.host();

        // (a) vs the host operator.
        scalar num = 0, den = 0;
        for (std::size_t j = 0; j < ifs.size(); ++j)
            for (label i = 0; i < ifs[j].size(); ++i)
            {
                num = std::fmax(num, std::fabs(fluxH[procStart[j] + i] - hostFlux[j][i]));
                den = std::fmax(den, std::fabs(hostFlux[j][i]));
            }
        const scalar relH = den > 0 ? num / den : num;
        if (relH > 1e-11)
        {
            std::printf("[rank %d] FAIL pflux vs host rel=%.3e\n", rank, relH);
            ++fails;
        }

        // (b) CONSERVATION: exchange the computed flux; the two sides must cancel.
        std::vector<std::vector<scalar>> sendF(ifs.size()), recvF(ifs.size());
        for (std::size_t j = 0; j < ifs.size(); ++j)
        {
            const label n = ifs[j].size();
            sendF[j].assign(fluxH.begin() + procStart[j], fluxH.begin() + procStart[j] + n);
            recvF[j].assign(n, 0.0);
            Pstream::irecv(recvF[j].data(), static_cast<int>(n), nbrParts[j], 9);
            Pstream::isend(sendF[j].data(), static_cast<int>(n), nbrParts[j], 9);
        }
        Pstream::waitAll();
        scalar worst = 0, scale = 0;
        for (std::size_t j = 0; j < ifs.size(); ++j)
            for (std::size_t i = 0; i < sendF[j].size(); ++i)
            {
                worst = std::fmax(worst, std::fabs(sendF[j][i] + recvF[j][i]));
                scale = std::fmax(scale, std::fabs(sendF[j][i]));
            }
        const scalar gWorst = Pstream::allReduce(worst, ReduceOp::Max);
        const scalar gScale = Pstream::allReduce(scale, ReduceOp::Max);
        const scalar relC = gScale > 0 ? gWorst / gScale : gWorst;
        if (relC > 1e-11)
        {
            if (Pstream::master()) std::printf("FAIL pflux conservation rel=%.3e\n", relC);
            ++fails;
        }
        else if (Pstream::master())
            std::printf("  cut-face pressure-flux cancellation: rel=%.3e (conservative)\n", relC);
    }

    const label totalFail = Pstream::allReduce(static_cast<label>(fails), ReduceOp::Sum);
    if (Pstream::master())
        std::printf("test_gpu_parallel_pflux: nproc=%d  %s (failures=%d)\n",
                    nproc, totalFail == 0 ? "PASS" : "FAIL", static_cast<int>(totalFail));

    Pstream::finalize();
    return totalFail == 0 ? 0 : 1;
}
