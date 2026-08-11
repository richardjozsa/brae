// Phase 4a L4: the DEVICE distributed H() == the host parallelMatrixH.
//
// H(psi) = (diag*psi - A_parallel*psi + source)/V, i.e. lduMatrix::H with the processor-interface off-diagonal
// included. deviceParallelMatrixH = the local deviceMatrixH + the interface term (+ifCoeff*psiNbr/V). This is
// the H() that feeds HbyA in the SIMPLE corrector, so it must match the validated host operator exactly.
//
// Run: mpirun -np {1,2,4,8} test_gpu_parallel_h <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "scotch_decomposition.cuh"
#include "domain_decomposition.cuh"
#include "parallel_amul.cuh"
#include "parallel_matrix_ops.cuh"
#include "device_mesh.cuh"
#include "device_ldu.cuh"
#include "device_halo.cuh"
#include "device_buffer.cuh"
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
        if (Pstream::master()) std::printf("test_gpu_parallel_h: NVSHMEM inactive -- SKIP\n");
        Pstream::finalize();
        return 0;
    }
    const std::string caseDir = argv[1];

    PrimitiveMesh gm;  gm.read(caseDir + "/constant/polyMesh");
    FvGeometry gg;     gg.build(gm);
    const std::vector<FvPatch> gpatches = buildPatches(gm, gg);
    const label nC = gm.nCells();

    GeometricField<scalar> p = buildField<scalar>(readField<scalar>(caseDir + "/282/p"), gpatches, nC);
    p.evaluateBoundary();
    FvScalarMatrix M = fvm::laplacian(p, 1.0, gm, gg, gpatches);
    for (label c = 0; c < nC; ++c) M.source[c] = gg.V()[c] * (1.0 + 0.5 * std::sin(40.0 * gg.C()[c].x));
    const std::vector<scalar>& psiG = p.internal;

    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(gm, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);
    DomainDecomposition dd(gm, cellToPart, rank);
    DistributedMatrix L = distribute(M, dd);
    foldBoundary(L, M, gpatches, dd.cellProcAddressing(), cellToPart, rank);
    const std::vector<label>& addr = dd.cellProcAddressing();
    const label lnC = static_cast<label>(addr.size());

    // Local psi + local cell volumes (this partition's slice of the global mesh).
    std::vector<scalar> psiL(lnC), Vl(lnC);
    for (label lc = 0; lc < lnC; ++lc)
    {
        psiL[lc] = psiG[addr[lc]];
        Vl[lc]   = gg.V()[addr[lc]];
    }

    // HOST oracle: parallelMatrixH (H = (diag*psi - A_parallel*psi + b)/V).
    std::vector<ProcessorInterface> ifs = dd.interfaces();
    const std::vector<scalar> Hhost = parallelMatrixH(L, psiL, ifs, Vl);

    int fails = 0;
    {
        // Device: local matrix over the partition's own mesh addressing, V from the global slice.
        const DeviceLduMatrix dMl = buildDeviceLdu(L.diag, L.upper, L.lower, L.lowerAddr, L.upperAddr,
                                                   static_cast<int>(lnC));
        DeviceMesh dm;
        dm.nCells = static_cast<int>(lnC);
        dm.nInternalFaces = static_cast<int>(L.upper.size());
        dm.V.copyFrom(Vl);
        // No real-boundary term in this comparison: parallelMatrixH has none either (bdDiag/bdSrc are the
        // vector-momentum path). An empty boundary gather makes deviceMatrixH's boundary loop a no-op.
        dm.nBndFaces = 0;
        dm.bndCellStart.copyFrom(std::vector<label>(lnC + 1, 0));
        dm.bndPerm.copyFrom(std::vector<label>{});

        DeviceBuffer<scalar> psi_d(psiL), src_d(L.b), Hk;
        DeviceBuffer<scalar> bdDiag, bdSrc;   // empty: nBndFaces == 0

        std::vector<int>                nbrParts;
        std::vector<std::vector<label>> faceCellsH;
        for (auto& itf : ifs)
        {
            nbrParts.push_back(itf.neighbProcNo());
            faceCellsH.push_back(itf.faceCells());
        }
        DeviceHalo halo(rank, nbrParts, faceCellsH);

        std::vector<DeviceBuffer<label>>  faceCellsD(faceCellsH.size());
        std::vector<DeviceBuffer<scalar>> ifCoeff(L.interfaceCoeffs.size());
        for (std::size_t i = 0; i < faceCellsH.size(); ++i) faceCellsD[i].copyFrom(faceCellsH[i]);
        for (std::size_t i = 0; i < L.interfaceCoeffs.size(); ++i) ifCoeff[i].copyFrom(L.interfaceCoeffs[i]);

        deviceParallelMatrixH(dMl.view(), dm, halo, faceCellsD, ifCoeff, psi_d, src_d, bdDiag, bdSrc, Hk);
        cudaDeviceSynchronize();

        const std::vector<scalar> Hdev = Hk.host();
        scalar num = 0, den = 0;
        for (label c = 0; c < lnC; ++c)
        {
            num = std::fmax(num, std::fabs(Hdev[c] - Hhost[c]));
            den = std::fmax(den, std::fabs(Hhost[c]));
        }
        const scalar rel = den > 0 ? num / den : num;
        if (rel > 1e-11)
        {
            std::printf("[rank %d] FAIL H rel=%.3e\n", rank, rel);
            ++fails;
        }
    }

    const label totalFail = Pstream::allReduce(static_cast<label>(fails), ReduceOp::Sum);
    if (Pstream::master())
        std::printf("test_gpu_parallel_h: nproc=%d  %s (failures=%d)\n",
                    nproc, totalFail == 0 ? "PASS" : "FAIL", static_cast<int>(totalFail));

    Pstream::finalize();
    return totalFail == 0 ? 0 : 1;
}
