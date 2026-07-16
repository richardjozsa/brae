// Phase 2 gate: the DEVICE distributed matrix-vector product == the serial global Amul.
//
// Same construction as the host test_parallel_spmv (build a global laplacian, SCOTCH-decompose, distribute),
// but the product runs on the GPU: buildDeviceLdu(local matrix) + deviceParallelAmul (local cell-gather Amul +
// NVSHMEM interface coupling). Every local cell must reproduce the serial global Amul, and the result must
// match the host distributed oracle. The device gather sums each row in a different order than the serial
// face-scatter, so the match is to machine precision (FP non-associativity), not bit-exact.
//
// Run: mpirun -np {1,2,4,8} test_gpu_parallel_spmv <caseDir>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "geometric_field.cuh"
#include "foam_field_reader.cuh"
#include "fvm.cuh"
#include "ldu_spmv.cuh"
#include "scotch_decomposition.cuh"
#include "domain_decomposition.cuh"
#include "parallel_amul.cuh"
#include "device_ldu.cuh"
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
    if (argc < 2) { if (Pstream::master()) std::printf("usage: %s <caseDir>\n", argv[0]); Pstream::finalize(); return 2; }
    if (!Pstream::nvshmemActive())
    {
        if (Pstream::master()) std::printf("test_gpu_parallel_spmv: NVSHMEM inactive -- SKIP\n");
        Pstream::finalize();
        return 0;
    }
    const std::string caseDir = argv[1];

    PrimitiveMesh m;  m.read(caseDir + "/constant/polyMesh");
    FvGeometry g;     g.build(m);
    const std::vector<FvPatch> patches = buildPatches(m, g);
    const label nC = m.nCells();

    // Global laplacian(1, p) matrix; psi = the 282/p field. Serial reference = raw lduMatrix Amul.
    GeometricField<scalar> p = buildField<scalar>(readField<scalar>(caseDir + "/282/p"), patches, nC);
    p.evaluateBoundary();
    const FvScalarMatrix M = fvm::laplacian(p, 1.0, m, g, patches);
    const std::vector<scalar>& psiG = p.internal;
    const std::vector<scalar> ApsiSerial = Amul(M, psiG, m);

    // Decompose on master, broadcast; build this partition's local matrix.
    std::vector<label> cellToPart(nC, 0);
    if (Pstream::master()) cellToPart = scotchDecompose(m, nproc);
    Pstream::broadcast(cellToPart.data(), nC, 0);

    DomainDecomposition dd(m, cellToPart, rank);
    const DistributedMatrix L = distribute(M, dd);
    const std::vector<label>& addr = dd.cellProcAddressing();
    std::vector<scalar> psiL(addr.size());
    for (std::size_t lc = 0; lc < addr.size(); ++lc) psiL[lc] = psiG[addr[lc]];

    scalar gMax = 0, gMag = 0;
    {   // scope: DeviceHalo's symmetric buffers freed before Pstream::finalize()
        // Local device matrix (lowerAddr = localOwner, upperAddr = localNeighbour).
        const DeviceLduMatrix dM = buildDeviceLdu(L.diag, L.upper, L.lower, L.lowerAddr, L.upperAddr,
                                                  static_cast<int>(addr.size()));
        DeviceBuffer<scalar> psi_d(psiL), Apsi_d(addr.size());

        // Device halo from this partition's interfaces (same order as L.interfaceCoeffs).
        std::vector<int>                nbrParts;
        std::vector<std::vector<label>> faceCells;
        for (auto& itf : dd.interfaces()) { nbrParts.push_back(itf.neighbProcNo()); faceCells.push_back(itf.faceCells()); }
        DeviceHalo halo(rank, nbrParts, faceCells);

        std::vector<DeviceBuffer<scalar>> coeffs(L.interfaceCoeffs.size());
        for (std::size_t i = 0; i < L.interfaceCoeffs.size(); ++i) coeffs[i].copyFrom(L.interfaceCoeffs[i]);

        deviceParallelAmul(dM.view(), halo, coeffs, psi_d, Apsi_d);
        const std::vector<scalar> ApsiDev = Apsi_d.host();

        scalar maxAbs = 0, mag = 0;
        for (std::size_t lc = 0; lc < addr.size(); ++lc)
        {
            maxAbs = std::fmax(maxAbs, std::fabs(ApsiDev[lc] - ApsiSerial[addr[lc]]));
            mag    = std::fmax(mag, std::fabs(ApsiSerial[addr[lc]]));
        }
        gMax = Pstream::allReduce(maxAbs, ReduceOp::Max);
        gMag = Pstream::allReduce(mag, ReduceOp::Max);
    }

    const scalar rel = gMag > 0 ? gMax / gMag : gMax;
    const bool pass = rel < 1e-10;
    if (Pstream::master())
    {
        std::printf("test_gpu_parallel_spmv np=%d: device distributed Amul vs serial rel=%.3e\n", nproc, rel);
        std::printf("%s\n", pass ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return pass ? 0 : 1;
}
