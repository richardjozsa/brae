// Distributed AMI mapDistribute FOUNDATION test: does distributedAmiGather reproduce, on each rank, the target-cell
// values its local source faces need -- even when the AMI patches split across ranks?
//
// Method: encode identity in the field, psi[localCell] = its GLOBAL cell index. After the gather, for every one of
// this rank's source faces and every stencil entry k, psiExt[ nbrCell_local[k] ] must equal the GLOBAL nbrCell of
// that entry (known from the global AMI). If the addressing (local re-index) or the gather (remote fetch) is wrong,
// the fetched global index won't match. np=1 exercises the local re-index (no remote); np>1 exercises the gather.
//
// Run: mpirun -np {1,2,4} test_distributed_ami_gather <caseDir with a cyclicAMI mesh>
#include "primitive_mesh.cuh"
#include "fv_geometry.cuh"
#include "fv_patch.cuh"
#include "scotch_decomposition.cuh"
#include "parallel_simple.cuh"
#include "interface/ami_interface.cuh"
#include "parallel_device_ami.cuh"
#include "device_buffer.cuh"
#include "cf_pstream.cuh"
#include <cstdio>
#include <string>
#include <vector>
#include <cmath>

using namespace brae;

int main(int argc, char** argv)
{
    Pstream::init(argc, argv);
    const int rank = Pstream::myProcNo(), nproc = Pstream::nProcs();
    const std::string caseDir = argc > 1 ? argv[1] : "validation/tc_wedge_ami";

    bool ok = true;
    scalar gMaxErr = 0, gMaxErrN = 0;
    long gChecked = 0, gRecv = 0;
    {
        PrimitiveMesh gm;
        gm.read(caseDir + "/constant/polyMesh");
        const label nC = gm.nCells();

        // global AMI first (needs only the global mesh/geometry/patches, not the decomposition)
        FvGeometry gg;
        gg.build(gm);
        const std::vector<FvPatch> gfvp = buildPatches(gm, gg);
        const std::vector<AMIInterface> gAMIs = buildAMIInterfaces(gm, gg, gfvp);
        if (gAMIs.empty())
        {
            if (Pstream::master()) std::printf("test_distributed_ami_gather: no cyclicAMI in %s -- SKIP\n", caseDir.c_str());
            Pstream::finalize();
            return 0;
        }

        // Decomposition that FORCES the AMI across the cut (so the mapDistribute gather is genuinely exercised, not
        // left trivially local as SCOTCH does for a small conformal-zone AMI): source owner cells -> rank 0, target
        // cells -> rank (1 % nproc); everything else contiguous. Every source face on rank 0 then needs its target
        // cells from rank 1 -> a real cross-rank gather.
        std::vector<label> cellToPart(nC, 0);
        for (label c = 0; c < nC; ++c) cellToPart[c] = static_cast<label>((static_cast<long long>(c) * nproc) / nC);
        if (nproc > 1)
            for (const AMIInterface& a : gAMIs)
            {
                for (label oc : a.ownCell) cellToPart[oc] = 0;
                for (label ncl : a.nbrCell) cellToPart[ncl] = 1;
            }
        Pstream::broadcast(cellToPart.data(), nC, 0);
        const Partition P(gm, cellToPart, rank);
        DistributedAMI D = buildDistributedAMI(gAMIs, cellToPart, P);

        // psi[localCell] = its GLOBAL index
        const label nLoc = P.nCells();
        std::vector<scalar> hpsi(nLoc);
        for (label c = 0; c < nLoc; ++c) hpsi[c] = static_cast<scalar>(P.Lm.cellProcAddr[c]);
        DeviceBuffer<scalar> psi;
        psi.copyFrom(hpsi);

        DeviceBuffer<scalar> psiExt;
        distributedAmiGather(D, psi, psiExt);
        std::vector<scalar> hext;
        psiExt.copyTo(hext);

        // Verify: replicate buildDistributedAMI's stencil-entry order (iterate the same global AMIs, keep this rank's
        // source faces) and check the re-indexed local nbrCell fetches the correct global index.
        std::vector<label> hNbr;
        D.local.nbrCell.copyTo(hNbr);
        scalar maxErr = 0;
        long checked = 0, entry = 0;
        for (const AMIInterface& a : gAMIs)
            for (std::size_t i = 0; i < a.ownCell.size(); ++i)
            {
                if (cellToPart[a.ownCell[i]] != rank) continue;
                for (label k = a.srcOffset[i]; k < a.srcOffset[i + 1]; ++k)
                {
                    const label lNbr = hNbr[entry++];
                    maxErr = std::fmax(maxErr, std::fabs(hext[lNbr] - static_cast<scalar>(a.nbrCell[k])));
                    ++checked;
                }
            }

        gMaxErr  = Pstream::allReduce(maxErr, ReduceOp::Max);
        gChecked = static_cast<long>(Pstream::allReduce(static_cast<scalar>(checked), ReduceOp::Sum));
        gRecv    = static_cast<long>(Pstream::allReduce(static_cast<scalar>(D.nRecv), ReduceOp::Sum));

        // NVSHMEM on-GPU gather must reproduce the SAME result (this is the path the implicit matvec will use).
        scalar maxErrN = 0;
        if (Pstream::nvshmemActive())
        {
            DeviceBuffer<scalar> psiExtN;
            distributedAmiGatherNvshmem(D, psi, psiExtN);
            std::vector<scalar> hextN;
            psiExtN.copyTo(hextN);
            label e2 = 0;
            for (const AMIInterface& a : gAMIs)
                for (std::size_t i = 0; i < a.ownCell.size(); ++i)
                {
                    if (cellToPart[a.ownCell[i]] != rank) continue;
                    for (label k = a.srcOffset[i]; k < a.srcOffset[i + 1]; ++k)
                        maxErrN = std::fmax(maxErrN, std::fabs(hextN[hNbr[e2++]] - static_cast<scalar>(a.nbrCell[k])));
                }
        }
        gMaxErrN = Pstream::allReduce(maxErrN, ReduceOp::Max);
        ok = (gMaxErr < 0.5) && (gMaxErrN < 0.5) && (gChecked > 0);
    }

    if (Pstream::master())
    {
        std::printf("distributed AMI gather: np=%d  checked=%ld  gathered=%ld remote cells  MPI maxErr=%.3e  NVSHMEM maxErr=%.3e\n",
                    nproc, gChecked, gRecv, gMaxErr, gMaxErrN);
        std::printf("%s\n", ok ? "PASS" : "FAIL");
    }
    Pstream::finalize();
    return ok ? 0 : 1;
}
