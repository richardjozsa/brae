#pragma once
// Distributed cyclicAMI (Arbitrary Mesh Interface) for the multi-GPU device path.
//
// The single-GPU DeviceAMI (device_ami.cuh) reduces every AMI operation to one pattern:
//     pnf[srcFace] = sum_k weight[k] * psi[nbrCell[k]]   (interpolate-to-source: GATHER target cells)
//     out[ownCell[i]] += ifCoeff[i] * pnf[srcFace]       (SCATTER to the local source cell)
// so the ONLY thing that breaks under decomposition is that nbrCell[k] (a target cell) may live on another rank.
//
// This file distributes it WITHOUT touching the AMI kernels: it re-indexes nbrCell into an EXTENDED field buffer
//     psiExt = [ local cells (0..nLocal) | gathered remote cells (nLocal..nLocal+nRecv) ]
// and gathers the remote tail (a mapDistribute) before each AMI op. The kernels then run unchanged on psiExt --
// ownCell reads the local part, nbrCell reads local-or-gathered, and the scatter target ownCell stays local.
//
// No setup communication is needed: every rank reads the GLOBAL mesh (gm) and has cellToPart (the decomposition),
// so each rank builds the GLOBAL AMIInterface and computes -- purely locally --
//   * its RECV list: remote cells its own source faces need   (grouped by owning rank), and
//   * its SEND list: its local cells that OTHER ranks' source faces need   (by symmetry over the same global data).
// The per-iteration gather is then a plain Pstream isend/irecv of packed cell values (D2H pack -> MPI -> H2D tail).
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_ami.cuh"
#include "interface/ami_interface.cuh"
#include "parallel_simple.cuh"      // Partition
#include "cf_pstream.cuh"
#include "sym_buffer.cuh"           // SymBuffer: NVSHMEM symmetric recv buffer for the on-GPU gather
#ifdef BRAE_WITH_NVSHMEM
#include <nvshmem_host.h>           // nvshmemx_putmem_on_stream / nvshmemx_barrier_all_on_stream (host on-stream API)
#endif
#include <cuda_runtime.h>
#include <vector>
#include <map>
#include <set>
#include <unordered_map>
#include <algorithm>

namespace brae {

struct DistributedAMI
{
    DeviceAMI local;                       // re-indexed AMI: nbrCell in [0,nLocal) local, [nLocal,nLocal+nRecv) gathered
    int nLocal = 0;                        // this rank's cell count (the extended buffer's local head)
    int nRecv  = 0;                        // number of gathered remote cells (the extended buffer's tail)
    bool active = false;

    // mapDistribute gather schedule (all known locally from the global AMI + cellToPart -> no setup comms):
    //   RECV: recvRank[b] sends recvCount[b] cells, landing contiguously at psiExt[nLocal + recvStart[b] ...].
    //   SEND: to sendRank[b], pack sendCells[b] (local indices) and isend.
    std::vector<int>   recvRank, recvCount, recvStart;
    std::vector<int>   sendRank;
    std::vector<std::vector<label>> sendCells;   // per send-block: the LOCAL cells to pack and send

    // NVSHMEM put-based gather (blocks SORTED by rank -> the remote offset in the destination's recv buffer is
    // computable locally from the global AMI symmetry, no setup exchange): for send-block b, put the packed cells to
    // destination sendRank[b]'s symmetric recv buffer at sendRemoteOffset[b]. recvCap = GLOBAL max nRecv (symmetric
    // buffers must be the same size on every PE). sendIdx/sendLocalOff = the flattened pack layout for a GPU kernel.
    std::vector<int>   sendRemoteOffset;         // per send-block: offset in the destination rank's recv buffer
    std::vector<int>   sendLocalOff;             // per send-block: offset into the flattened sendIdx
    std::vector<label> sendIdx;                  // flattened local cell indices to pack (all send blocks, in order)
    int recvCap = 0;                             // global max nRecv across ranks (symmetric recv buffer capacity)

    // NVSHMEM on-GPU gather resources (allocated in buildDistributedAMI; symmetric recv is collective).
    SymBuffer<scalar>    recvSym;                // symmetric recv buffer (size recvCap); senders put my cells here
    DeviceBuffer<scalar> sendPack;              // packed send values (size sendIdx.size())
    DeviceBuffer<label>  sendIdxD;              // sendIdx on device (for the pack kernel)

    // scratch (host staging for the MPI gather; sized at build)
    mutable std::vector<std::vector<scalar>> sendBuf;   // per send-block pack buffer
    mutable std::vector<scalar> recvBuf;                // flattened recv (size nRecv), copied to psiExt tail
};

// pack kernel for the on-GPU gather: out[i] = psi[idx[i]]. static -> internal linkage per TU (header-only, no ODR clash).
static __global__ void amiPackKernel(
    scalar*       __restrict__ out,
    const scalar* __restrict__ psi,
    const label*  __restrict__ idx,
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = psi[idx[i]];
}

// NVSHMEM on-GPU gather (for the implicit matvec: no D2H/H2D, no MPI, so it can run inside every solver matvec).
// Packs the send cells on the device, puts each block to the destination's symmetric recv buffer at the precomputed
// remote offset, barriers, and copies the recv into psiExt[nLocal..]. The blocks/offsets align by construction (my
// send block == the destination's recv-from-me block, placed at its recvStart for source==me). NVSHMEM active only.
inline void distributedAmiGatherNvshmem(DistributedAMI& D, const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& psiExt)
{
    const cudaStream_t stream = cudaStreamPerThread;
    psiExt.resize(static_cast<std::size_t>(D.nLocal) + static_cast<std::size_t>(D.nRecv));
    cudaCheck(cudaMemcpyAsync(psiExt.data(), psi.data(), static_cast<std::size_t>(D.nLocal) * sizeof(scalar),
                              cudaMemcpyDeviceToDevice, stream), "amiNvshmem local head");
    if (D.recvCap == 0) return;   // no cross-rank AMI coupling anywhere
#ifdef BRAE_WITH_NVSHMEM
    const int nSend = static_cast<int>(D.sendIdx.size());
    if (nSend > 0)
    {
        constexpr int TPB = 128;
        amiPackKernel<<<(nSend + TPB - 1) / TPB, TPB, 0, stream>>>(
            D.sendPack.data(), psi.data(), D.sendIdxD.data(), nSend);
        cudaCheck(cudaGetLastError(), "amiPackKernel");
    }
    for (std::size_t b = 0; b < D.sendRank.size(); ++b)
    {
        const int cnt = static_cast<int>(D.sendCells[b].size());
        if (cnt <= 0) continue;
        nvshmemx_putmem_on_stream(
            D.recvSym.data() + D.sendRemoteOffset[b],          // symmetric addr in the destination's recv buffer
            D.sendPack.data() + D.sendLocalOff[b],             // my packed block
            static_cast<std::size_t>(cnt) * sizeof(scalar),
            D.sendRank[b],
            stream);
    }
    nvshmemx_barrier_all_on_stream(stream);                     // all puts land before anyone reads its recv buffer
    if (D.nRecv > 0)
        cudaCheck(cudaMemcpyAsync(psiExt.data() + D.nLocal, D.recvSym.data(),
                                  static_cast<std::size_t>(D.nRecv) * sizeof(scalar),
                                  cudaMemcpyDeviceToDevice, stream), "amiNvshmem recv tail");
#endif
}

// Build the per-rank distributed AMI. `globalAMIs` is buildAMIInterfaces() on the GLOBAL mesh (same on every
// rank); `cellToPart` is the global cell->rank decomposition; P is this rank's Partition. The returned .local is
// a DeviceAMI whose nbrCell indexes the extended [local|gathered] buffer, ready for the existing AMI kernels.
inline DistributedAMI buildDistributedAMI(
    const std::vector<AMIInterface>& globalAMIs,
    const std::vector<label>& cellToPart,
    const Partition& P)
{
    DistributedAMI D;
    const int me = P.rank;
    const label nLoc = P.nCells();
    D.nLocal = static_cast<int>(nLoc);
    if (globalAMIs.empty()) return D;   // inactive
    D.active = true;

    // global cell -> this rank's local index (or -1 if not local)
    std::unordered_map<label, label> g2l;
    g2l.reserve(static_cast<std::size_t>(nLoc) * 2);
    for (label c = 0; c < nLoc; ++c) g2l[P.Lm.cellProcAddr[c]] = c;

    // ---- RECV side: my source faces' remote target cells, deduped per owning rank, assigned tail slots ----
    // recvSlot[globalCell] = its position in the extended tail (nLocal + slot). Built in first-seen order per rank.
    std::unordered_map<label, int> recvSlot;
    std::map<int, std::vector<label>> recvByRank;   // owning rank -> global cells I need from it (SORTED by rank)
    auto tailSlotOf = [&](label gNbr, int owner) -> int
    {
        auto it = recvSlot.find(gNbr);
        if (it != recvSlot.end()) return it->second;
        const int slot = static_cast<int>(recvSlot.size());
        recvSlot.emplace(gNbr, slot);
        recvByRank[owner].push_back(gNbr);
        return slot;
    };

    // Re-index the AMI stencils for MY source faces into a fresh set of per-interface AMIInterfaces.
    std::vector<AMIInterface> localAMIs;
    for (const AMIInterface& a : globalAMIs)
    {
        AMIInterface la;
        la.patch = a.patch;
        la.nbrPatch = a.nbrPatch;
        la.translational = a.translational;
        la.separation = a.separation;
        la.forwardT = a.forwardT;
        la.srcOffset.push_back(0);
        for (std::size_t i = 0; i < a.ownCell.size(); ++i)
        {
            const label gOwn = a.ownCell[i];
            if (cellToPart[gOwn] != me) continue;   // not my source face
            la.ownCell.push_back(g2l.at(gOwn));
            la.weightsSum.push_back(a.weightsSum[i]);
            la.magSf.push_back(a.magSf[i]);
            la.deltaCoeffs.push_back(a.deltaCoeffs[i]);
            la.weights.push_back(a.weights[i]);
            la.Sf.push_back(a.Sf[i]);
            if (i < a.corrVec.size()) la.corrVec.push_back(a.corrVec[i]);
            if (i < a.dOwn.size())    la.dOwn.push_back(a.dOwn[i]);
            const label b = a.srcOffset[i], e = a.srcOffset[i + 1];
            for (label k = b; k < e; ++k)
            {
                const label gNbr = a.nbrCell[k];
                const int owner = static_cast<int>(cellToPart[gNbr]);
                label lNbr;
                if (owner == me) lNbr = g2l.at(gNbr);                          // local target cell
                else             lNbr = nLoc + tailSlotOf(gNbr, owner);        // gathered remote (extended tail)
                la.nbrCell.push_back(lNbr);
                la.weight.push_back(a.weight[k]);
                if ((std::size_t)k < a.dNbr.size()) la.dNbr.push_back(a.dNbr[k]);
            }
            la.srcOffset.push_back(static_cast<label>(la.nbrCell.size()));
        }
        if (!la.ownCell.empty()) localAMIs.push_back(std::move(la));
    }
    D.nRecv = static_cast<int>(recvSlot.size());
    D.local = buildDeviceAMI(localAMIs);

    // Finalise the RECV schedule: contiguous blocks per source rank, in tail-slot order within each block.
    // recvSlot already assigns global tail slots first-seen; regroup so each rank's cells are contiguous and
    // remap the tail slots accordingly (so a single isend per source rank lands in one contiguous tail range).
    std::vector<int> remap(D.nRecv, -1);
    int cursor = 0;
    for (auto& kv : recvByRank)
    {
        D.recvRank.push_back(kv.first);
        D.recvCount.push_back(static_cast<int>(kv.second.size()));
        D.recvStart.push_back(cursor);
        for (label gc : kv.second) remap[recvSlot.at(gc)] = cursor++;
    }
    // apply remap to the DeviceAMI nbrCell tail indices (host rebuild of nbrCell, then re-upload)
    if (D.nRecv > 0)
    {
        std::vector<label> nc(D.local.nnz);
        D.local.nbrCell.copyTo(nc);
        for (label& x : nc)
            if (x >= nLoc) x = nLoc + remap[static_cast<int>(x - nLoc)];
        D.local.nbrCell.copyFrom(nc);
    }

    // ---- SEND side: by symmetry, scan ALL global source faces; any whose target cell is MINE and whose source
    // cell is on another rank r means r will request that cell from me -> add it to my send-block for r. ----
    std::map<int, std::vector<label>> sendByRank;   // dest rank -> my local cells it needs (deduped, SORTED by rank)
    std::map<int, std::set<label>> seen;
    for (const AMIInterface& a : globalAMIs)
        for (std::size_t i = 0; i < a.ownCell.size(); ++i)
        {
            const int srcRank = static_cast<int>(cellToPart[a.ownCell[i]]);
            if (srcRank == me) continue;   // that rank's target cells it fetches; my sends are to OTHER ranks
            for (label k = a.srcOffset[i]; k < a.srcOffset[i + 1]; ++k)
            {
                const label gNbr = a.nbrCell[k];
                if (cellToPart[gNbr] != me) continue;   // target cell not mine -> not my job to send
                if (seen[srcRank].insert(gNbr).second) sendByRank[srcRank].push_back(g2l.at(gNbr));
            }
        }
    // The SEND order matches how the receiver laid out its recv block for this rank (both iterate the SAME global
    // AMIs, dedupe first-seen). The REMOTE OFFSET (where my block lands in the destination's recv buffer) is the
    // destination's recvStart for source==me = the count of distinct cells the destination fetches from ranks < me;
    // computable here from the same global data (both sides sort blocks by rank) -> still no setup exchange.
    auto remoteRecvOffset = [&](int dest) -> int
    {
        std::map<int, std::set<label>> byRank;   // dest's recv, source rank -> distinct cells (only ranks < me matter)
        for (const AMIInterface& a : globalAMIs)
            for (std::size_t i = 0; i < a.ownCell.size(); ++i)
            {
                if (static_cast<int>(cellToPart[a.ownCell[i]]) != dest) continue;   // the destination's source faces
                for (label k = a.srcOffset[i]; k < a.srcOffset[i + 1]; ++k)
                {
                    const int owner = static_cast<int>(cellToPart[a.nbrCell[k]]);
                    if (owner != dest) byRank[owner].insert(a.nbrCell[k]);
                }
            }
        int off = 0;
        for (const auto& kv : byRank) { if (kv.first < me) off += static_cast<int>(kv.second.size()); else break; }
        return off;
    };
    int flat = 0;
    for (auto& kv : sendByRank)
    {
        D.sendRank.push_back(kv.first);
        D.sendRemoteOffset.push_back(remoteRecvOffset(kv.first));
        D.sendLocalOff.push_back(flat);
        for (label c : kv.second) { D.sendIdx.push_back(c); ++flat; }
        D.sendCells.push_back(std::move(kv.second));
    }
    D.sendBuf.resize(D.sendCells.size());
    for (std::size_t b = 0; b < D.sendCells.size(); ++b) D.sendBuf[b].resize(D.sendCells[b].size());
    D.recvBuf.assign(static_cast<std::size_t>(D.nRecv), 0.0);
    D.recvCap = static_cast<int>(Pstream::allReduce(static_cast<label>(D.nRecv), ReduceOp::Max));   // symmetric buffer size

    // NVSHMEM resources: the symmetric recv buffer is a COLLECTIVE alloc (recvCap is the same on every rank), so all
    // ranks construct it together even if their own nRecv is 0. The packed-send buffer + device sendIdx are local.
    if (D.recvCap > 0)
    {
        D.recvSym = SymBuffer<scalar>(static_cast<std::size_t>(D.recvCap));
        if (!D.sendIdx.empty())
        {
            D.sendPack.resize(D.sendIdx.size());
            D.sendIdxD.copyFrom(D.sendIdx);
        }
    }
    return D;
}

// Gather the remote target-cell values of a device cell field `psi` (size nLocal) into psiExt (size nLocal+nRecv):
// psiExt[0..nLocal) = psi; psiExt[nLocal..) = the remote cells this rank needs. Uses Pstream isend/irecv (pack on
// the device -> host staging -> H2D the recv tail). Call before an AMI op; then run the op on psiExt.
inline void distributedAmiGather(const DistributedAMI& D, const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& psiExt)
{
    psiExt.resize(static_cast<std::size_t>(D.nLocal) + static_cast<std::size_t>(D.nRecv));
    cudaCheck(cudaMemcpyAsync(psiExt.data(), psi.data(), static_cast<std::size_t>(D.nLocal) * sizeof(scalar),
                              cudaMemcpyDeviceToDevice, cudaStreamPerThread), "amiGather local head");
    if (D.nRecv == 0 && D.sendCells.empty()) return;

    // pack each send block from the device field (D2H the requested local cells)
    std::vector<scalar> hpsi(static_cast<std::size_t>(D.nLocal));
    psi.copyTo(hpsi);
    for (std::size_t b = 0; b < D.sendCells.size(); ++b)
        for (std::size_t j = 0; j < D.sendCells[b].size(); ++j)
            D.sendBuf[b][j] = hpsi[D.sendCells[b][j]];

    const int tag = 0x0A31;
    for (std::size_t b = 0; b < D.recvRank.size(); ++b)
        Pstream::irecv(D.recvBuf.data() + D.recvStart[b], D.recvCount[b], D.recvRank[b], tag);
    for (std::size_t b = 0; b < D.sendRank.size(); ++b)
        Pstream::isend(D.sendBuf[b].data(), static_cast<int>(D.sendBuf[b].size()), D.sendRank[b], tag);
    Pstream::waitAll();

    if (D.nRecv > 0)   // H2D the gathered tail into psiExt[nLocal..]
    {
        DeviceBuffer<scalar> tail;
        tail.copyFrom(D.recvBuf);
        cudaCheck(cudaMemcpyAsync(psiExt.data() + D.nLocal, tail.data(), static_cast<std::size_t>(D.nRecv) * sizeof(scalar),
                                  cudaMemcpyDeviceToDevice, cudaStreamPerThread), "amiGather tail");
        cudaStreamSynchronize(cudaStreamPerThread);
    }
}

} // namespace brae
