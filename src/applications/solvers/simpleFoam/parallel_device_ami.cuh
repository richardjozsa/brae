#pragma once
// Distributed cyclicAMI BUILDER (app side; needs the Partition). The runtime struct DistributedAMI + the gathers
// live in the lib (distributed_ami.cuh) so the linear-solver matvec can include the AMI coupling. See that header
// for the design: every AMI op is pnf[srcFace]=sum_k w[k]*psi[nbrCell[k]] scattered to ownCell; nbrCell is
// re-indexed into an EXTENDED buffer [local | gathered-remote] and the tail is gathered before each op.
//
// No setup communication: every rank reads the GLOBAL mesh + has cellToPart, so each builds the GLOBAL AMIInterface
// and computes -- purely locally -- its RECV list (remote cells its source faces need) AND its SEND list (its cells
// other ranks' source faces need, by symmetry) AND the NVSHMEM remote offsets (blocks sorted by rank).
#include "distributed_ami.cuh"       // DistributedAMI struct + gathers (lib)
#include "interface/ami_interface.cuh"
#include "parallel_simple.cuh"       // Partition
#include <vector>
#include <map>
#include <set>
#include <unordered_map>

namespace brae {

// Build the per-rank distributed AMI. `globalAMIs` is buildAMIInterfaces() on the GLOBAL mesh (same on every rank);
// `cellToPart` is the global cell->rank decomposition; P is this rank's Partition. The returned .local is a DeviceAMI
// whose nbrCell indexes the extended [local|gathered] buffer, ready for the existing AMI kernels on psiExt.
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

    // Finalise the RECV schedule: contiguous blocks per source rank (sorted), tail-slot order within each block.
    std::vector<int> remap(D.nRecv, -1);
    int cursor = 0;
    for (auto& kv : recvByRank)
    {
        D.recvRank.push_back(kv.first);
        D.recvCount.push_back(static_cast<int>(kv.second.size()));
        D.recvStart.push_back(cursor);
        for (label gc : kv.second) remap[recvSlot.at(gc)] = cursor++;
    }
    if (D.nRecv > 0)   // apply remap to the DeviceAMI nbrCell tail indices (host rebuild, then re-upload)
    {
        std::vector<label> nc(D.local.nnz);
        D.local.nbrCell.copyTo(nc);
        for (label& x : nc)
            if (x >= nLoc) x = nLoc + remap[static_cast<int>(x - nLoc)];
        D.local.nbrCell.copyFrom(nc);
    }

    // ---- SEND side: by symmetry, any global source face whose target cell is MINE and whose source cell is on
    // another rank r means r requests that cell from me -> my send-block for r. Blocks SORTED by rank (std::map). ----
    std::map<int, std::vector<label>> sendByRank;
    std::map<int, std::set<label>> seen;
    for (const AMIInterface& a : globalAMIs)
        for (std::size_t i = 0; i < a.ownCell.size(); ++i)
        {
            const int srcRank = static_cast<int>(cellToPart[a.ownCell[i]]);
            if (srcRank == me) continue;
            for (label k = a.srcOffset[i]; k < a.srcOffset[i + 1]; ++k)
            {
                const label gNbr = a.nbrCell[k];
                if (cellToPart[gNbr] != me) continue;
                if (seen[srcRank].insert(gNbr).second) sendByRank[srcRank].push_back(g2l.at(gNbr));
            }
        }
    // The REMOTE OFFSET (where my block lands in the destination's recv buffer) = the destination's recvStart for
    // source==me = the count of distinct cells the destination fetches from ranks < me; computable from the same
    // global data (both sides sort blocks by rank + dedup first-seen) -> still no setup exchange.
    auto remoteRecvOffset = [&](int dest) -> int
    {
        std::map<int, std::set<label>> byRank;
        for (const AMIInterface& a : globalAMIs)
            for (std::size_t i = 0; i < a.ownCell.size(); ++i)
            {
                if (static_cast<int>(cellToPart[a.ownCell[i]]) != dest) continue;
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

    // NVSHMEM resources: the symmetric recv buffer is a COLLECTIVE alloc (recvCap same on every rank), so all ranks
    // construct it together even if their own nRecv is 0. The packed-send buffer + device sendIdx are local.
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

} // namespace brae
