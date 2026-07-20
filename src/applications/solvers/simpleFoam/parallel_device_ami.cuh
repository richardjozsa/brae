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
#include <vector>
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

    // scratch (host staging for the gather; sized at build)
    mutable std::vector<std::vector<scalar>> sendBuf;   // per send-block pack buffer
    mutable std::vector<scalar> recvBuf;                // flattened recv (size nRecv), copied to psiExt tail
};

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
    std::unordered_map<int, std::vector<label>> recvByRank;   // owning rank -> global cells I need from it
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
    std::unordered_map<int, std::vector<label>> sendByRank;   // dest rank -> my local cells it needs (deduped)
    std::unordered_map<int, std::unordered_map<label, char>> seen;
    for (const AMIInterface& a : globalAMIs)
        for (std::size_t i = 0; i < a.ownCell.size(); ++i)
        {
            const int srcRank = static_cast<int>(cellToPart[a.ownCell[i]]);
            if (srcRank == me) continue;   // that rank's target cells it fetches; my sends are to OTHER ranks
            const label b = a.srcOffset[i], e = a.srcOffset[i + 1];
            for (label k = b; k < e; ++k)
            {
                const label gNbr = a.nbrCell[k];
                if (cellToPart[gNbr] != me) continue;   // target cell not mine -> not my job to send
                auto& s = seen[srcRank];
                if (s.emplace(gNbr, 1).second) sendByRank[srcRank].push_back(g2l.at(gNbr));
            }
        }
    // The SEND order must match how the receiver laid out its recv block for this rank. Both sides iterate the
    // SAME global AMIs in the SAME order and dedupe first-seen, so the k-th cell I send to r equals the k-th cell
    // r placed in its recv block from me -> the orders agree by construction (no exchange of the ordering needed).
    for (auto& kv : sendByRank)
    {
        D.sendRank.push_back(kv.first);
        D.sendCells.push_back(std::move(kv.second));
    }
    D.sendBuf.resize(D.sendCells.size());
    for (std::size_t b = 0; b < D.sendCells.size(); ++b) D.sendBuf[b].resize(D.sendCells[b].size());
    D.recvBuf.assign(static_cast<std::size_t>(D.nRecv), 0.0);
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
