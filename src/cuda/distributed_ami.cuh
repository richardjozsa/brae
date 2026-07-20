#pragma once
// Distributed cyclicAMI RUNTIME data + gathers (lib side; the BUILDER buildDistributedAMI stays app-side because it
// needs the Partition). The struct uses only lib types (DeviceAMI, DeviceBuffer, SymBuffer), so the linear-solver
// matvec (device_spmv/device_pcg) can include the AMI coupling: gather psi's remote target cells, then deviceAmiAmul.
//
// See parallel_device_ami.cuh for the design: every AMI op is pnf[srcFace]=sum_k w[k]*psi[nbrCell[k]] scattered to
// ownCell; nbrCell is re-indexed into an EXTENDED buffer [local | gathered-remote] and the tail is gathered here.
#include "cf_types.cuh"
#include "device_buffer.cuh"
#include "device_ami.cuh"           // DeviceAMI + deviceAmiAmul
#include "sym_buffer.cuh"           // SymBuffer: symmetric recv buffer for the on-GPU gather
#include "cf_pstream.cuh"           // Pstream isend/irecv for the (explicit-op) host gather
#ifdef BRAE_WITH_NVSHMEM
#include <nvshmem_host.h>
#endif
#include <cuda_runtime.h>
#include <vector>

namespace brae {

struct DistributedAMI
{
    DeviceAMI local;                       // re-indexed AMI: nbrCell in [0,nLocal) local, [nLocal,nLocal+nRecv) gathered
    int nLocal = 0;                        // this rank's cell count (the extended buffer's local head)
    int nRecv  = 0;                        // number of gathered remote cells (the extended buffer's tail)
    bool active = false;

    // mapDistribute schedule (all computed locally from the global AMI + cellToPart -> no setup comms):
    std::vector<int>   recvRank, recvCount, recvStart;
    std::vector<int>   sendRank;
    std::vector<std::vector<label>> sendCells;   // per send-block: the LOCAL cells to pack and send

    // NVSHMEM put-based gather (blocks SORTED by rank -> the remote offset is computable locally):
    std::vector<int>   sendRemoteOffset;         // per send-block: offset in the destination rank's recv buffer
    std::vector<int>   sendLocalOff;             // per send-block: offset into the flattened sendIdx
    std::vector<label> sendIdx;                  // flattened local cell indices to pack (all send blocks, in order)
    int recvCap = 0;                             // global max nRecv across ranks (symmetric recv buffer capacity)
    SymBuffer<scalar>    recvSym;                // symmetric recv buffer (size recvCap); senders put my cells here
    DeviceBuffer<scalar> sendPack;              // packed send values (size sendIdx.size())
    DeviceBuffer<label>  sendIdxD;              // sendIdx on device (for the pack kernel)
    mutable DeviceBuffer<scalar> psiExt;        // scratch extended field [local | gathered] reused across matvecs
    mutable int amulComp = -1;                  // matvec coeff selector: -1 -> ifCoeff (pressure/translational);
                                                // 0/1/2 -> ifCoeffC[comp] for the ROTATIONAL momentum component

    // scratch (host staging for the MPI gather; sized at build)
    mutable std::vector<std::vector<scalar>> sendBuf;   // per send-block pack buffer
    mutable std::vector<scalar> recvBuf;                // flattened recv (size nRecv), copied to psiExt tail
};

// Host MPI gather (D2H/H2D + isend/irecv): fine ONCE per SIMPLE iteration for the explicit ops. psiExt = [psi | remote].
inline void distributedAmiGather(const DistributedAMI& D, const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& psiExt)
{
    psiExt.resize(static_cast<std::size_t>(D.nLocal) + static_cast<std::size_t>(D.nRecv));
    cudaCheck(cudaMemcpyAsync(psiExt.data(), psi.data(), static_cast<std::size_t>(D.nLocal) * sizeof(scalar),
                              cudaMemcpyDeviceToDevice, cudaStreamPerThread), "amiGather local head");
    if (D.nRecv == 0 && D.sendCells.empty()) return;

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

    if (D.nRecv > 0)
    {
        DeviceBuffer<scalar> tail;
        tail.copyFrom(D.recvBuf);
        cudaCheck(cudaMemcpyAsync(psiExt.data() + D.nLocal, tail.data(), static_cast<std::size_t>(D.nRecv) * sizeof(scalar),
                                  cudaMemcpyDeviceToDevice, cudaStreamPerThread), "amiGather tail");
        cudaStreamSynchronize(cudaStreamPerThread);
    }
}

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

// NVSHMEM on-GPU gather (for the implicit matvec: no D2H/H2D, no MPI). Pack -> put each block to the destination's
// symmetric recv at sendRemoteOffset[b] -> barrier -> copy recv into psiExt[nLocal..]. Const D: only mutable scratch
// (sendPack via the kernel, recvSym via NVSHMEM) is written, so it is callable from the const matvec path.
inline void distributedAmiGatherNvshmem(const DistributedAMI& D, const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& psiExt)
{
    const cudaStream_t stream = cudaStreamPerThread;
    psiExt.resize(static_cast<std::size_t>(D.nLocal) + static_cast<std::size_t>(D.nRecv));
    cudaCheck(cudaMemcpyAsync(psiExt.data(), psi.data(), static_cast<std::size_t>(D.nLocal) * sizeof(scalar),
                              cudaMemcpyDeviceToDevice, stream), "amiNvshmem local head");
    if (D.recvCap == 0) return;
#ifdef BRAE_WITH_NVSHMEM
    const int nSend = static_cast<int>(D.sendIdx.size());
    if (nSend > 0)
    {
        constexpr int TPB = 128;
        amiPackKernel<<<(nSend + TPB - 1) / TPB, TPB, 0, stream>>>(
            const_cast<scalar*>(D.sendPack.data()), psi.data(), D.sendIdxD.data(), nSend);
        cudaCheck(cudaGetLastError(), "amiPackKernel");
    }
    for (std::size_t b = 0; b < D.sendRank.size(); ++b)
    {
        const int cnt = static_cast<int>(D.sendCells[b].size());
        if (cnt <= 0) continue;
        nvshmemx_putmem_on_stream(
            const_cast<scalar*>(D.recvSym.data()) + D.sendRemoteOffset[b],
            const_cast<scalar*>(D.sendPack.data()) + D.sendLocalOff[b],
            static_cast<std::size_t>(cnt) * sizeof(scalar),
            D.sendRank[b],
            stream);
    }
    nvshmemx_barrier_all_on_stream(stream);
    if (D.nRecv > 0)
        cudaCheck(cudaMemcpyAsync(psiExt.data() + D.nLocal, const_cast<scalar*>(D.recvSym.data()),
                                  static_cast<std::size_t>(D.nRecv) * sizeof(scalar),
                                  cudaMemcpyDeviceToDevice, stream), "amiNvshmem recv tail");
#endif
}

// AMI matvec contribution: gather psi's remote target cells (on-GPU), then Apsi[ownCell] += ifCoeff*sum_k w[k]*psiExt.
// Adds to Apsi AFTER the internal+halo matvec (deviceParallelAmul). deviceAmiAmul is the validated single-GPU kernel.
inline void distributedAmiAmul(const DistributedAMI& D, const DeviceBuffer<scalar>& psi, DeviceBuffer<scalar>& Apsi)
{
    if (!D.active) return;
    distributedAmiGatherNvshmem(D, psi, D.psiExt);
    // ROTATIONAL momentum component: use ifCoeffC[comp] (matches the deferred-rotation source). Else ami.ifCoeff.
    if (D.amulComp >= 0 && D.local.rotational)
        deviceAmiAmulCoeff(D.local, D.local.ifCoeffC[D.amulComp], D.psiExt, Apsi);
    else
        deviceAmiAmul(D.local, D.psiExt, Apsi);
}

} // namespace brae
