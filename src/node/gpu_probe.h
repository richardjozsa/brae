#pragma once
// GPU detection, through NVML loaded at run time.
//
// dlopen rather than linking, for three reasons that all matter on a contributor's machine:
//   * the agent must BUILD with no CUDA toolkit present (docs/04-agent.md);
//   * it must START on a machine with no NVIDIA driver, report zero GPUs, and keep heartbeating -- a node that
//     crashes on a driver upgrade is a node that silently leaves the network;
//   * a driver update replaces libnvidia-ml.so.1 underneath us, and dlopen makes that a reopen, not a crash.
//
// The GPU UUID is hashed HERE, at the source. It is a hardware serial, and the control plane only ever stores
// the hash (docs/01-contracts.md), so the raw value must never reach a struct that gets serialized -- otherwise
// someone adds a field to a snapshot one day and publishes it without noticing.
#include "api_client.h"

#include <string>
#include <vector>

namespace brae::node {

struct GpuProbeResult
{
    bool nvmlAvailable = false;
    std::string unavailableReason;      // safe to log; shown by `brae node status`
    std::vector<GpuIdentity> gpus;      // identity: model, VRAM, capability, driver, uuid HASH
};

// Identity of every GPU on this machine. Never throws: no driver is a normal state, not an error.
GpuProbeResult probeGpus();

// Live state for a snapshot. Returns as many entries as it can read; a GPU that fails mid-read is skipped
// rather than failing the whole snapshot, because a partial report beats no report.
std::vector<GpuState> probeGpuState();

// Which shared object to load. BRAE_NVML_LIB overrides it, which is how the "no driver" path gets tested.
std::string nvmlLibraryName();

}  // namespace brae::node
