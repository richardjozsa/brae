#include "cli.h"

#include <cstdio>
#include <memory>
#include <sstream>

namespace brae::node {
namespace {

std::string pad(const std::string& s, std::size_t width)
{
    return s.size() >= width ? s : s + std::string(width - s.size(), ' ');
}

std::string gigabytes(int mb)
{
    return std::to_string((mb + 512) / 1024) + " GB";
}

}  // namespace

int cmdRegister(const CliDeps& d, const std::string& enrollmentToken)
{
    auto say = [&](const std::string& m) { if (d.out) d.out(m); };
    auto oops = [&](const std::string& m) { if (d.err) d.err(m); };

    if (enrollmentToken.empty())
    {
        oops("an enrollment token is required: sudo brae node register --token brae_ent_...");
        return 2;
    }

    // 1. GPUs. A node with none cannot do the one job the network has, so registering it would only add a
    //    machine that is permanently unavailable.
    const GpuProbeResult probe = d.probeGpus ? d.probeGpus() : GpuProbeResult{};
    if (!probe.nvmlAvailable)
    {
        oops("no NVIDIA driver found (" + probe.unavailableReason + ").\n"
             "brae node needs a working GPU; install the driver and try again.");
        return 1;
    }
    if (probe.gpus.empty())
    {
        oops("the NVIDIA driver is present but reports no usable GPU. Nothing to contribute.");
        return 1;
    }

    // 2. Existing identity. Re-registering silently would strand the old node id and split this machine's
    //    history across two rows.
    const IdentityResult existing = loadIdentity(d.identityPath);
    if (existing.ok)
    {
        oops("this machine is already registered as " + existing.identity.nodeId + ".\n"
             "Run `sudo brae node unregister` first if you want to re-register it.");
        return 1;
    }

    // 3. Register with the control plane.
    RegisterRequest req;
    req.displayName = d.displayName.empty() ? probe.gpus.front().model : d.displayName;
    req.operatingSystem = d.operatingSystem;
    req.architecture = d.architecture;
    req.agentVersion = d.agentVersion;
    req.braeVersion = d.braeVersion;
    req.gpus = probe.gpus;
    req.systemRamMb = d.systemRamMb ? d.systemRamMb() : 0;
    req.timezone = d.timezone ? d.timezone() : std::string();

    const RegisterResult reg = d.api->registerNode(req, enrollmentToken);
    if (!reg.ok)
    {
        oops("the control plane refused the registration: " + reg.error);
        return 1;
    }

    Identity id;
    id.nodeId = reg.nodeId;
    id.nodeToken = reg.nodeToken;
    id.apiUrl = d.apiUrl;
    id.registeredAt = d.nowIso8601 ? d.nowIso8601() : "";

    std::string error;
    if (!saveIdentity(d.identityPath, id, error))
    {
        // We hold a node token we cannot persist. Give it back rather than leaving a registry row that no
        // machine can ever authenticate as.
        d.api->unregisterNode(reg.nodeId, reg.nodeToken);
        oops("could not save the node identity: " + error + "\n(the registration was rolled back)");
        return 1;
    }

    // 4. NetBird enrolment belongs here (milestone 5). Until then the agent talks to the control plane
    //    directly, and `register` does not create a peer -- so there is nothing to roll back for it either.

    // 5. The service account, and the identity it has to be able to read.
    //
    //    This command runs under sudo, so everything above was written by root, while the daemon runs as an
    //    unprivileged user. Without both steps the unit installs cleanly and then dies on every start with
    //    "no identity" -- which looks like a broken agent rather than a permissions mistake.
    const auto rollback = [&](const std::string& what) {
        std::string ignored;
        if (d.service) d.service->stopAndRemove(ignored);
        removeIdentity(d.identityPath, ignored);
        d.api->unregisterNode(reg.nodeId, reg.nodeToken);
        oops(what + ": " + error + "\n(the registration was rolled back)");
    };

    if (d.service)
    {
        if (!d.service->ensureUser(error))
        {
            rollback("could not create the service account");
            return 1;
        }
        if (!d.service->takeOwnership(d.identityPath, error))
        {
            rollback("could not give the node identity to the service account");
            return 1;
        }

        // 6. Install and start.
        if (!d.service->install(error) || !d.service->start(error))
        {
            rollback("could not start the brae-agent service");
            return 1;
        }
    }

    // 7. Verify. Registration is only real once the control plane has heard from the agent -- otherwise the
    //    contributor is told they joined and then never appears.
    bool verified = true;
    if (d.verifyOnce)
    {
        verified = false;
        for (int waited = 0; waited < d.verifySeconds; waited += 2)
        {
            if (d.verifyOnce()) { verified = true; break; }
            if (d.sleepFor) d.sleepFor(2);
        }
    }

    // 8. Say what happened.
    const GpuIdentity& g = probe.gpus.front();
    std::ostringstream o;
    o << "\nBrae Node\n\n"
      << pad("Node ID", 14) << reg.nodeId << "\n"
      << pad("GPU", 14) << g.model << (probe.gpus.size() > 1
                                       ? " (+" + std::to_string(probe.gpus.size() - 1) + " more)" : "") << "\n"
      << pad("VRAM", 14) << gigabytes(g.vramMb) << "\n"
      << pad("Driver", 14) << g.driverVersion << "\n"
      << pad("Network", 14) << (verified ? "Connected" : "Not yet reporting") << "\n"
      << pad("Status", 14) << (verified ? "Available" : "Starting") << "\n\n";
    if (verified) o << "Your GPU is now registered:\n"
                    << (reg.liveUrl.empty() ? d.liveUrlFallback + "/node/" + reg.nodeId : reg.liveUrl) << "\n";
    else o << "Registered, but the agent has not reported yet.\n"
              "Check it with: brae node status\n";
    say(o.str());
    return verified ? 0 : 1;
}

int cmdStatus(const CliDeps& d)
{
    auto say = [&](const std::string& m) { if (d.out) d.out(m); };

    // Must work with the network down, and say so: this is the command someone runs precisely when things are
    // not working, so it reads local state first and never blocks on the control plane.
    const IdentityResult id = loadIdentity(d.identityPath);
    if (!id.ok)
    {
        say("Brae Node\n\nNot registered on this machine.\n" + id.error +
            "\n\nRegister with: sudo brae node register --token brae_ent_...\n");
        return 1;
    }

    std::ostringstream o;
    o << "\nBrae Node   " << id.identity.nodeId << "\n\n";
    o << pad("Agent", 14) << (d.service ? d.service->statusLine() : std::string("unknown")) << "\n";
    o << pad("API", 14) << id.identity.apiUrl << "\n";

    const GpuProbeResult probe = d.probeGpus ? d.probeGpus() : GpuProbeResult{};
    if (!probe.nvmlAvailable)
    {
        o << pad("GPUs", 14) << "none (" << probe.unavailableReason << ")\n";
    }
    else
    {
        const std::vector<GpuState> state = d.probeGpuState ? d.probeGpuState() : std::vector<GpuState>{};
        for (std::size_t i = 0; i < probe.gpus.size(); ++i)
        {
            const GpuIdentity& g = probe.gpus[i];
            o << pad(i == 0 ? "GPUs" : "", 14) << g.index << "  " << pad(g.model, 26)
              << pad(gigabytes(g.vramMb), 10);
            if (i < state.size()) o << state[i].temperatureCelsius << " C  "
                                    << state[i].utilizationPercent << "%";
            o << "\n";
        }
    }
    say(o.str() + "\n");
    return 0;
}

int cmdUnregister(const CliDeps& d)
{
    auto say = [&](const std::string& m) { if (d.out) d.out(m); };
    auto oops = [&](const std::string& m) { if (d.err) d.err(m); };

    const IdentityResult id = loadIdentity(d.identityPath);
    if (!id.ok)
    {
        oops("this machine is not registered (" + id.error + ")");
        return 1;
    }

    std::string error;
    if (d.service) d.service->stopAndRemove(error);

    // Tell the control plane, but do not depend on it: the local side must come off even when the API is
    // unreachable, or a contributor cannot leave the network while it is down.
    const SimpleResult remote = d.api ? d.api->unregisterNode(id.identity.nodeId, id.identity.nodeToken)
                                      : SimpleResult{};
    removeIdentity(d.identityPath, error);

    if (!remote.ok)
    {
        oops("removed locally, but the control plane could not be told (" + remote.error + ").\n"
             "It will still list " + id.identity.nodeId + " until it times out.");
        return 1;
    }
    say("\n" + id.identity.nodeId + " has left the network.\n"
        "It stays in the network's history and can be registered again later.\n");
    return 0;
}

}  // namespace brae::node
