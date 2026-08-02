#pragma once
// `brae node register | status | unregister`, reached as `brae node ...` (which execs brae-agent).
//
// Registration touches four things that outlive the command: an identity file, a NetBird peer, a systemd unit,
// and a row in the registry. It must therefore be ATOMIC -- a failure at any step undoes the ones before it.
// A half-registered machine is worse than an unregistered one: it looks joined, never works, and the owner has
// no obvious way to tell which half is missing.
//
// Every step is injected so that atomicity is testable rather than asserted: the tests drive an API that
// refuses at step three, a service manager that refuses at step five, and check that nothing is left behind.
#include "api_client.h"
#include "gpu_probe.h"
#include "identity.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace brae::node {

// Installing and starting the background service. Real implementation writes a systemd unit; tests use a fake,
// and `--no-service` uses a no-op so a developer can register without root.
class ServiceManager
{
public:
    virtual ~ServiceManager() = default;

    // Create the unprivileged account the unit runs as, if it does not exist. The unit declares User=brae, so
    // without this `systemctl start` fails on a clean machine -- on the very first command a contributor runs.
    virtual bool ensureUser(std::string& error) = 0;

    // Hand a file to that account. `register` runs under sudo, so anything it writes is owned by root, and the
    // daemon -- which does not run as root -- then cannot read its own identity. The service starts and dies.
    virtual bool takeOwnership(const std::string& path, std::string& error) = 0;

    virtual bool install(std::string& error) = 0;
    virtual bool start(std::string& error) = 0;
    virtual bool stopAndRemove(std::string& error) = 0;
    virtual std::string statusLine() = 0;          // human-readable, e.g. "active (running) since ..."
};

std::unique_ptr<ServiceManager> makeSystemdService(const std::string& unitPath, const std::string& execPath);
std::unique_ptr<ServiceManager> makeNoopService();

/// Is this a path the service can actually execute once the unit's hardening is applied?
///
/// The unit runs as the `brae` system user with `ProtectHome=true`, so anything under a home directory is
/// invisible to it -- and a home directory is mode 0700 besides, so the user could not traverse it anyway.
/// Registering straight out of a build tree therefore installed a unit that could never start: systemd looped
/// on 203/EXEC forever while `brae node status` said "activating" and the node sat OFFLINE in the registry.
///
/// Anyone who builds from source hits this, which at present is everyone.
bool isServiceReachablePath(const std::string& path);

/// Where the agent is copied so the service can reach it. `brae` goes alongside it, because the agent finds the
/// solver as its own sibling.
std::string systemInstallDir();

struct CliDeps
{
    ApiClient* api = nullptr;
    ServiceManager* service = nullptr;

    std::function<GpuProbeResult()> probeGpus;
    std::function<std::vector<GpuState>()> probeGpuState;
    std::function<std::string()> nowIso8601;
    std::function<void(const std::string&)> out;   // stdout
    std::function<void(const std::string&)> err;   // stderr

    std::string identityPath;
    std::string apiUrl;
    std::string agentVersion = "0.3.0";
    std::string braeVersion;
    std::string displayName;
    std::string architecture;
    std::string operatingSystem = "linux";
    std::string liveUrlFallback = "https://brae.sh/live";

    // How long `register` waits for the node to appear as connected before declaring success.
    int verifySeconds = 30;
    std::function<bool()> verifyOnce;              // one attempt; null means skip verification
    std::function<void(int)> sleepFor;
};

int cmdRegister(const CliDeps& deps, const std::string& enrollmentToken);
int cmdStatus(const CliDeps& deps);
int cmdUnregister(const CliDeps& deps);

}  // namespace brae::node
