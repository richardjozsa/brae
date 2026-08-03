// brae-agent: the node service, and the implementation behind `brae node ...`.
//
//   brae-agent run                       the daemon (systemd runs this)
//   brae-agent node register --token T   join this machine to the network
//   brae-agent node status               what this machine thinks is happening
//   brae-agent node unregister           leave
//
// Users type `brae node ...`; brae execs this binary. Kept separate from brae because it links neither CUDA nor
// brae_core: a node must keep reporting even when the solver build on it is broken.
#include "agent.h"
#include "cli.h"
#include "gpu_probe.h"
#include "http_curl.h"
#include "identity.h"

#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <filesystem>
#include <string>
#include <system_error>
#include <thread>
#include <vector>

#include <sys/utsname.h>
#include <unistd.h>

namespace {

using namespace brae::node;

constexpr const char* kAgentVersion = "0.3.1";

std::string nowIso8601()
{
    const std::time_t t = std::time(nullptr);
    std::tm tm {};
    ::gmtime_r(&t, &tm);
    char buf[32];
    std::strftime(buf, sizeof buf, "%Y-%m-%dT%H:%M:%SZ", &tm);
    return buf;
}

long long monotonicSeconds()
{
    // Monotonic, never the wall clock: an NTP step or a resume from suspend must not produce a burst of
    // snapshots or stall the loop until the clock catches up.
    return std::chrono::duration_cast<std::chrono::seconds>(
               std::chrono::steady_clock::now().time_since_epoch()).count();
}

std::string machineArchitecture()
{
    struct utsname u {};
    return ::uname(&u) == 0 ? u.machine : "unknown";
}

std::string envOr(const char* name, const std::string& dflt)
{
    const char* v = std::getenv(name);
    return v && *v ? v : dflt;
}

std::string braeBinaryPath()
{
    // Next to this binary, as brae's own dispatch does; then PATH.
    char buf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", buf, sizeof buf - 1);
    if (n > 0)
    {
        buf[n] = '\0';
        std::string self(buf);
        const std::size_t slash = self.find_last_of('/');
        if (slash != std::string::npos)
        {
            const std::string sibling = self.substr(0, slash) + "/brae";
            if (::access(sibling.c_str(), X_OK) == 0) return sibling;
        }
    }
    return envOr("BRAE_BINARY", "/usr/local/bin/brae");
}

void logLine(const std::string& m)
{
    // journald takes stdout; a timestamp would only duplicate what it already records.
    std::printf("%s\n", m.c_str());
    std::fflush(stdout);
}

std::string braeVersion()
{
    return envOr("BRAE_VERSION", "0.3.0");
}

int usage()
{
    std::printf(
"brae-agent, the Brae node service. Normally reached as `brae node ...`.\n\n"
"  brae-agent run                          run the node service (systemd does this)\n"
"  brae-agent node register --token <t>    join this machine to the Brae network\n"
"  brae-agent node status                  connection, GPUs and current job\n"
"  brae-agent node unregister              leave the network\n\n"
"Options for register:\n"
"  --token <t>       enrollment token from the dashboard (required)\n"
"  --api <url>       control plane URL (default %s)\n"
"  --name <name>     display name for this node (default: the GPU model)\n"
"  --no-service      do not install a systemd unit (for a foreground/dev run)\n\n"
"Environment:\n"
"  BRAE_API_URL         default control plane URL\n"
"  BRAE_IDENTITY_PATH   where the node identity lives (default /etc/brae/node.json)\n"
"  BRAE_BINARY          path to the brae solver binary\n",
        envOr("BRAE_API_URL", "https://api.brae.sh").c_str());
    return 0;
}

// ---- the daemon --------------------------------------------------------------------------------------------

int runDaemon()
{
    const std::string identityPath = defaultIdentityPath();
    const IdentityResult id = loadIdentity(identityPath);
    if (!id.ok)
    {
        std::fprintf(stderr, "brae-agent: %s\n", id.error.c_str());
        std::fprintf(stderr, "Register this machine first: sudo brae node register --token brae_ent_...\n");
        return 1;
    }

    CurlHttp http;
    http.setNodeId(id.identity.nodeId);
    http.setUserAgent(std::string("brae-agent/") + kAgentVersion);
    ApiClient api(http, id.identity.apiUrl);

    RunSettings run;
    run.braeBinary = braeBinaryPath();
    run.scratchRoot = envOr("BRAE_JOB_SCRATCH", "/var/lib/brae/jobs");
    auto executor = makeThreadedExecutor(run);

    AgentDeps deps;
    deps.api = &api;
    deps.executor = executor.get();
    deps.probeGpuState = [] { return probeGpuState(); };
    deps.monotonicSeconds = monotonicSeconds;
    deps.nowIso8601 = nowIso8601;
    deps.sleepFor = [](int s) { std::this_thread::sleep_for(std::chrono::seconds(s)); };
    deps.log = logLine;
    deps.pendingResultPath = envOr("BRAE_PENDING_RESULT", "/var/lib/brae/pending_result.json");
    deps.agentVersion = kAgentVersion;

    logLine("brae-agent " + std::string(kAgentVersion) + " starting as " + id.identity.nodeId
            + " (solver: " + run.braeBinary + ")");

    Agent agent(id.identity, std::move(deps));
    agent.run();                                    // returns only if stopped
    return 0;
}

// ---- installing the binaries where the service can see them -------------------------------------------------

namespace {

bool copyExecutable(const std::string& from, const std::string& to, std::string& error)
{
    // Write beside the target, then rename over it. Copying onto the destination directly fails with ETXTBSY
    // ("Text file busy") whenever the service is running from it -- which is exactly the case when someone
    // re-registers an already-installed node. rename(2) does not touch the running inode: the live process
    // keeps the old file, and the name points at the new one for the next start. Same trick identity.cpp uses.
    std::error_code ec;
    const std::filesystem::path dst(to);
    std::filesystem::create_directories(dst.parent_path(), ec);

    const std::filesystem::path tmp =
        dst.parent_path() / ("." + dst.filename().string() + ".new." + std::to_string(::getpid()));

    std::filesystem::copy_file(from, tmp, std::filesystem::copy_options::overwrite_existing, ec);
    if (ec)
    {
        error = "cannot write " + tmp.string() + ": " + ec.message();
        if (ec.value() == EACCES || ec.value() == EPERM) error += "\n(this needs root: use sudo)";
        return false;
    }

    std::filesystem::permissions(tmp, std::filesystem::perms::owner_all | std::filesystem::perms::group_read
                                      | std::filesystem::perms::group_exec | std::filesystem::perms::others_read
                                      | std::filesystem::perms::others_exec,
                                  std::filesystem::perm_options::replace, ec);
    if (ec)
    {
        error = "cannot make " + tmp.string() + " executable: " + ec.message();
        std::filesystem::remove(tmp, ec);
        return false;
    }

    std::filesystem::rename(tmp, dst, ec);
    if (ec)
    {
        error = "cannot install " + dst.string() + ": " + ec.message();
        std::error_code ignored;
        std::filesystem::remove(tmp, ignored);      // never leave a stray .brae-agent.new.<pid> behind
        return false;
    }
    return true;
}

}  // namespace

/// Copy the agent, and the solver beside it, into a directory the hardened unit can reach.
///
/// The solver comes too because the agent looks for `brae` as its own sibling (braeBinaryPath), so leaving it
/// behind would produce a service that starts, reports in happily, and then fails every job it accepts.
bool installForService(const std::string& selfPath, std::string& installedPath, std::string& error)
{
    const std::filesystem::path dir(systemInstallDir());
    const std::filesystem::path self(selfPath);
    installedPath = (dir / self.filename()).string();

    if (!copyExecutable(selfPath, installedPath, error)) return false;

    std::error_code ec;
    const std::filesystem::path solver = self.parent_path() / "brae";
    if (std::filesystem::exists(solver, ec))
    {
        std::string ignored;
        copyExecutable(solver.string(), (dir / "brae").string(), ignored);   // best effort: jobs need it, snapshots do not
    }
    return true;
}

// ---- the CLI verbs -----------------------------------------------------------------------------------------

int runNodeCommand(int argc, char** argv)
{
    std::string verb, token, apiUrl = envOr("BRAE_API_URL", "https://api.brae.sh"), name;
    bool noService = false;

    for (int i = 2; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--token" && i + 1 < argc) token = argv[++i];
        else if (a == "--api" && i + 1 < argc) apiUrl = argv[++i];
        else if (a == "--name" && i + 1 < argc) name = argv[++i];
        else if (a == "--no-service") noService = true;
        else if (a == "--help" || a == "-h") return usage();
        else if (a[0] != '-' && verb.empty()) verb = a;
        else { std::fprintf(stderr, "brae node: unknown option '%s'\n", a.c_str()); return 2; }
    }
    if (verb.empty()) return usage();

    const std::string identityPath = defaultIdentityPath();

    CurlHttp http;
    http.setUserAgent(std::string("brae-agent/") + kAgentVersion);
    ApiClient api(http, apiUrl);

    char selfBuf[4096];
    const ssize_t n = ::readlink("/proc/self/exe", selfBuf, sizeof selfBuf - 1);
    const std::string selfPath = n > 0 ? std::string(selfBuf, static_cast<std::size_t>(n))
                                       : std::string("/usr/local/bin/brae-agent");

    // The unit runs as the `brae` user with ProtectHome=true, so a build tree under $HOME is invisible to it:
    // pointing ExecStart there installs a service that loops on 203/EXEC forever while the node reads OFFLINE.
    // Copy the binaries somewhere the service can actually see, unless they already live there.
    std::string execPath = selfPath;
    if (verb == "register" && !noService && !isServiceReachablePath(selfPath))
    {
        std::string err;
        if (!installForService(selfPath, execPath, err))
        {
            std::fprintf(stderr, "brae node: %s\n", err.c_str());
            return 1;
        }
        std::fprintf(stderr, "brae: installed the agent to %s so the service can reach it\n", execPath.c_str());
    }

    std::unique_ptr<ServiceManager> service =
        noService ? makeNoopService()
                  : makeSystemdService(envOr("BRAE_UNIT_PATH", "/etc/systemd/system/brae-agent.service"),
                                       execPath);

    CliDeps d;
    d.api = &api;
    d.service = service.get();
    d.probeGpus = [] { return probeGpus(); };
    d.probeGpuState = [] { return probeGpuState(); };
    d.nowIso8601 = nowIso8601;
    d.out = [](const std::string& s) { std::fputs(s.c_str(), stdout); };
    d.err = [](const std::string& s) { std::fputs(s.c_str(), stderr); };
    d.identityPath = identityPath;
    d.apiUrl = apiUrl;
    d.agentVersion = kAgentVersion;
    d.braeVersion = braeVersion();
    d.displayName = name;
    d.architecture = machineArchitecture();
    d.sleepFor = [](int s) { std::this_thread::sleep_for(std::chrono::seconds(s)); };

    if (verb == "register")
    {
        // Verification asks the control plane whether this node has actually reported yet, so the command only
        // claims success once the node is genuinely on the network.
        d.verifyOnce = [&] {
            const IdentityResult saved = loadIdentity(identityPath);
            if (!saved.ok) return false;
            SnapshotRequest s;
            s.timestamp = nowIso8601();
            s.status = "available";
            s.acceptsJobs = true;
            s.agentVersion = kAgentVersion;
            s.gpus = probeGpuState();
            return api.sendSnapshot(saved.identity.nodeId, saved.identity.nodeToken, s).ok;
        };
        return cmdRegister(d, token);
    }
    if (verb == "status") return cmdStatus(d);
    if (verb == "unregister") return cmdUnregister(d);

    std::fprintf(stderr, "brae node: unknown command '%s'\n", verb.c_str());
    return 2;
}

}  // namespace

int main(int argc, char** argv)
{
    if (argc < 2) return usage();
    const std::string first = argv[1];
    if (first == "--help" || first == "-h") return usage();
    if (first == "run") return runDaemon();
    if (first == "node") return runNodeCommand(argc, argv);
    std::fprintf(stderr, "brae-agent: unknown command '%s'\n", first.c_str());
    return 2;
}
