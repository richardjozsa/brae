// `brae node register | status | unregister`.
//
// The criterion this file exists for is atomicity: a failure at ANY step of registration must leave no identity
// file, no service, and no registry row. A half-registered machine looks joined, never works, and gives its
// owner nothing to go on. So each step is made to fail in turn, and the aftermath is inspected.
#include "cli.h"
#include "json.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

using namespace brae::node;

static int failures = 0;

static void check(bool cond, const std::string& what)
{
    if (cond) { std::printf("ok:   %s\n", what.c_str()); return; }
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

class FakeHttp : public HttpClient
{
public:
    std::vector<HttpRequest> sent;
    std::vector<HttpResponse> queued;
    HttpResponse fallback;

    FakeHttp() { fallback.status = 200; fallback.body = "{}"; }
    void reply(long status, std::string body)
    {
        HttpResponse r; r.status = status; r.body = std::move(body); queued.push_back(std::move(r));
    }
    HttpResponse send(const HttpRequest& req) override
    {
        sent.push_back(req);
        if (queued.empty()) return fallback;
        HttpResponse r = queued.front();
        queued.erase(queued.begin());
        return r;
    }
    int countTo(const std::string& fragment) const
    {
        int n = 0;
        for (const HttpRequest& r : sent) if (r.url.find(fragment) != std::string::npos) ++n;
        return n;
    }
};

class FakeService : public ServiceManager
{
public:
    bool failInstall = false;
    bool failStart = false;
    bool installed = false;
    bool started = false;
    int removeCalls = 0;

    bool install(std::string& error) override
    {
        if (failInstall) { error = "cannot write the unit file"; return false; }
        installed = true;
        return true;
    }
    bool start(std::string& error) override
    {
        if (failStart) { error = "systemctl start failed"; return false; }
        started = true;
        return true;
    }
    bool stopAndRemove(std::string&) override
    {
        ++removeCalls;
        installed = false;
        started = false;
        return true;
    }
    std::string statusLine() override { return started ? "active (running)" : "inactive"; }
};

static std::string gTmp;

struct Harness
{
    FakeHttp http;
    FakeService service;
    std::unique_ptr<ApiClient> api;
    std::string identityPath;
    std::string output, errors;
    bool verifyResult = true;

    Harness()
    {
        identityPath = gTmp + "/node.json";
        ::unlink(identityPath.c_str());
        api = std::make_unique<ApiClient>(http, "https://api.brae.sh");
    }

    CliDeps deps()
    {
        CliDeps d;
        d.api = api.get();
        d.service = &service;
        d.identityPath = identityPath;
        d.apiUrl = "https://api.brae.sh";
        d.braeVersion = "0.3.0";
        d.architecture = "aarch64";
        d.nowIso8601 = [] { return std::string("2026-07-29T12:00:00Z"); };
        d.out = [this](const std::string& s) { output += s; };
        d.err = [this](const std::string& s) { errors += s; };
        d.sleepFor = [](int) {};
        d.verifyOnce = [this] { return verifyResult; };
        d.probeGpus = [] {
            GpuProbeResult r;
            r.nvmlAvailable = true;
            GpuIdentity g;
            g.index = 0;
            g.model = "NVIDIA GB10";
            g.vramMb = 124551;
            g.computeCapability = "12.1";
            g.driverVersion = "580.95.05";
            g.uuidHash = std::string(64, 'a');
            r.gpus.push_back(g);
            return r;
        };
        d.probeGpuState = [] {
            GpuState s;
            s.index = 0; s.utilizationPercent = 3; s.memoryUsedMb = 900;
            s.memoryTotalMb = 124551; s.temperatureCelsius = 38;
            return std::vector<GpuState>{s};
        };
        return d;
    }

    bool identityExists() const
    {
        struct stat st {};
        return ::stat(identityPath.c_str(), &st) == 0;
    }
    bool saidSomething(const std::string& fragment) const
    {
        return output.find(fragment) != std::string::npos || errors.find(fragment) != std::string::npos;
    }
};

static const char* kGoodRegister =
    R"({"node_id":"brae-7f21","node_token":"brae_nt_abc","snapshot_interval_idle_s":10,
        "snapshot_interval_running_s":5,"live_url":"https://brae.sh/live/node/brae-7f21"})";

// ---- the happy path ----------------------------------------------------------------------------------------

static void testRegisterSucceeds()
{
    Harness h;
    h.http.reply(201, kGoodRegister);
    check(cmdRegister(h.deps(), "brae_ent_x") == 0, "register: succeeds");
    check(h.identityExists(), "register: the identity file is written");
    check(h.service.installed && h.service.started, "register: the service is installed and started");
    check(h.saidSomething("brae-7f21"), "register: prints the node id");
    check(h.saidSomething("NVIDIA GB10"), "register: prints the GPU");
    check(h.saidSomething("122 GB"), "register: prints the VRAM");
    check(h.saidSomething("https://brae.sh/live/node/brae-7f21"), "register: prints the live URL");

    struct stat st {};
    ::stat(h.identityPath.c_str(), &st);
    check((st.st_mode & 07777) == 0600, "register: the identity file is 0600");
}

// ---- atomicity: every step made to fail in turn -------------------------------------------------------------

static void testNoGpuIsRefused()
{
    Harness h;
    CliDeps d = h.deps();
    d.probeGpus = [] { GpuProbeResult r; r.unavailableReason = "no driver"; return r; };
    check(cmdRegister(d, "brae_ent_x") != 0, "atomic: a machine with no driver is refused");
    check(!h.identityExists(), "atomic: nothing written");
    check(h.http.countTo("/register") == 0, "atomic: the control plane was never contacted");
    check(!h.service.installed, "atomic: no service installed");
}

static void testDriverButNoUsableGpu()
{
    Harness h;
    CliDeps d = h.deps();
    d.probeGpus = [] { GpuProbeResult r; r.nvmlAvailable = true; return r; };   // driver, zero cards
    check(cmdRegister(d, "brae_ent_x") != 0, "atomic: a driver with no usable GPU is refused");
    check(h.http.countTo("/register") == 0, "atomic: still no registration attempted");
}

static void testApiRefusalLeavesNothing()
{
    Harness h;
    h.http.reply(401, R"({"error":{"code":"unauthorized","message":"bad token"}})");
    check(cmdRegister(h.deps(), "wrong") != 0, "atomic: a rejected enrollment token fails");
    check(!h.identityExists(), "atomic: no identity file after an API refusal");
    check(!h.service.installed, "atomic: no service after an API refusal");
    check(h.saidSomething("unauthorized"), "atomic: the reason is shown");
}

static void testApi500LeavesNothing()
{
    Harness h;
    h.http.reply(500, R"({"error":{"code":"internal_error","message":"boom"}})");
    check(cmdRegister(h.deps(), "brae_ent_x") != 0, "atomic: a 500 at registration fails");
    check(!h.identityExists(), "atomic: no identity file after a 500");
    check(!h.service.installed, "atomic: no service after a 500");
}

static void testServiceFailureRollsBack()
{
    // The interesting one: registration SUCCEEDED server-side, then the service could not start. The node row
    // and the identity must both be undone, or the registry holds a machine that will never report.
    Harness h;
    h.http.reply(201, kGoodRegister);
    h.service.failStart = true;
    check(cmdRegister(h.deps(), "brae_ent_x") != 0, "atomic: a service that will not start fails the command");
    check(!h.identityExists(), "atomic: the identity file is removed again");
    check(h.service.removeCalls > 0, "atomic: the service is cleaned up");
    check(h.http.countTo("/unregister") == 1, "atomic: the control plane is told to undo the registration");
    check(h.saidSomething("rolled back"), "atomic: the user is told it was rolled back");
}

static void testInstallFailureRollsBack()
{
    Harness h;
    h.http.reply(201, kGoodRegister);
    h.service.failInstall = true;
    check(cmdRegister(h.deps(), "brae_ent_x") != 0, "atomic: a unit that cannot be installed fails");
    check(!h.identityExists(), "atomic: identity removed");
    check(h.http.countTo("/unregister") == 1, "atomic: registration undone server-side");
}

static void testUnsavableIdentityRollsBack()
{
    Harness h;
    CliDeps d = h.deps();
    d.identityPath = "/proc/definitely/not/writable/node.json";
    h.http.reply(201, kGoodRegister);
    check(cmdRegister(d, "brae_ent_x") != 0, "atomic: an identity that cannot be saved fails");
    check(h.http.countTo("/unregister") == 1,
          "atomic: a token we cannot persist is handed back rather than orphaning a registry row");
}

static void testDoubleRegistrationIsRefused()
{
    Harness h;
    h.http.reply(201, kGoodRegister);
    check(cmdRegister(h.deps(), "brae_ent_x") == 0, "atomic: first registration succeeds");
    const int registersBefore = h.http.countTo("/register");

    check(cmdRegister(h.deps(), "brae_ent_x") != 0, "atomic: registering twice is refused");
    check(h.http.countTo("/register") == registersBefore, "atomic: no second registration is attempted");
    check(h.saidSomething("already registered"), "atomic: says the machine is already registered");
    check(h.saidSomething("unregister"), "atomic: and how to undo it");
}

static void testMissingTokenIsRefused()
{
    Harness h;
    check(cmdRegister(h.deps(), "") == 2, "register: refuses without an enrollment token");
    check(h.http.countTo("/register") == 0, "register: nothing is sent without a token");
}

// ---- status -------------------------------------------------------------------------------------------------

static void testStatusUnregistered()
{
    Harness h;
    check(cmdStatus(h.deps()) != 0, "status: exits non-zero when not registered, so scripts can branch");
    check(h.saidSomething("Not registered"), "status: says so plainly");
    check(h.saidSomething("brae node register"), "status: says how to register");
}

static void testStatusRegistered()
{
    Harness h;
    h.http.reply(201, kGoodRegister);
    cmdRegister(h.deps(), "brae_ent_x");
    h.output.clear();

    check(cmdStatus(h.deps()) == 0, "status: exits zero when registered");
    check(h.saidSomething("brae-7f21"), "status: shows the node id");
    check(h.saidSomething("NVIDIA GB10"), "status: shows the GPU");
    check(h.saidSomething("active (running)"), "status: shows the service state");
    check(h.saidSomething("38 C"), "status: shows live GPU state");
    check(!h.saidSomething("brae_nt_"), "status: never prints the node token");
}

static void testStatusWorksWithTheApiDown()
{
    // The command someone runs precisely when things are broken must not need the control plane.
    Harness h;
    h.http.reply(201, kGoodRegister);
    cmdRegister(h.deps(), "brae_ent_x");
    h.output.clear();
    const int callsBefore = static_cast<int>(h.http.sent.size());

    check(cmdStatus(h.deps()) == 0, "status: works without contacting the control plane");
    check(static_cast<int>(h.http.sent.size()) == callsBefore, "status: makes no network calls at all");
}

// ---- unregister ------------------------------------------------------------------------------------------

static void testUnregister()
{
    Harness h;
    h.http.reply(201, kGoodRegister);
    cmdRegister(h.deps(), "brae_ent_x");
    h.output.clear();

    check(cmdUnregister(h.deps()) == 0, "unregister: succeeds");
    check(!h.identityExists(), "unregister: the identity file is gone");
    check(h.service.removeCalls > 0, "unregister: the service is stopped and removed");
    check(h.http.countTo("/unregister") == 1, "unregister: the control plane is told");
    check(h.saidSomething("history"), "unregister: explains the node stays in history");
}

static void testUnregisterWithApiDown()
{
    // A contributor must be able to leave even when the control plane is unreachable.
    Harness h;
    h.http.reply(201, kGoodRegister);
    cmdRegister(h.deps(), "brae_ent_x");
    h.output.clear();
    h.http.reply(500, R"({"error":{"code":"internal_error","message":"down"}})");

    check(cmdUnregister(h.deps()) != 0, "unregister: reports that the server was not told");
    check(!h.identityExists(), "unregister: but the local side is removed anyway");
    check(h.service.removeCalls > 0, "unregister: and the service is stopped anyway");
    check(h.saidSomething("still list"), "unregister: warns the node will linger server-side");
}

static void testUnregisterWhenNotRegistered()
{
    Harness h;
    check(cmdUnregister(h.deps()) != 0, "unregister: refuses when there is nothing to remove");
}

int main()
{
    char dir[] = "/tmp/brae-cli-testXXXXXX";
    if (!mkdtemp(dir)) { std::printf("FAIL: cannot create temp dir\n"); return 1; }
    gTmp = dir;

    testRegisterSucceeds();
    testNoGpuIsRefused();
    testDriverButNoUsableGpu();
    testApiRefusalLeavesNothing();
    testApi500LeavesNothing();
    testServiceFailureRollsBack();
    testInstallFailureRollsBack();
    testUnsavableIdentityRollsBack();
    testDoubleRegistrationIsRefused();
    testMissingTokenIsRefused();
    testStatusUnregistered();
    testStatusRegistered();
    testStatusWorksWithTheApiDown();
    testUnregister();
    testUnregisterWithApiDown();
    testUnregisterWhenNotRegistered();

    if (failures == 0) std::printf("PASS: brae node CLI\n");
    else std::printf("FAILED: %d check(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
