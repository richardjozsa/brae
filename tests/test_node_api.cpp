// The agent's protocol behaviour, against a fake transport.
//
// No network and no server: the point is to drive the cases a real orchestrator will not produce on request --
// a 500 at exactly the wrong moment, a body that is not JSON, JSON of the wrong shape, an assignment for a job
// type this agent has never heard of, a 401 after months of working fine. Those decide whether a node quietly
// stops working, and they have to be exercised deliberately.
#include "api_client.h"
#include "json.h"

#include <cstdio>
#include <string>
#include <vector>

using namespace brae::node;

static int failures = 0;

static void check(bool cond, const std::string& what)
{
    if (cond) { std::printf("ok:   %s\n", what.c_str()); return; }
    std::printf("FAIL: %s\n", what.c_str());
    ++failures;
}

static void checkEq(const std::string& got, const std::string& want, const std::string& what)
{
    if (got == want) { std::printf("ok:   %s\n", what.c_str()); return; }
    std::printf("FAIL: %s\n        got:  %s\n        want: %s\n", what.c_str(), got.c_str(), want.c_str());
    ++failures;
}

// A transport that returns whatever the test tells it to, and records what it was asked to send.
class FakeHttp : public HttpClient
{
public:
    std::vector<HttpRequest> sent;
    std::vector<HttpResponse> queued;

    void reply(long status, std::string body)
    {
        HttpResponse r;
        r.status = status;
        r.body = std::move(body);
        queued.push_back(std::move(r));
    }
    void failTransport(std::string why)
    {
        HttpResponse r;
        r.status = 0;
        r.error = std::move(why);
        queued.push_back(std::move(r));
    }
    HttpResponse send(const HttpRequest& req) override
    {
        sent.push_back(req);
        if (queued.empty()) return HttpResponse{};
        HttpResponse r = queued.front();
        queued.erase(queued.begin());
        return r;
    }
};

static RegisterRequest sampleRegister()
{
    RegisterRequest r;
    r.displayName = "workshop 3090";
    r.operatingSystem = "linux";
    r.architecture = "x86_64";
    r.agentVersion = "0.3.0";
    r.braeVersion = "0.3.0";
    GpuIdentity g;
    g.index = 0;
    g.model = "NVIDIA GeForce RTX 3090";
    g.vramMb = 24576;
    g.computeCapability = "8.6";
    g.driverVersion = "580.65.06";
    g.uuidHash = std::string(64, 'a');
    r.gpus.push_back(g);
    return r;
}

// ---- url joining ------------------------------------------------------------------------------------------

static void testUrlJoin()
{
    checkEq(urlJoin("https://api.brae.sh", "/v1/live"), "https://api.brae.sh/v1/live", "urljoin: plain");
    checkEq(urlJoin("https://api.brae.sh/", "/v1/live"), "https://api.brae.sh/v1/live",
            "urljoin: trailing slash does not double up");
    checkEq(urlJoin("https://api.brae.sh", "v1/live"), "https://api.brae.sh/v1/live",
            "urljoin: missing separator is added");
    checkEq(urlJoin("https://api.brae.sh/", "v1/live"), "https://api.brae.sh/v1/live", "urljoin: one of each");
}

// ---- bodies -----------------------------------------------------------------------------------------------

static void testBodies()
{
    const std::string body = buildRegisterBody(sampleRegister());
    Json j;
    std::string err;
    check(Json::parse(body, j, err), "register body: is valid JSON");
    checkEq(j["gpus"][0]["uuid_hash"].asString(), std::string(64, 'a'), "register body: sends the uuid HASH");
    check(body.find("\"uuid\":") == std::string::npos, "register body: never sends a raw uuid field");
    check(j["gpus"][0]["vram_mb"].asInt() == 24576, "register body: vram is an integer, not 24576.0");

    SnapshotRequest s;
    s.timestamp = "2026-07-29T12:00:00Z";
    s.status = "available";
    s.acceptsJobs = true;
    s.agentVersion = "0.3.0";
    GpuState g;
    g.index = 0; g.utilizationPercent = 91; g.memoryUsedMb = 18420; g.memoryTotalMb = 24576;
    g.temperatureCelsius = 72;
    s.gpus.push_back(g);

    check(Json::parse(buildSnapshotBody(s), j, err), "snapshot body: valid JSON when idle");
    check(j["job"].isNull(), "snapshot body: job is null when idle");
    check(j["gpus"][0]["utilization_percent"].asInt() == 91, "snapshot body: carries live gpu state");

    s.hasJob = true;
    s.jobId = "job-0182";
    s.jobType = JobType::BraeBenchmark;
    s.jobProgressPercent = 64;
    check(Json::parse(buildSnapshotBody(s), j, err), "snapshot body: valid JSON when running");
    checkEq(j["job"]["type"].asString(), "brae-benchmark", "snapshot body: job type by name");
    check(j["job"]["progress_percent"].asInt() == 64, "snapshot body: progress");

    JobOutcome o;
    o.jobId = "job-0182";
    o.success = true;
    o.resultJson = R"({"sample":"pimplefoam/pitzDaily-1M","runtime_s":41.5,"success":true})";
    check(Json::parse(buildResultBody(o, "2026-07-29T12:00:00Z", "2026-07-29T12:07:31Z"), j, err),
          "result body: valid JSON");
    checkEq(j["state"].asString(), "completed", "result body: completed");
    checkEq(j["result"]["sample"].asString(), "pimplefoam/pitzDaily-1M",
            "result body: passes brae's metrics through");

    // Unreadable metrics must not become invented ones.
    o.resultJson = "this is not json";
    check(Json::parse(buildResultBody(o, "", ""), j, err), "result body: still valid with unreadable metrics");
    check(j["result"].isNull(), "result body: unreadable metrics are sent as null, never guessed");

    o.success = false;
    o.error = "timeout";
    check(Json::parse(buildResultBody(o, "", ""), j, err), "result body: failure is valid JSON");
    checkEq(j["state"].asString(), "failed", "result body: failed");
    checkEq(j["error"].asString(), "timeout", "result body: carries the error class");

    o.error.clear();
    Json::parse(buildResultBody(o, "", ""), j, err);
    checkEq(j["error"].asString(), "unknown_failure", "result body: a failure always has some class");
}

// ---- registration -------------------------------------------------------------------------------------------

static void testRegister()
{
    {
        FakeHttp http;
        http.reply(201, R"({"node_id":"brae-7f21","node_token":"brae_nt_abc","snapshot_interval_idle_s":10,
                            "snapshot_interval_running_s":5,"live_url":"https://brae.sh/live/node/brae-7f21"})");
        ApiClient api(http, "https://api.brae.sh");
        const RegisterResult r = api.registerNode(sampleRegister(), "brae_ent_x");
        check(r.ok, "register: accepts a well-formed 201");
        checkEq(r.nodeId, "brae-7f21", "register: node id");
        checkEq(r.nodeToken, "brae_nt_abc", "register: token");
        checkEq(http.sent.at(0).url, "https://api.brae.sh/v1/nodes/register", "register: url");
        checkEq(http.sent.at(0).bearer, "brae_ent_x", "register: sends the ENROLLMENT token");
        checkEq(http.sent.at(0).method, "POST", "register: method");
    }
    {
        // A 201 with no identity is a server we cannot work with -- better to fail here than to install a
        // service that can never authenticate.
        FakeHttp http;
        http.reply(201, R"({"ok":true})");
        ApiClient api(http, "https://api.brae.sh");
        const RegisterResult r = api.registerNode(sampleRegister(), "t");
        check(!r.ok, "register: refuses a 201 with no node_id/node_token");
        check(r.error.find("no node_id") != std::string::npos, "register: says what was missing");
    }
    {
        FakeHttp http;
        http.reply(401, R"({"error":{"code":"unauthorized","message":"invalid enrollment token"}})");
        ApiClient api(http, "https://api.brae.sh");
        const RegisterResult r = api.registerNode(sampleRegister(), "wrong");
        check(!r.ok, "register: a rejected enrollment token fails");
        checkEq(r.error, "unauthorized", "register: surfaces the error CODE, not the prose");
    }
    {
        FakeHttp http;
        http.reply(409, R"({"error":{"code":"already_registered","message":"as brae-0001"}})");
        ApiClient api(http, "https://api.brae.sh");
        checkEq(api.registerNode(sampleRegister(), "t").error, "already_registered",
                "register: already-registered is reported by code");
    }
    {
        FakeHttp http;
        http.reply(500, "<html><body>502 Bad Gateway</body></html>");
        ApiClient api(http, "https://api.brae.sh");
        const RegisterResult r = api.registerNode(sampleRegister(), "t");
        check(!r.ok, "register: a proxy's HTML error page does not crash the parser");
        checkEq(r.error, "http_500", "register: falls back to the status when there is no envelope");
    }
    {
        FakeHttp http;
        http.failTransport("could not resolve api.brae.sh");
        ApiClient api(http, "https://api.brae.sh");
        const RegisterResult r = api.registerNode(sampleRegister(), "t");
        check(!r.ok, "register: a transport failure fails");
        check(r.error.find("transport:") == 0, "register: transport failures are distinguishable");
    }
}

// ---- snapshots ----------------------------------------------------------------------------------------------

static void testSnapshot()
{
    SnapshotRequest s;
    s.timestamp = "2026-07-29T12:00:00Z";
    s.status = "available";
    s.agentVersion = "0.3.0";

    {
        FakeHttp http;
        http.reply(200, R"({"accepts_jobs":true,"snapshot_interval_s":10,"assignment":null,
                            "agent_upgrade":null})");
        ApiClient api(http, "https://api.brae.sh");
        const SnapshotResult r = api.sendSnapshot("brae-7f21", "tok", s);
        check(r.ok, "snapshot: accepts an idle reply");
        check(!r.assignment.present, "snapshot: no assignment");
        check(r.snapshotIntervalS == 10, "snapshot: interval");
        checkEq(http.sent.at(0).url, "https://api.brae.sh/v1/nodes/brae-7f21/snapshot", "snapshot: url");
        checkEq(http.sent.at(0).bearer, "tok", "snapshot: sends the NODE token");
    }
    {
        FakeHttp http;
        http.reply(200, R"({"accepts_jobs":false,"snapshot_interval_s":5,
            "assignment":{"job_id":"job-0182","type":"brae-benchmark",
                          "params":{"sample":"pimplefoam/pitzDaily-1M"},
                          "deadline":"2026-07-29T13:00:00Z"}})");
        ApiClient api(http, "https://api.brae.sh");
        const SnapshotResult r = api.sendSnapshot("brae-7f21", "tok", s);
        check(r.assignment.present, "snapshot: takes a well-formed assignment");
        checkEq(r.assignment.jobId, "job-0182", "snapshot: job id");
        checkEq(r.assignment.sample, "pimplefoam/pitzDaily-1M", "snapshot: sample");
        check(!r.acceptsJobs, "snapshot: obeys the dashboard toggle");
    }
    {
        // The server clamps too, but the agent does not rely on that: a control plane saying "every 0 seconds"
        // must not turn every contributor's machine into a request flood.
        FakeHttp http;
        http.reply(200, R"({"accepts_jobs":true,"snapshot_interval_s":0})");
        ApiClient api(http, "https://api.brae.sh");
        check(api.sendSnapshot("n", "t", s).snapshotIntervalS == 5, "snapshot: clamps a zero interval up");

        FakeHttp http2;
        http2.reply(200, R"({"accepts_jobs":true,"snapshot_interval_s":86400})");
        ApiClient api2(http2, "https://api.brae.sh");
        check(api2.sendSnapshot("n", "t", s).snapshotIntervalS == 60, "snapshot: clamps a huge interval down");
    }
    {
        // An assignment this agent cannot fully understand is LEFT UNTAKEN: it expires and is offered
        // elsewhere, which beats accepting work we cannot run.
        const char* bad[] = {
            R"({"assignment":{"job_id":"j","type":"run-shell","params":{"sample":"x"}}})",
            R"({"assignment":{"job_id":"j","type":"brae-benchmark","params":{"sample":"../../etc"}}})",
            R"({"assignment":{"job_id":"","type":"brae-benchmark","params":{"sample":"ok"}}})",
            R"({"assignment":{"job_id":"j","type":"brae-benchmark","params":{}}})",
        };
        for (const char* body : bad)
        {
            FakeHttp http;
            http.reply(200, body);
            ApiClient api(http, "https://api.brae.sh");
            const SnapshotResult r = api.sendSnapshot("n", "t", s);
            check(r.ok, "snapshot: an unusable assignment still leaves the snapshot itself valid");
            check(!r.assignment.present, std::string("snapshot: refuses assignment ") + body);
        }
    }
    {
        FakeHttp http;
        http.reply(401, R"({"error":{"code":"unauthorized","message":"token does not match"}})");
        ApiClient api(http, "https://api.brae.sh");
        const SnapshotResult r = api.sendSnapshot("n", "bad", s);
        check(!r.ok, "snapshot: 401 fails");
        check(r.authRejected, "snapshot: 401 is flagged so the loop can stop instead of hammering");
    }
    {
        FakeHttp http;
        http.reply(200, "not json");
        ApiClient api(http, "https://api.brae.sh");
        const SnapshotResult r = api.sendSnapshot("n", "t", s);
        check(!r.ok, "snapshot: a 200 that is not JSON is a failure, not an empty success");
        check(!r.assignment.present, "snapshot: and carries no assignment");
    }
    {
        FakeHttp http;
        http.reply(200, "{}");
        ApiClient api(http, "https://api.brae.sh");
        const SnapshotResult r = api.sendSnapshot("n", "t", s);
        check(r.ok, "snapshot: an empty object is acceptable");
        check(r.acceptsJobs, "snapshot: accepts_jobs defaults to true when absent");
        check(r.snapshotIntervalS == 10, "snapshot: interval defaults when absent");
    }
}

// ---- accept / result / unregister -------------------------------------------------------------------------

static void testJobCalls()
{
    {
        FakeHttp http;
        http.reply(200, R"({"accepted":"job-0182"})");
        ApiClient api(http, "https://api.brae.sh");
        check(api.acceptJob("brae-7f21", "tok", "job-0182").ok, "accept: ok");
        checkEq(http.sent.at(0).url, "https://api.brae.sh/v1/jobs/job-0182/accept?node_id=brae-7f21",
                "accept: url carries the node id");
    }
    {
        // Someone else took it. Not an error worth retrying or logging loudly -- just drop it.
        FakeHttp http;
        http.reply(409, R"({"error":{"code":"node_busy","message":"node already holds job-0001"}})");
        ApiClient api(http, "https://api.brae.sh");
        const SimpleResult r = api.acceptJob("n", "t", "job-0182");
        check(!r.ok, "accept: 409 is not success");
        check(r.conflict, "accept: 409 is flagged as a conflict, distinct from a failure");
    }
    {
        FakeHttp http;
        http.reply(200, "{}");
        ApiClient api(http, "https://api.brae.sh");
        JobOutcome o;
        o.jobId = "job-0182";
        o.success = true;
        o.resultJson = R"({"runtime_s":41.5})";
        check(api.reportResult("brae-7f21", "tok", o, "2026-07-29T12:00:00Z", "2026-07-29T12:07:31Z").ok,
              "result: ok");
        checkEq(http.sent.at(0).url, "https://api.brae.sh/v1/jobs/job-0182/result?node_id=brae-7f21",
                "result: url");
    }
    {
        FakeHttp http;
        http.reply(200, "{}");
        ApiClient api(http, "https://api.brae.sh");
        check(api.unregisterNode("brae-7f21", "tok").ok, "unregister: ok");
        checkEq(http.sent.at(0).url, "https://api.brae.sh/v1/nodes/brae-7f21/unregister", "unregister: url");
    }
}

// ---- the token must not travel anywhere it should not -------------------------------------------------------

static void testTokenHandling()
{
    FakeHttp http;
    http.reply(200, "{}");
    ApiClient api(http, "https://api.brae.sh");
    SnapshotRequest s;
    s.timestamp = "2026-07-29T12:00:00Z";
    s.status = "available";
    s.agentVersion = "0.3.0";
    api.sendSnapshot("brae-7f21", "brae_nt_supersecret", s);

    const HttpRequest& req = http.sent.at(0);
    check(req.body.find("brae_nt_supersecret") == std::string::npos,
          "token: never appears in the request BODY, only the Authorization header");
    check(req.url.find("brae_nt_supersecret") == std::string::npos,
          "token: never appears in the URL, where it would land in every proxy log");
}

int main()
{
    testUrlJoin();
    testBodies();
    testRegister();
    testSnapshot();
    testJobCalls();
    testTokenHandling();

    if (failures == 0) std::printf("PASS: brae-agent protocol tests\n");
    else std::printf("FAILED: %d check(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
