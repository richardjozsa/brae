// The agent loop, driven at exact instants.
//
// Fake transport, fake executor, fake clock -- no threads, no sleeps, no network. That makes the awkward cases
// assertable rather than hopeful: a job that outlives several heartbeats, a result POST that fails four times
// and then succeeds, a process that dies between finishing a job and reporting it, a 401 arriving after months
// of working fine.
#include "agent.h"
#include "json.h"

#include <cstdio>
#include <cstdlib>
#include <string>
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

static void checkEq(const std::string& got, const std::string& want, const std::string& what)
{
    if (got == want) { std::printf("ok:   %s\n", what.c_str()); return; }
    std::printf("FAIL: %s\n        got:  %s\n        want: %s\n", what.c_str(), got.c_str(), want.c_str());
    ++failures;
}

// ---- fakes ----------------------------------------------------------------------------------------------

class FakeHttp : public HttpClient
{
public:
    std::vector<HttpRequest> sent;
    std::vector<HttpResponse> queued;
    HttpResponse fallback;                       // used once `queued` runs out

    FakeHttp() { fallback.status = 200; fallback.body = R"({"accepts_jobs":true,"snapshot_interval_s":10})"; }

    void reply(long status, std::string body)
    {
        HttpResponse r; r.status = status; r.body = std::move(body); queued.push_back(std::move(r));
    }
    void failTransport(std::string why)
    {
        HttpResponse r; r.status = 0; r.error = std::move(why); queued.push_back(std::move(r));
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
    const HttpRequest* lastTo(const std::string& fragment) const
    {
        for (auto it = sent.rbegin(); it != sent.rend(); ++it)
            if (it->url.find(fragment) != std::string::npos) return &*it;
        return nullptr;
    }
};

// Completes after a set number of polls, so "the job outlives three heartbeats" is exact.
class FakeExecutor : public JobExecutor
{
public:
    int pollsUntilDone = 1;
    JobOutcome outcome;
    JobRequest started;
    int startCount = 0;

    void start(const JobRequest& job) override
    {
        started = job;
        ++startCount;
        polls_ = 0;
        running_ = true;
        collected_ = false;
    }
    bool running() const override { return running_; }
    int progressPercent() const override { return running_ ? progress_ : 100; }
    bool takeOutcome(JobOutcome& out) override
    {
        if (running_ || collected_) return false;
        out = outcome;
        collected_ = true;
        return true;
    }
    void poll()
    {
        if (!running_) return;
        if (++polls_ >= pollsUntilDone) { running_ = false; progress_ = 100; }
        else progress_ = (100 * polls_) / (pollsUntilDone ? pollsUntilDone : 1);
    }

private:
    bool running_ = false;
    bool collected_ = true;
    int polls_ = 0;
    int progress_ = 0;
};

struct Harness
{
    FakeHttp http;
    FakeExecutor executor;
    long long now = 1000;
    std::vector<std::string> logs;
    std::string pendingPath;

    std::unique_ptr<ApiClient> api;
    std::unique_ptr<Agent> agent;

    explicit Harness(const std::string& tmpdir)
    {
        pendingPath = tmpdir + "/pending_result.json";
        api = std::make_unique<ApiClient>(http, "https://api.brae.sh");

        Identity id;
        id.nodeId = "brae-7f21";
        id.nodeToken = "brae_nt_secret";
        id.apiUrl = "https://api.brae.sh";

        AgentDeps d;
        d.api = api.get();
        d.executor = &executor;
        d.probeGpuState = [] {
            GpuState g;
            g.index = 0; g.utilizationPercent = 12; g.memoryUsedMb = 900;
            g.memoryTotalMb = 24576; g.temperatureCelsius = 41;
            return std::vector<GpuState>{g};
        };
        d.monotonicSeconds = [this] { return now; };
        d.nowIso8601 = [] { return std::string("2026-07-29T12:00:00Z"); };
        d.sleepFor = [this](int s) { now += s; };
        d.log = [this](const std::string& m) { logs.push_back(m); };
        d.pendingResultPath = pendingPath;
        agent = std::make_unique<Agent>(id, std::move(d));
    }

    bool logged(const std::string& fragment) const
    {
        for (const std::string& l : logs) if (l.find(fragment) != std::string::npos) return true;
        return false;
    }
};

static std::string assignmentBody(const char* jobId = "job-0182",
                                  const char* sample = "pimplefoam/pitzDaily-1M",
                                  bool acceptsJobs = true)
{
    return std::string(R"({"accepts_jobs":)") + (acceptsJobs ? "true" : "false")
         + R"(,"snapshot_interval_s":5,"assignment":{"job_id":")" + jobId
         + R"(","type":"brae-benchmark","params":{"sample":")" + sample
         + R"("},"deadline":"2026-07-29T13:00:00Z"}})";
}

static std::string gTmp;

// ---- heartbeat ----------------------------------------------------------------------------------------------

static void testIdleHeartbeat()
{
    Harness h(gTmp);
    h.http.reply(200, R"({"accepts_jobs":true,"snapshot_interval_s":10,"assignment":null})");
    check(h.agent->tick() == TickOutcome::Continue, "idle: a tick continues");

    const HttpRequest* snap = h.http.lastTo("/snapshot");
    check(snap != nullptr, "idle: a snapshot was sent");
    Json j;
    std::string err;
    check(Json::parse(snap->body, j, err), "idle: snapshot body is valid JSON");
    checkEq(j["status"].asString(), "available", "idle: reports available");
    check(j["job"].isNull(), "idle: no job");
    check(j["gpus"][0]["memory_total_mb"].asInt() == 24576, "idle: carries live GPU state");
    check(h.agent->snapshotIntervalSeconds() == 10, "idle: adopts the server's interval");
}

static void testToggleOff()
{
    Harness h(gTmp);
    h.http.reply(200, R"({"accepts_jobs":false,"snapshot_interval_s":10})");
    h.agent->tick();
    check(!h.agent->acceptsJobs(), "toggle: the agent obeys the dashboard switch");

    // accepts_jobs stays false in the same reply that carries the offer: the response is authoritative and is
    // applied before the offer is considered, so a server that says "off" and offers work anyway is refused.
    h.http.reply(200, assignmentBody("job-0182", "pimplefoam/pitzDaily-1M", /*acceptsJobs=*/false));
    h.agent->tick();
    check(h.http.countTo("/accept") == 0, "toggle: with acceptance off, an offer is not taken");
    check(h.logged("job acceptance is off"), "toggle: and it says why");
}

// ---- a job across several heartbeats -----------------------------------------------------------------------

static void testJobLifecycle()
{
    Harness h(gTmp);
    h.executor.pollsUntilDone = 3;                 // the job outlives three heartbeats
    h.executor.outcome.jobId = "job-0182";
    h.executor.outcome.success = true;
    h.executor.outcome.runtimeSeconds = 41.5;
    h.executor.outcome.resultJson = R"({"sample":"pimplefoam/pitzDaily-1M","runtime_s":41.5})";

    h.http.reply(200, assignmentBody());
    h.http.reply(200, R"({"accepted":"job-0182"})");
    h.agent->tick();

    check(h.http.countTo("/accept") == 1, "job: the offer was accepted");
    check(h.executor.startCount == 1, "job: execution started");
    checkEq(h.executor.started.sample, "pimplefoam/pitzDaily-1M", "job: the sample was passed through");
    check(h.agent->busy(), "job: the node is busy");

    // While it runs, heartbeats must keep going -- otherwise the node is marked offline mid-job and the work
    // is abandoned out from under it.
    for (int i = 0; i < 2; ++i)
    {
        h.executor.poll();
        h.agent->tick();
        const HttpRequest* snap = h.http.lastTo("/snapshot");
        Json j;
        std::string err;
        Json::parse(snap->body, j, err);
        checkEq(j["status"].asString(), "running", "job: heartbeats continue and report running");
        checkEq(j["job"]["job_id"].asString(), "job-0182", "job: the heartbeat names the job");
        check(j["job"]["progress_percent"].asInt() > 0, "job: progress is reported");
    }

    // A second offer while busy is refused locally, without asking the server.
    const int acceptsBefore = h.http.countTo("/accept");
    h.http.reply(200, assignmentBody("job-0999"));
    h.agent->tick();
    check(h.http.countTo("/accept") == acceptsBefore, "job: a second offer is refused without contacting the server");
    check(h.logged("already holds"), "job: and it says the node is busy");

    h.executor.poll();                             // finishes
    h.agent->tick();
    check(!h.agent->busy(), "job: the node is free again");
    const HttpRequest* result = h.http.lastTo("/result");
    check(result != nullptr, "job: the result was reported");
    Json j;
    std::string err;
    Json::parse(result->body, j, err);
    checkEq(j["state"].asString(), "completed", "job: reported as completed");
    checkEq(j["result"]["sample"].asString(), "pimplefoam/pitzDaily-1M", "job: brae's metrics were forwarded");
    check(!h.agent->hasPendingResult(), "job: the pending file is cleared once acknowledged");
}

// ---- the result must survive ----------------------------------------------------------------------------------

static void testResultRetries()
{
    Harness h(gTmp);
    h.agent->clearPendingResult();
    h.executor.pollsUntilDone = 1;
    h.executor.outcome.jobId = "job-0182";
    h.executor.outcome.success = true;
    h.executor.outcome.resultJson = R"({"runtime_s":1})";

    h.http.reply(200, assignmentBody());
    h.http.reply(200, R"({"accepted":"job-0182"})");
    h.agent->tick();
    h.executor.poll();

    // The report fails. The result must not be lost.
    h.http.failTransport("connection refused");
    h.agent->tick();
    check(h.agent->hasPendingResult(), "retry: an undelivered result is persisted");
    check(h.logged("retrying in 2s"), "retry: first retry is scheduled at 2s");

    // Inside the backoff window nothing is sent.
    const int before = h.http.countTo("/result");
    h.agent->tick();
    check(h.http.countTo("/result") == before, "retry: nothing is sent inside the backoff window");

    // After it, one attempt, and the delay grows.
    h.now += 2;
    h.http.failTransport("connection refused");
    h.agent->tick();
    check(h.http.countTo("/result") == before + 1, "retry: one attempt after the window");
    check(h.logged("retrying in 4s"), "retry: the delay doubles");

    // A node with an undelivered result must not take new work and risk losing it.
    // Note the queue order: within a tick the pending result is POSTed BEFORE the snapshot, so the first
    // queued reply answers the result attempt and the second answers the snapshot.
    h.now += 4;
    h.http.failTransport("still down");                              // answers the result retry
    h.http.reply(200, assignmentBody("job-0999"));                   // answers the snapshot
    h.agent->tick();
    check(h.executor.startCount == 1, "retry: no new job is taken while a result is outstanding");
    check(h.logged("has not been delivered"), "retry: and it says why");

    // Eventually it lands.
    h.now += 60;
    h.agent->tick();
    check(!h.agent->hasPendingResult(), "retry: the result is delivered and the file cleared");
}

static void testResultSurvivesRestart()
{
    // The process dies between finishing a job and reporting it. A fresh agent must pick the result up.
    JobOutcome o;
    o.jobId = "job-0182";
    o.success = false;
    o.error = "timeout";

    {
        Harness first(gTmp);
        check(first.agent->savePendingResult(o, "2026-07-29T12:00:00Z", "2026-07-29T12:30:00Z"),
              "restart: the result is written to disk");
    }
    {
        Harness second(gTmp);                      // a brand-new agent, as after a systemd restart
        check(second.agent->hasPendingResult(), "restart: a fresh agent finds the pending result");
        second.agent->tick();
        const HttpRequest* result = second.http.lastTo("/result");
        check(result != nullptr, "restart: it is reported on the next tick");
        Json j;
        std::string err;
        Json::parse(result->body, j, err);
        checkEq(j["state"].asString(), "failed", "restart: the outcome survived intact");
        checkEq(j["error"].asString(), "timeout", "restart: including the error class");
        check(!second.agent->hasPendingResult(), "restart: and is then cleared");
    }
}

static void testDuplicateReportIsSettled()
{
    // The server got the result but the acknowledgement was lost. A 409 means "already recorded", so the file
    // must be cleared -- otherwise the agent retries it forever and never takes another job.
    Harness h(gTmp);
    JobOutcome o;
    o.jobId = "job-0182";
    o.success = true;
    h.agent->savePendingResult(o, "", "");
    h.http.reply(409, R"({"error":{"code":"job_finished","message":"already completed"}})");
    h.agent->tick();
    check(!h.agent->hasPendingResult(), "duplicate: a 409 settles the pending result rather than looping");
    check(h.logged("already recorded"), "duplicate: and says so");
}

// ---- failures ---------------------------------------------------------------------------------------------

static void testAuthRejection()
{
    Harness h(gTmp);
    h.agent->clearPendingResult();
    h.http.reply(401, R"({"error":{"code":"unauthorized","message":"token does not match"}})");
    check(h.agent->tick() == TickOutcome::AuthRejected, "401: the loop is told to stop");
    check(h.logged("rejected this node's token"), "401: the log says what happened");
    check(h.logged("brae node register"), "401: and how to fix it");
}

static void testTransportOutage()
{
    Harness h(gTmp);
    h.agent->clearPendingResult();
    for (int i = 0; i < 5; ++i)
    {
        h.http.failTransport("could not resolve api.brae.sh");
        check(h.agent->tick() == TickOutcome::Continue, "outage: the loop keeps going");
        h.now += 10;
    }
    check(h.http.countTo("/snapshot") == 5, "outage: every interval is still attempted, none are queued");

    int complaints = 0;
    for (const std::string& l : h.logs) if (l.find("cannot reach") != std::string::npos) ++complaints;
    check(complaints == 1, "outage: the failure is logged once a minute, not once a tick");
}

static void testUnusableAssignmentIsIgnored()
{
    Harness h(gTmp);
    h.agent->clearPendingResult();
    h.http.reply(200, assignmentBody("job-0182", "../../etc/passwd"));
    h.agent->tick();
    check(h.http.countTo("/accept") == 0, "bad assignment: never accepted");
    check(h.executor.startCount == 0, "bad assignment: nothing is executed");
    check(h.agent->tick() == TickOutcome::Continue, "bad assignment: the loop is unharmed");
}

static void testConflictOnAccept()
{
    Harness h(gTmp);
    h.agent->clearPendingResult();
    h.http.reply(200, assignmentBody());
    h.http.reply(409, R"({"error":{"code":"node_busy","message":"taken"}})");
    h.agent->tick();
    check(h.executor.startCount == 0, "conflict: a refused acceptance does not start work");
    check(!h.agent->busy(), "conflict: the node stays free");
}

int main()
{
    char dir[] = "/tmp/brae-loop-testXXXXXX";
    if (!mkdtemp(dir)) { std::printf("FAIL: cannot create temp dir\n"); return 1; }
    gTmp = dir;

    testIdleHeartbeat();
    testToggleOff();
    testJobLifecycle();
    testResultRetries();
    testResultSurvivesRestart();
    testDuplicateReportIsSettled();
    testAuthRejection();
    testTransportOutage();
    testUnusableAssignmentIsIgnored();
    testConflictOnAccept();

    if (failures == 0) std::printf("PASS: brae-agent loop\n");
    else std::printf("FAILED: %d check(s)\n", failures);
    return failures == 0 ? 0 : 1;
}
