#include "agent.h"

#include "backoff.h"
#include "json.h"

#include <atomic>
#include <fstream>
#include <mutex>
#include <sstream>
#include <thread>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace brae::node {
namespace {

class ThreadedExecutor : public JobExecutor
{
public:
    explicit ThreadedExecutor(RunSettings s) : settings_(std::move(s)) {}

    ~ThreadedExecutor() override
    {
        if (worker_.joinable()) worker_.join();
    }

    void start(const JobRequest& job) override
    {
        if (running_) return;
        if (worker_.joinable()) worker_.join();      // reap the previous one
        running_ = true;
        collected_ = false;
        progress_ = 0;
        worker_ = std::thread([this, job] {
            JobOutcome o = runJob(job, settings_, [this](int p) { progress_ = p; });
            {
                std::lock_guard<std::mutex> lock(mutex_);
                outcome_ = std::move(o);
            }
            running_ = false;
        });
    }

    bool running() const override { return running_; }
    int progressPercent() const override { return progress_; }

    bool takeOutcome(JobOutcome& out) override
    {
        if (running_ || collected_) return false;
        if (!worker_.joinable()) return false;
        worker_.join();
        std::lock_guard<std::mutex> lock(mutex_);
        out = outcome_;
        collected_ = true;
        return true;
    }

private:
    RunSettings settings_;
    std::thread worker_;
    std::atomic<bool> running_{false};
    std::atomic<int> progress_{0};
    bool collected_ = true;
    mutable std::mutex mutex_;
    JobOutcome outcome_;
};

}  // namespace

std::unique_ptr<JobExecutor> makeThreadedExecutor(RunSettings settings)
{
    return std::make_unique<ThreadedExecutor>(std::move(settings));
}

Agent::Agent(Identity identity, AgentDeps deps) : id_(std::move(identity)), deps_(std::move(deps))
{
    if (!deps_.log) deps_.log = [](const std::string&) {};
}

bool Agent::busy() const
{
    return !runningJobId_.empty() || (deps_.executor && deps_.executor->running());
}

// ---- pending result ---------------------------------------------------------------------------------------

bool Agent::savePendingResult(const JobOutcome& o, const std::string& startedAt, const std::string& finishedAt)
{
    Json j = Json::object();
    j.set("job_id", Json::str(o.jobId));
    j.set("success", Json::boolean(o.success));
    j.set("runtime_s", Json::num(o.runtimeSeconds));
    j.set("result_json", Json::str(o.resultJson));
    j.set("error", Json::str(o.error));
    j.set("started_at", Json::str(startedAt));
    j.set("finished_at", Json::str(finishedAt));

    const std::string tmp = deps_.pendingResultPath + ".tmp";
    const int fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd < 0) return false;
    const std::string text = j.dump() + "\n";
    const ssize_t n = ::write(fd, text.data(), text.size());
    ::fsync(fd);                                   // the point of the file is surviving a crash
    ::close(fd);
    if (n != static_cast<ssize_t>(text.size())) { ::unlink(tmp.c_str()); return false; }
    return ::rename(tmp.c_str(), deps_.pendingResultPath.c_str()) == 0;
}

bool Agent::loadPendingResult(JobOutcome& o, std::string& startedAt, std::string& finishedAt) const
{
    std::ifstream f(deps_.pendingResultPath);
    if (!f) return false;
    const std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    Json j;
    std::string err;
    if (!Json::parse(text, j, err)) return false;
    o.jobId = j["job_id"].asString();
    if (o.jobId.empty()) return false;
    o.success = j["success"].asBool();
    o.runtimeSeconds = j["runtime_s"].asNumber();
    o.resultJson = j["result_json"].asString();
    o.error = j["error"].asString();
    startedAt = j["started_at"].asString();
    finishedAt = j["finished_at"].asString();
    return true;
}

void Agent::clearPendingResult()
{
    ::unlink(deps_.pendingResultPath.c_str());
}

bool Agent::hasPendingResult() const
{
    JobOutcome o;
    std::string a, b;
    return loadPendingResult(o, a, b);
}

void Agent::reportPendingIfAny()
{
    JobOutcome o;
    std::string startedAt, finishedAt;
    if (!loadPendingResult(o, startedAt, finishedAt)) return;

    const long long now = deps_.monotonicSeconds();
    if (now < nextPendingAttemptAt_) return;       // still inside the backoff window

    const SimpleResult r = deps_.api->reportResult(id_.nodeId, id_.nodeToken, o, startedAt, finishedAt);
    if (r.ok || r.conflict)
    {
        // conflict == the server already has it (a retry after a lost acknowledgement). Either way it is
        // recorded, so the file has done its job.
        deps_.log("reported result for " + o.jobId + (r.conflict ? " (already recorded)" : ""));
        clearPendingResult();
        pendingAttempts_ = 0;
        nextPendingAttemptAt_ = 0;
        return;
    }

    // Never give up: this is the only record that the work happened.
    const int delay = resultBackoffSeconds(pendingAttempts_);
    ++pendingAttempts_;
    nextPendingAttemptAt_ = now + delay;
    deps_.log("could not report " + o.jobId + " (" + r.error + "); retrying in " + std::to_string(delay) + "s");
}

// ---- jobs -------------------------------------------------------------------------------------------------

void Agent::maybeTakeJob(const Assignment& a)
{
    if (!a.present) return;

    // Invariant 1, enforced locally. The scheduler should never offer work to a busy node, but the agent does
    // not rely on that being true -- the second guard is the point.
    if (busy())
    {
        deps_.log("ignoring " + a.jobId + ": this node already holds " + runningJobId_);
        return;
    }
    if (!acceptsJobs_)
    {
        deps_.log("ignoring " + a.jobId + ": job acceptance is off");
        return;
    }
    if (hasPendingResult())
    {
        // An undelivered result from a previous job is still outstanding; taking new work would risk losing it.
        deps_.log("ignoring " + a.jobId + ": a previous result has not been delivered yet");
        return;
    }

    const SimpleResult accepted = deps_.api->acceptJob(id_.nodeId, id_.nodeToken, a.jobId);
    if (!accepted.ok)
    {
        // 409 means someone else got there first, or it is already finished. Not worth logging loudly.
        deps_.log("did not take " + a.jobId + ": " + accepted.error);
        return;
    }

    JobRequest job;
    job.jobId = a.jobId;
    job.type = a.type;
    job.sample = a.sample;
    runningJobId_ = a.jobId;
    jobStartedAt_ = deps_.nowIso8601();
    deps_.executor->start(job);
    deps_.log("started " + a.jobId + " (" + a.sample + ")");
}

void Agent::collectFinishedJob()
{
    if (runningJobId_.empty() || !deps_.executor) return;
    JobOutcome outcome;
    if (!deps_.executor->takeOutcome(outcome)) return;

    const std::string finishedAt = deps_.nowIso8601();
    deps_.log("finished " + outcome.jobId + (outcome.success ? " ok" : " failed: " + outcome.error));

    // Disk first, then the network. If the process dies in between, the next start picks this up.
    savePendingResult(outcome, jobStartedAt_, finishedAt);
    runningJobId_.clear();
    jobStartedAt_.clear();
    pendingAttempts_ = 0;
    nextPendingAttemptAt_ = 0;
    reportPendingIfAny();
}

// ---- the tick ------------------------------------------------------------------------------------------------

TickOutcome Agent::tick()
{
    collectFinishedJob();
    reportPendingIfAny();

    SnapshotRequest s;
    s.timestamp = deps_.nowIso8601();
    s.agentVersion = deps_.agentVersion;
    s.acceptsJobs = acceptsJobs_;
    s.gpus = deps_.probeGpuState ? deps_.probeGpuState() : std::vector<GpuState>{};

    if (busy())
    {
        s.status = "running";
        s.hasJob = true;
        s.jobId = runningJobId_;
        s.jobType = JobType::BraeBenchmark;
        s.jobProgressPercent = deps_.executor ? deps_.executor->progressPercent() : 0;
    }
    else
    {
        s.status = acceptsJobs_ ? "available" : "online";
    }

    const SnapshotResult r = deps_.api->sendSnapshot(id_.nodeId, id_.nodeToken, s);

    if (r.authRejected)
    {
        // Stop the loop but not the process: `brae node status` must be able to say why this node is silent.
        deps_.log("the control plane rejected this node's token; not retrying. "
                  "Re-register with: sudo brae node unregister && sudo brae node register");
        return TickOutcome::AuthRejected;
    }

    if (!r.ok)
    {
        // Do NOT back off and do NOT queue: the next snapshot supersedes this one, so retrying at the normal
        // interval is both correct and self-limiting. Log at most once a minute so an outage does not fill
        // the journal with one line per attempt.
        const long long now = deps_.monotonicSeconds();
        if (now - lastTransportLogAt_ >= 60)
        {
            deps_.log("cannot reach the control plane (" + r.error + "); still trying");
            lastTransportLogAt_ = now;
        }
        return TickOutcome::Continue;
    }

    acceptsJobs_ = r.acceptsJobs;
    intervalSeconds_ = r.snapshotIntervalS;        // already clamped by the parser
    maybeTakeJob(r.assignment);
    return TickOutcome::Continue;
}

void Agent::run()
{
    stopped_ = false;
    while (!stopped_)
    {
        if (tick() == TickOutcome::AuthRejected)
        {
            // Idle rather than exit: systemd would restart us into the same rejection, and a node that
            // disappears entirely is harder to diagnose than one that is present and explaining itself.
            deps_.sleepFor(60);
            continue;
        }
        deps_.sleepFor(intervalSeconds_);
    }
}

}  // namespace brae::node
