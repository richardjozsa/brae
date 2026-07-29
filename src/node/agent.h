#pragma once
// The agent loop: heartbeat, take at most one job, report it durably, reconnect forever.
//
// Written as a `tick()` over injected dependencies rather than a `while(true)` with sleeps inside, so the whole
// lifecycle is testable at exact instants: "31 seconds of silence", "the result POST fails four times then
// succeeds", "a 401 after a month of working" are all assertions here, not timing races.
//
// Two structural decisions that the shape depends on:
//
//   A JOB MUST NOT BLOCK THE HEARTBEAT. A benchmark runs for minutes; the offline threshold is 30 seconds. If
//   the loop waited for the job, every running node would drop off /live and its job would be abandoned out
//   from under it. Execution therefore sits behind JobExecutor, which the loop polls.
//
//   A RESULT IS PERSISTED BEFORE IT IS SENT. The result is the only record that the work happened. If the
//   process dies between finishing and reporting, the run is wasted and the job sits until its deadline. So it
//   goes to disk first and is retried -- across restarts -- until the server acknowledges it.
#include "api_client.h"
#include "identity.h"
#include "jobs.h"
#include "runner.h"

#include <functional>
#include <memory>
#include <string>

namespace brae::node {

// Job execution, behind an interface: the loop polls rather than waits, and tests drive it without threads.
class JobExecutor
{
public:
    virtual ~JobExecutor() = default;
    virtual void start(const JobRequest& job) = 0;
    virtual bool running() const = 0;
    virtual int progressPercent() const = 0;
    // True exactly once, when a finished job's outcome is collected.
    virtual bool takeOutcome(JobOutcome& out) = 0;
};

// Runs jobs on a worker thread via runner.cpp. Production implementation.
std::unique_ptr<JobExecutor> makeThreadedExecutor(RunSettings settings);

struct AgentDeps
{
    ApiClient* api = nullptr;
    JobExecutor* executor = nullptr;

    std::function<std::vector<GpuState>()> probeGpuState;
    std::function<long long()> monotonicSeconds;      // never the wall clock: see tick()
    std::function<std::string()> nowIso8601;          // for timestamps only, never for scheduling
    std::function<void(int seconds)> sleepFor;
    std::function<void(const std::string&)> log;

    std::string pendingResultPath = "/var/lib/brae/pending_result.json";
    std::string agentVersion = "0.3.0";
};

enum class TickOutcome
{
    Continue,          // normal: keep looping
    AuthRejected,      // 401: stop looping, but stay alive so `brae node status` can explain
};

class Agent
{
public:
    Agent(Identity identity, AgentDeps deps);

    // One pass: report anything pending, send a snapshot, act on the reply. Returns how the loop should go on.
    TickOutcome tick();

    // Loops until the token is rejected or stop() is called. Sleeps via deps.sleepFor.
    void run();
    void stop() { stopped_ = true; }

    int snapshotIntervalSeconds() const { return intervalSeconds_; }
    bool busy() const;
    bool acceptsJobs() const { return acceptsJobs_; }
    bool hasPendingResult() const;

    // A result that could not be delivered is written here and retried, including after a restart.
    bool savePendingResult(const JobOutcome& o, const std::string& startedAt, const std::string& finishedAt);
    bool loadPendingResult(JobOutcome& o, std::string& startedAt, std::string& finishedAt) const;
    void clearPendingResult();

private:
    void reportPendingIfAny();
    void maybeTakeJob(const Assignment& a);
    void collectFinishedJob();

    Identity id_;
    AgentDeps deps_;

    int intervalSeconds_ = 10;
    bool acceptsJobs_ = true;
    bool stopped_ = false;

    std::string runningJobId_;
    std::string jobStartedAt_;

    int pendingAttempts_ = 0;
    long long nextPendingAttemptAt_ = 0;      // monotonic seconds
    long long lastTransportLogAt_ = -100000;  // rate-limits "cannot reach the server" to once a minute
};

}  // namespace brae::node
