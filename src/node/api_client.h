#pragma once
// The agent's half of the orchestrator contract (brae-cloud docs/02-api.md).
//
// All of the agent's protocol behaviour lives here, above the HttpClient interface, so it can be tested against
// a fake transport: every status code, a truncated body, a reply that is not JSON, a reply that is JSON but the
// wrong shape, an assignment for a job type this agent does not know. Those are the cases that decide whether a
// node quietly stops working at 3am, and none of them are convenient to produce from a real server.
//
// The rule that shapes every function below: a reply we cannot read is a reply we do not act on. There is no
// partial acceptance and no guessing at intent.
#include "http.h"
#include "jobs.h"

#include <string>
#include <vector>

namespace brae::node {

struct GpuIdentity
{
    int index = 0;
    std::string model;
    int vramMb = 0;
    std::string computeCapability;
    std::string driverVersion;
    std::string uuidHash;          // sha256 of the UUID; the raw value never leaves gpu_probe
};

struct GpuState
{
    int index = 0;
    int utilizationPercent = 0;
    int memoryUsedMb = 0;
    int memoryTotalMb = 0;
    int temperatureCelsius = 0;
};

struct RegisterRequest
{
    std::string displayName, operatingSystem, architecture, agentVersion, braeVersion;
    std::vector<GpuIdentity> gpus;
};

struct RegisterResult
{
    bool ok = false;
    std::string error;             // safe to log
    std::string nodeId;
    std::string nodeToken;         // secret
    int snapshotIntervalIdleS = 10;
    int snapshotIntervalRunningS = 5;
    std::string liveUrl;
};

struct SnapshotRequest
{
    std::string timestamp;         // RFC 3339 UTC
    std::string status;            // available | running | online
    bool acceptsJobs = true;
    std::string agentVersion;
    std::vector<GpuState> gpus;
    bool hasJob = false;
    std::string jobId;
    JobType jobType = JobType::Unknown;
    int jobProgressPercent = 0;
};

struct Assignment
{
    bool present = false;
    std::string jobId;
    JobType type = JobType::Unknown;
    std::string sample;
    std::string deadline;
};

struct SnapshotResult
{
    bool ok = false;
    std::string error;
    bool authRejected = false;     // 401: stop the loop, keep the process alive so `status` can explain
    bool acceptsJobs = true;
    int snapshotIntervalS = 10;    // already clamped
    Assignment assignment;
};

struct SimpleResult
{
    bool ok = false;
    std::string error;
    bool conflict = false;         // 409: someone else took the job, or it is already terminal -- not an error
    bool authRejected = false;
};

struct JobOutcome
{
    std::string jobId;
    bool success = false;
    double runtimeSeconds = 0;
    std::string resultJson;        // the brae-benchmark.json the run produced, verbatim
    std::string error;             // short class, e.g. "timeout", "solver_exit_1"
};

// Builders are separate from senders so a test can assert on the exact bytes that go on the wire.
std::string buildRegisterBody(const RegisterRequest& r);
std::string buildSnapshotBody(const SnapshotRequest& s);
std::string buildResultBody(const JobOutcome& o, const std::string& startedAt, const std::string& finishedAt);

// Parsers take the raw response and never throw.
RegisterResult parseRegisterResponse(const HttpResponse& res);
SnapshotResult parseSnapshotResponse(const HttpResponse& res);
SimpleResult parseSimpleResponse(const HttpResponse& res);

class ApiClient
{
public:
    ApiClient(HttpClient& http, std::string baseUrl) : http_(http), base_(std::move(baseUrl)) {}

    RegisterResult registerNode(const RegisterRequest& r, const std::string& enrollmentToken);
    SnapshotResult sendSnapshot(const std::string& nodeId, const std::string& token, const SnapshotRequest& s);
    SimpleResult acceptJob(const std::string& nodeId, const std::string& token, const std::string& jobId);
    SimpleResult reportResult(const std::string& nodeId, const std::string& token, const JobOutcome& o,
                              const std::string& startedAt, const std::string& finishedAt);
    SimpleResult unregisterNode(const std::string& nodeId, const std::string& token);

private:
    HttpClient& http_;
    std::string base_;
};

}  // namespace brae::node
