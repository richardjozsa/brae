#pragma once
// Running a job, and the rules around doing so.
//
// jobs.h decides WHAT may run; this decides HOW. Both halves are the security boundary (docs/10-security.md):
//
//   * posix_spawn with an explicit argv. Never system(), never popen(), never /bin/sh. There is no string
//     anywhere in this file that a shell will ever see.
//   * a hard deadline, enforced with SIGTERM then SIGKILL. A job that hangs must not hold the node forever.
//   * the child runs in its own scratch directory, which is created 0700 and removed afterwards.
//   * stdout is captured (the metrics), stderr is scanned for progress. Neither is uploaded: the result
//     contract carries numbers, not logs.
#include "api_client.h"
#include "jobs.h"

#include <functional>
#include <string>

namespace brae::node {

struct RunSettings
{
    std::string braeBinary;          // absolute path to brae
    std::string scratchRoot = "/var/lib/brae/jobs";
    int defaultTimeoutSeconds = 1800;
    int killGraceSeconds = 10;       // between SIGTERM and SIGKILL
};

// Called as the job reports progress, so the snapshot loop can put it in the next heartbeat. Never called
// after run() returns.
using ProgressFn = std::function<void(int percent)>;

// Runs the job to completion, a failure, or its deadline. Never throws.
JobOutcome runJob(const JobRequest& job, const RunSettings& settings, const ProgressFn& onProgress);

// Progress lines look like "progress: 64" on stderr. Exposed for testing, and because the format is a contract
// with `brae benchmark` rather than an implementation detail.
bool parseProgressLine(const std::string& line, int& percent);

// Where a job's scratch directory goes. Kept out of runJob so the path rule can be asserted directly: a job id
// arrives from the server and must never be able to escape the scratch root.
bool jobScratchDir(const RunSettings& s, const std::string& jobId, std::string& out);

}  // namespace brae::node
