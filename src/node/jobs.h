#pragma once
// What the agent is willing to run, and how a job becomes a process.
//
// THIS IS THE SECURITY BOUNDARY (docs/10-security.md, rule 1). The question it exists to answer is: if the
// orchestrator were compromised tomorrow, what could it do to a contributor's machine? The answer has to stay
// "make it run a Brae benchmark it did not ask for", and that holds only because:
//
//   * the job TYPE is a fixed enum -- an unknown type is refused, never passed through;
//   * the argv is built HERE, from constants, with only validated parameters substituted;
//   * nothing from the server is ever used as a command, a path, a shell string, or a URL to fetch.
//
// If a future feature seems to need "just a small string passed through to the command line", that is the
// moment to stop and redesign, not to widen this file.
#include <string>
#include <vector>

namespace brae::node {

enum class JobType
{
    Unknown,
    BraeBenchmark,      // "brae-benchmark": run a published sample and report the metrics
};

JobType jobTypeFromString(const std::string& s);
const char* jobTypeName(JobType t);

struct JobRequest
{
    std::string jobId;
    JobType type = JobType::Unknown;
    std::string sample;          // a brae-bench branch name; validated, never used as a path
    int timeoutSeconds = 0;
};

struct ArgvResult
{
    bool ok = false;
    std::string reason;                 // why it was refused, for the log and the failure report
    std::vector<std::string> argv;      // argv[0] is the brae binary
};

// A sample name reaches brae as a git branch and a cache directory. brae validates it again on its side
// (solver_dispatch's sibling in benchmark.cuh), but the agent must not hand on something it would not accept
// itself: grouped names like "pimplefoam/pitzDaily-1M" are fine, traversal and option leaders are not.
bool validSampleName(const std::string& s);

// The ONLY place a job turns into a process. Adding a job type means adding a case here, deliberately.
ArgvResult argvFor(const JobRequest& job, const std::string& braeBinary);

}  // namespace brae::node
