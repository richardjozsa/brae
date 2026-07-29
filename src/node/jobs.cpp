#include "jobs.h"

#include <cctype>

namespace brae::node {

JobType jobTypeFromString(const std::string& s)
{
    if (s == "brae-benchmark") return JobType::BraeBenchmark;
    return JobType::Unknown;                 // anything else, including a type added server-side later
}

const char* jobTypeName(JobType t)
{
    switch (t)
    {
        case JobType::BraeBenchmark: return "brae-benchmark";
        case JobType::Unknown:       return "unknown";
    }
    return "unknown";
}

bool validSampleName(const std::string& s)
{
    // Mirrors benchmark.cuh's validateSampleName. Two implementations of one rule is a smell, but the
    // alternative is the agent forwarding a name that brae will reject after the process has already been
    // spawned -- and the agent must be able to refuse without consulting the solver.
    if (s.empty() || s.size() > 96) return false;
    if (s.find("..") != std::string::npos) return false;
    if (s.front() == '/' || s.back() == '/') return false;

    std::size_t start = 0;
    for (;;)
    {
        const std::size_t slash = s.find('/', start);
        const std::string part = s.substr(start, slash == std::string::npos ? std::string::npos : slash - start);
        if (part.empty()) return false;
        if (part.front() == '-' || part.front() == '.') return false;
        for (const char c : part)
            if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '-' || c == '_' || c == '.'))
                return false;
        if (slash == std::string::npos) return true;
        start = slash + 1;
    }
}

ArgvResult argvFor(const JobRequest& job, const std::string& braeBinary)
{
    ArgvResult r;
    if (braeBinary.empty())
    {
        r.reason = "no brae binary configured";
        return r;
    }

    switch (job.type)
    {
        case JobType::BraeBenchmark:
        {
            if (!validSampleName(job.sample))
            {
                r.reason = "rejected sample name '" + job.sample + "'";
                return r;
            }
            // Fixed shape. The sample is the ONLY value that varies, and it has just been validated as a
            // benchmark branch name -- not a path, not an option, not a URL.
            r.argv = {braeBinary, "benchmark", job.sample};
            r.ok = true;
            return r;
        }
        case JobType::Unknown:
        default:
            r.reason = "unknown job type; this agent runs only " + std::string(jobTypeName(JobType::BraeBenchmark));
            return r;
    }
}

}  // namespace brae::node
